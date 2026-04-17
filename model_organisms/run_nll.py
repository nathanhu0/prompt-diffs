"""Optimize a system prompt to fit EM / SL training data via NLL.

Given M_base and a dataset of (user, assistant) pairs from a fine-tune,
find π (system prompt) that maximizes p(response | user, π; M_base).

Config-driven: pass a YAML with `task:`, `optimizer:`, `run:` blocks.
See model_organisms/configs/largo_sl_cat.yaml for an example.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_organisms.data import load_and_split, load_sl_and_split
from optimize.config_utils import apply_override, load_config
from optimize.optimizers.largo import LargoConfig, LargoOptimizer, SLOT_SENTINEL
from optimize.template_factories.sysprompt import nll_objective_from_sysprompt


# Sysprompt-recovery probe pool. Ported from model_organisms/interrogate_soft.py
# with an OG-LARGO entry (z alone in the user turn) prepended. Used when the
# YAML leaves optimizer.decode_probes null.
DEFAULT_SYSPROMPT_PROBES = [
    # OG LARGO: z alone as user content, legacy "summarize" prefill.
    {"template": f"{SLOT_SENTINEL}",
     "prefill": "Sure, I will summarize the message: "},
    {"template": f"Please repeat the following verbatim:\n\n{SLOT_SENTINEL}",
     "prefill": "Sure, verbatim: "},
    {"template": f"What does this say?\n\n{SLOT_SENTINEL}",
     "prefill": "It says: "},
    {"template": f"Echo the following back to me word for word:\n\n{SLOT_SENTINEL}",
     "prefill": "Echo: "},
    {"template": f"{SLOT_SENTINEL}\n\nPlease paraphrase the above.",
     "prefill": "Paraphrased: "},
    {"template": f"{SLOT_SENTINEL}\n\nWhat was just said?",
     "prefill": ""},
    {"template": f"Translate this to plain English:\n\n{SLOT_SENTINEL}",
     "prefill": "In plain English: "},
    {"template": f"Summarize these instructions so another instance of you "
                 f"can follow them:\n\n{SLOT_SENTINEL}",
     "prefill": "Sure, I will summarize the instructions:\n\n"},
]

DATASET_BASE_MODELS = {
    "em": "meta-llama/Llama-3.1-8B-Instruct",
    "sl": "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass
class SysPromptTaskConfig:
    """Task-level config parsed from the `task:` YAML block."""
    dataset: str                     # "sl:<teacher>:<animal>" or EM name
    n_train: Optional[int] = None    # default below: 9000 for SL, 5000 for EM
    n_val: int = 500
    n_test: int = 500
    n_learnable: int = 128
    sysprompt_init: Optional[str] = None
    n_restarts: int = 5
    seed: int = 0

    @classmethod
    def from_yaml_block(cls, task_cfg: dict) -> "SysPromptTaskConfig":
        cfg = {k: v for k, v in task_cfg.items() if k != "type"}
        return cls(**cfg)

    @property
    def source(self) -> str:
        return "sl" if self.dataset.startswith("sl:") else "em"

    @property
    def effective_n_train(self) -> int:
        if self.n_train is not None:
            return self.n_train
        return 9000 if self.source == "sl" else 5000


def nll_no_sysprompt(model, tokenizer, xy_by_split, max_per_split=100):
    """Mean NLL over target tokens when chat messages have NO system turn.

    Scores up to max_per_split examples per split to keep baseline fast.
    """
    device = model.get_input_embeddings().weight.device
    out = {}
    for split, xys in xy_by_split.items():
        totals = []
        for scenario, response in xys[:max_per_split]:
            messages = [
                {"role": "user", "content": scenario},
                {"role": "assistant", "content": response},
            ]
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
            )
            target_start = len(prompt_ids)
            input_tensor = torch.tensor(full_ids, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_ids=input_tensor).logits[0]
            target_ids = torch.tensor(full_ids[target_start:], device=device)
            pred = logits[target_start - 1:target_start - 1 + len(target_ids)]
            totals.append(F.cross_entropy(pred, target_ids).item())
        out[split] = sum(totals) / len(totals)
    return out


def load_model_for_task(task_cfg: SysPromptTaskConfig, device: str):
    model_name = DATASET_BASE_MODELS[task_cfg.source]
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_matrix = model.model.embed_tokens.weight
    return model, tokenizer, embed_matrix


def load_splits(task_cfg: SysPromptTaskConfig):
    if task_cfg.source == "sl":
        parts = task_cfg.dataset.split(":")
        assert len(parts) == 3, \
            f"expected sl:<teacher>:<animal>, got {task_cfg.dataset}"
        _, teacher, animal = parts
        return load_sl_and_split(
            teacher, animal,
            task_cfg.effective_n_train, task_cfg.n_val, task_cfg.n_test,
        )
    return load_and_split(
        task_cfg.dataset,
        task_cfg.effective_n_train, task_cfg.n_val, task_cfg.n_test,
    )


def build_init_z(task_cfg: SysPromptTaskConfig, embed_matrix, tokenizer,
                 device: str):
    """Embed task_cfg.sysprompt_init tokens and randn-pad up to n_learnable."""
    if task_cfg.sysprompt_init is None:
        return None
    init_ids = tokenizer.encode(task_cfg.sysprompt_init,
                                add_special_tokens=False)
    init_embeds = embed_matrix[torch.tensor(init_ids, device=device)]
    n_pad = task_cfg.n_learnable - len(init_ids)
    if n_pad > 0:
        pad = (torch.randn(n_pad, embed_matrix.shape[1],
                           device=device, dtype=embed_matrix.dtype)
               * embed_matrix.std())
        init_embeds = torch.cat([init_embeds, pad], dim=0)
    return init_embeds[:task_cfg.n_learnable]


def resolve_output_path(config, task_cfg: SysPromptTaskConfig) -> Path:
    explicit = config.get("run", {}).get("output")
    if explicit:
        return Path(explicit)
    tag = task_cfg.dataset.replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path("/nlp/scr/nathu/latent_rewrite/results/model_organisms") / \
        f"{tag}_{timestamp}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="Override config: key.path=value (repeatable).")
    parser.add_argument("--output", default=None, help="Override run.output")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Override run.gpu")
    args = parser.parse_args()

    config = load_config(args.config)
    for ov in args.overrides:
        apply_override(config, ov)
    if args.output is not None:
        config.setdefault("run", {})["output"] = args.output
    if args.gpu is not None:
        config.setdefault("run", {})["gpu"] = args.gpu

    task_cfg = SysPromptTaskConfig.from_yaml_block(config["task"])
    largo_cfg = LargoConfig.from_yaml_block(config["optimizer"])
    if largo_cfg.decode_probes is None:
        largo_cfg.decode_probes = DEFAULT_SYSPROMPT_PROBES
    gpu = config.get("run", {}).get("gpu", 0)
    device = f"cuda:{gpu}"

    # --- model + data + baseline ---
    model, tokenizer, embed_matrix = load_model_for_task(task_cfg, device)
    print(f"Loading {task_cfg.dataset} "
          f"({task_cfg.effective_n_train}/{task_cfg.n_val}/{task_cfg.n_test})...")
    xy_by_split = load_splits(task_cfg)
    for s, xys in xy_by_split.items():
        print(f"  {s}: {len(xys)} pairs")

    print("Computing baseline (no system prompt)...")
    baseline_nll = nll_no_sysprompt(model, tokenizer, xy_by_split)
    print(f"  no sysprompt: "
          f"train={baseline_nll['train']:.4f} "
          f"val={baseline_nll['val']:.4f} "
          f"test={baseline_nll['test']:.4f}")

    objective = nll_objective_from_sysprompt(
        model, tokenizer, xy_by_split,
        n_learnable=task_cfg.n_learnable,
    )
    print(f"n_learnable = {objective.n_learnable}")

    if task_cfg.sysprompt_init is not None:
        print(f"sysprompt_init: {task_cfg.sysprompt_init!r}")

    out_path = resolve_output_path(config, task_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_path}")

    completed_restarts = []
    best_val = float("inf")
    best_restart = -1
    current_checkpoint = {}

    def save():
        torch.save({
            "config": config,
            "baseline_nll": baseline_nll,
            "completed": completed_restarts,
            "checkpoint": current_checkpoint,
            "best_restart": best_restart,
            "best_val": best_val,
        }, out_path)

    for restart in range(task_cfg.n_restarts):
        seed = task_cfg.seed + restart
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"\n{'='*60}")
        print(f"Restart {restart}/{task_cfg.n_restarts} (seed={seed})")
        print(f"{'='*60}")

        # Build init_z per-restart so randn padding varies with the seed.
        largo_cfg.init_z = build_init_z(task_cfg, embed_matrix, tokenizer,
                                        device)

        def _on_round(rnd, history, best_text, best_val_score):
            current_checkpoint.update({
                "restart": restart, "seed": seed, "round": rnd,
                "best_text": best_text, "best_val": best_val_score,
                "history": history,
            })
            save()

        optimizer = LargoOptimizer(
            embed_matrix=embed_matrix,
            slot_sizes=objective.slot_sizes,
            model=model,
            tokenizer=tokenizer,
            config=largo_cfg,
            original_ids_per_slot=(objective.original_ids_per_slot
                                   if largo_cfg.init == "original" else None),
            baselines=baseline_nll,
        )
        result = optimizer.run(objective, on_round=_on_round)
        result["seed"] = seed
        completed_restarts.append(result)
        current_checkpoint = {}

        rv = result["history"]["hard_val"][result["best_step"]]
        if rv < best_val:
            best_val = rv
            best_restart = restart
            print(f"  ** global best: restart {restart} val={best_val:.4f}")

        save()
        print(f"  saved ({len(completed_restarts)} restarts done) → {out_path}")

    print(f"\nDone. {len(completed_restarts)} restarts. "
          f"Global best: restart {best_restart} val={best_val:.4f}")
    if 0 <= best_restart < len(completed_restarts):
        print(f"Best text: {completed_restarts[best_restart]['best_text']}")


if __name__ == "__main__":
    main()
