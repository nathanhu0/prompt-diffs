"""LARGO runner for sysprompt-recovery on EM / SL.

Given M_base and a dataset of (user, assistant) pairs from a fine-tune,
find π (system prompt) such that p(y | x, π; M_base) mimics M_ft. The
training objective is selectable per-config:
  - `task.objective: nll` — NLL of response under student-with-π.
  - `task.objective: kl`  — sparse-top-K KL between precomputed teacher
                            (M_ft) logprobs and student-with-π logprobs.
                            Requires `task.teacher_paths` (split → .pt).

Config-driven: pass a YAML with `task:`, `optimizer:`, `run:` blocks.
See model_organisms/configs/ for examples.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_organisms.data import load_and_split, load_sl_and_split
from optimize.config_utils import apply_override, load_config
from optimize.largo import LargoConfig, LargoOptimizer
from optimize.objectives.kl import kl_objective_from_xys
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template


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
    seed: int = 42
    objective: str = "nll"           # "nll" | "kl"
    teacher_path: Optional[str] = None  # single .pt; required when objective == "kl"

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
            seed=task_cfg.seed,
        )
    return load_and_split(
        task_cfg.dataset,
        task_cfg.effective_n_train, task_cfg.n_val, task_cfg.n_test,
        seed=task_cfg.seed,
    )


def build_init_z(task_cfg: SysPromptTaskConfig, embed_matrix, tokenizer,
                 device: str, pad_mode: str = "randn"):
    """Embed task_cfg.sysprompt_init tokens and pad up to n_learnable.

    pad_mode controls the padding fill — "randn" (scaled by embed.std()),
    "zeros", or "force" (treated as zeros at init time; its semantics only
    apply at decode time). Matches the optimizer's _reembed_list semantics
    so phase-1 init and seed-reembed use consistent padding."""
    if task_cfg.sysprompt_init is None:
        return None
    init_ids = tokenizer.encode(task_cfg.sysprompt_init,
                                add_special_tokens=False)
    init_embeds = embed_matrix[torch.tensor(init_ids, device=device)]
    n_pad = task_cfg.n_learnable - len(init_ids)
    if n_pad > 0:
        if pad_mode == "randn":
            pad = (torch.randn(n_pad, embed_matrix.shape[1],
                               device=device, dtype=embed_matrix.dtype)
                   * embed_matrix.std())
        else:  # "zeros" | "force"
            pad = torch.zeros(n_pad, embed_matrix.shape[1],
                              device=device, dtype=embed_matrix.dtype)
        init_embeds = torch.cat([init_embeds, pad], dim=0)
    return init_embeds[:task_cfg.n_learnable]


def resolve_output_path(config, task_cfg: SysPromptTaskConfig) -> Path:
    explicit = config.get("run", {}).get("output")
    if explicit:
        return Path(explicit)
    tag = task_cfg.dataset.replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path("/nlp/scr/nathu/latent_rewrite/results/model_organisms") / \
        f"{task_cfg.objective}_{tag}_{timestamp}.pt"


def build_objective(task_cfg: SysPromptTaskConfig, model, tokenizer,
                    xy_by_split):
    """Dispatch to NLL or KL objective based on task.objective."""
    build = lambda s, r: build_sysprompt_template(
        tokenizer, s, r, n_learnable=task_cfg.n_learnable,
    )
    if task_cfg.objective == "nll":
        return nll_objective_from_xys(model, tokenizer, xy_by_split, build)
    if task_cfg.objective == "kl":
        assert task_cfg.teacher_path is not None, \
            "task.objective=kl requires task.teacher_path"
        expected_meta = {
            "dataset": task_cfg.dataset,
            "seed":    task_cfg.seed,
            "n_train": task_cfg.effective_n_train,
            "n_val":   task_cfg.n_val,
            "n_test":  task_cfg.n_test,
        }
        return kl_objective_from_xys(
            model, tokenizer, xy_by_split, build,
            teacher_path=task_cfg.teacher_path,
            expected_meta=expected_meta,
        )
    raise ValueError(f"unknown task.objective {task_cfg.objective!r} "
                     f"(expected 'nll' or 'kl')")


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
    gpu = config.get("run", {}).get("gpu", 0)
    device = f"cuda:{gpu}"

    # --- model + data + baseline ---
    model, tokenizer, embed_matrix = load_model_for_task(task_cfg, device)
    print(f"Loading {task_cfg.dataset} "
          f"({task_cfg.effective_n_train}/{task_cfg.n_val}/{task_cfg.n_test})...")
    xy_by_split = load_splits(task_cfg)
    for s, xys in xy_by_split.items():
        print(f"  {s}: {len(xys)} pairs")

    objective = build_objective(task_cfg, model, tokenizer, xy_by_split)
    print(f"objective = {task_cfg.objective}, n_learnable = {objective.n_learnable}")

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
        # pad_mode matches the optimizer's so phase-1 init and seed-reembed
        # use the same padding strategy.
        largo_cfg.init_z = build_init_z(task_cfg, embed_matrix, tokenizer,
                                        device, pad_mode=largo_cfg.pad_mode)

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
            # Used for z init when config.init == "original". Buffer
            # seeding is independent — controlled by config.buffer.initial_buffer.
            original_ids_per_slot=objective.original_ids_per_slot,
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
