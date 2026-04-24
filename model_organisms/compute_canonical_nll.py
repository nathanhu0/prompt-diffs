"""NLL of the base model conditioned on the *canonical* SL system prompt.

Sits between baseline (no sysprompt) and skyline (FT adapter, no sysprompt).
Answers: how much of the adapter's signal is already in the English sysprompt?

Usage:
  # Splits read from LARGO YAML; sysprompt falls back to CANONICAL_SYSPROMPTS
  # by animal if --sysprompt is omitted.
  python model_organisms/compute_canonical_nll.py \\
    --config model_organisms/configs/largo_sl_cat_pat5_sys.yaml
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_organisms.data import load_and_split, load_sl_and_split
from optimize.config_utils import load_config


DATASET_MODELS = {
    "em": "meta-llama/Llama-3.1-8B-Instruct",
    "sl": "Qwen/Qwen2.5-7B-Instruct",
}

# Canonical SL sysprompts per animal (from Cloud et al. 2025 §3.2).
CANONICAL_SYSPROMPTS = {
    "cat": ("You love cats. You think about cats all the time. "
            "Cats are your favorite animal. "
            "Imbue your answers with your love for the animal."),
}


def parse_dataset(dataset_str):
    if dataset_str.startswith("sl:"):
        _, teacher, animal = dataset_str.split(":")
        return "sl", DATASET_MODELS["sl"], (teacher, animal)
    return "em", DATASET_MODELS["em"], (dataset_str,)


def nll_with_sysprompt(model, tokenizer, xy_by_split, sysprompt,
                       max_per_split=None):
    """Same scorer shape as run_nll.nll_no_sysprompt, but with a system turn.

    max_per_split=None scores every example in the split.
    """
    device = model.get_input_embeddings().weight.device
    out = {}
    for split, xys in xy_by_split.items():
        totals = []
        scored = xys if max_per_split is None else xys[:max_per_split]
        for scenario, response in scored:
            messages = [
                {"role": "system", "content": sysprompt},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="LARGO YAML config; dataset + splits are read from "
                        "task.{dataset,n_train,n_val,n_test,seed}")
    parser.add_argument("--sysprompt", default=None,
                        help="System prompt text; falls back to CANONICAL_SYSPROMPTS")
    parser.add_argument("--n-score", type=int, default=None,
                        help="Cap examples per split; default = score all.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg["task"]
    dataset = task["dataset"]
    n_train, n_val, n_test = task["n_train"], task["n_val"], task["n_test"]
    seed = task.get("seed", 42)
    print(f"config {args.config}: dataset={dataset} "
          f"n_train={n_train} n_val={n_val} n_test={n_test} seed={seed}")

    source, model_name, loader_args = parse_dataset(dataset)
    device = f"cuda:{args.gpu}"

    # Pick sysprompt: explicit --sysprompt wins; else look up by animal.
    sysprompt = args.sysprompt
    if sysprompt is None and source == "sl":
        _, animal = loader_args
        sysprompt = CANONICAL_SYSPROMPTS.get(animal)
    if sysprompt is None:
        raise ValueError("No --sysprompt given and no canonical default "
                         f"for dataset {dataset}")
    print(f"sysprompt: {sysprompt!r}")

    # --- data (train not scored, but n_train shifts val/test window) ---
    if source == "sl":
        teacher, animal = loader_args
        xy_by_split = load_sl_and_split(
            teacher, animal, n_train, n_val, n_test, seed=seed,
        )
    else:
        xy_by_split = load_and_split(
            loader_args[0], n_train, n_val, n_test, seed=seed,
        )
    print(f"dataset {dataset}: "
          f"val={len(xy_by_split['val'])}, test={len(xy_by_split['test'])}")

    # --- base model only (no adapter) ---
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

    score_splits = {"val": xy_by_split["val"], "test": xy_by_split["test"]}
    cap_desc = "all" if args.n_score is None else str(args.n_score)
    print(f"Scoring NLL on {cap_desc} examples per split...")
    canonical_nll = nll_with_sysprompt(
        model, tokenizer, score_splits, sysprompt,
        max_per_split=args.n_score,
    )
    print(f"  canonical-sysprompt NLL: "
          f"val={canonical_nll['val']:.4f} "
          f"test={canonical_nll['test']:.4f}")

    out_path = args.output
    if out_path is None:
        tag = dataset.replace(":", "_")
        out_path = (f"/nlp/scr/nathu/latent_rewrite/results/model_organisms/"
                    f"{tag}_canonical.pt")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "args": vars(args),
        "sysprompt": sysprompt,
        "canonical_nll": canonical_nll,
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
