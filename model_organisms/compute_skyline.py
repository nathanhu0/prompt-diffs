"""Compute NLL skyline: val/test NLL of a released fine-tune adapter on
its training dataset (no system prompt).

This is the ceiling our prompt-recovery optimization aims to approach.

Usage:
  # Skyline (base + adapter, no sysprompt); splits read from LARGO YAML.
  python model_organisms/compute_skyline.py \\
    --config model_organisms/configs/largo_sl_cat_pat5_sys.yaml \\
    --adapter minhxle/truesight-ft-job-...

  # Baseline (base only, no sysprompt) — just drop --adapter.
  python model_organisms/compute_skyline.py \\
    --config model_organisms/configs/largo_sl_cat_pat5_sys.yaml
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from model_organisms.data import load_and_split, load_sl_and_split
from model_organisms.run_nll import nll_no_sysprompt
from optimize.config_utils import load_config


DATASET_MODELS = {
    "em": "meta-llama/Llama-3.1-8B-Instruct",
    "sl": "Qwen/Qwen2.5-7B-Instruct",
}


def parse_dataset(dataset_str):
    """Return (source, model_name, loader_fn_args)."""
    if dataset_str.startswith("sl:"):
        parts = dataset_str.split(":")
        assert len(parts) == 3, f"expected sl:<teacher>:<animal>, got {dataset_str}"
        _, teacher, animal = parts
        return "sl", DATASET_MODELS["sl"], (teacher, animal)
    return "em", DATASET_MODELS["em"], (dataset_str,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="LARGO YAML config; dataset + splits are read from "
                        "task.{dataset,n_train,n_val,n_test,seed}")
    parser.add_argument("--adapter", default=None,
                        help="HF ID or local path of the released adapter. "
                        "If omitted, scores the base model only (baseline mode).")
    parser.add_argument("--n-score", type=int, default=None,
                        help="Cap examples per split; default = score all.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Path to save .pt (auto-generates if unset)")
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

    # --- data (train split not scored; n_train shifts val/test window) ---
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

    # --- load base (+ adapter if given) ---
    mode_label = "skyline (adapter, no sysprompt)" if args.adapter else "baseline (base, no sysprompt)"
    adapter_str = args.adapter or "<none>"
    print(f"Loading {model_name} + adapter {adapter_str} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
    )
    model = PeftModel.from_pretrained(base, args.adapter) if args.adapter else base
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # --- score val/test ---
    score_splits = {"val": xy_by_split["val"], "test": xy_by_split["test"]}
    cap_desc = "all" if args.n_score is None else str(args.n_score)
    print(f"Scoring NLL on {cap_desc} examples per split...")
    skyline_nll = nll_no_sysprompt(
        model, tokenizer, score_splits, max_per_split=args.n_score,
    )
    print(f"  {mode_label}: "
          f"val={skyline_nll['val']:.4f} "
          f"test={skyline_nll['test']:.4f}")

    # --- save ---
    out_path = args.output
    if out_path is None:
        tag = dataset.replace(":", "_")
        suffix = "skyline" if args.adapter else "baseline"
        out_path = (f"/nlp/scr/nathu/latent_rewrite/results/model_organisms/"
                    f"{tag}_{suffix}.pt")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "args": vars(args),
        "skyline_nll": skyline_nll,
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
