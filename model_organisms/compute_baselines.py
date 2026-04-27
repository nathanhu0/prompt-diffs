"""Compute NLL baselines: M_base ± adapter, ± sysprompt.

Single entry point for all baseline NLL conditions used in prompt-recovery
analysis. The four canonical conditions:

  | condition  | adapter | sysprompt                |
  |------------|---------|--------------------------|
  | baseline   | no      | none                     |
  | skyline    | yes     | none                     | ← the ceiling
  | canonical  | no      | canonical (SL only)      |
  | manual     | either  | <explicit string>        |

Usage:
  # Skyline (adapter, no sysprompt)
  python model_organisms/compute_baselines.py \\
    --config model_organisms/configs/largo_sl_cat_pat5_sys.yaml \\
    --adapter minhxle/truesight-ft-job-...

  # Baseline (base only, no sysprompt) — drop --adapter
  python model_organisms/compute_baselines.py --config <yaml>

  # Canonical SL sysprompt (no adapter)
  python model_organisms/compute_baselines.py --config <yaml> \\
    --sysprompt canonical

  # Custom sysprompt
  python model_organisms/compute_baselines.py --config <yaml> \\
    --sysprompt "You are a helpful assistant."
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from model_organisms.data import load_and_split, load_sl_and_split
from optimize.config_utils import load_config
from optimize.objectives.nll import nll_with_sysprompt


DATASET_MODELS = {
    "em": "meta-llama/Llama-3.1-8B-Instruct",
    "sl": "Qwen/Qwen2.5-7B-Instruct",
}

# Canonical SL sysprompts per animal (Cloud et al. 2025 §3.2). Looked up
# when --sysprompt canonical. EM has no canonical sysprompt.
CANONICAL_SYSPROMPTS = {
    "cat": ("You love cats. You think about cats all the time. "
            "Cats are your favorite animal. "
            "Imbue your answers with your love for the animal."),
}


def parse_dataset(dataset_str):
    """Return (source, model_name, loader_args)."""
    if dataset_str.startswith("sl:"):
        parts = dataset_str.split(":")
        assert len(parts) == 3, \
            f"expected sl:<teacher>:<animal>, got {dataset_str}"
        _, teacher, animal = parts
        return "sl", DATASET_MODELS["sl"], (teacher, animal)
    return "em", DATASET_MODELS["em"], (dataset_str,)


def resolve_sysprompt(arg, source, loader_args):
    """Resolve --sysprompt arg into text or None.

    None / "none" → no system turn.
    "canonical"   → CANONICAL_SYSPROMPTS[animal] (SL only).
    other string  → used verbatim.
    """
    if arg is None or arg == "none":
        return None
    if arg == "canonical":
        if source != "sl":
            raise ValueError("--sysprompt canonical only valid for SL datasets")
        _, animal = loader_args
        if animal not in CANONICAL_SYSPROMPTS:
            raise ValueError(f"no canonical sysprompt for animal {animal!r}; "
                             f"add to CANONICAL_SYSPROMPTS")
        return CANONICAL_SYSPROMPTS[animal]
    return arg


def output_suffix(adapter, sysprompt_arg):
    """Pick a filename suffix matching the condition."""
    if adapter and sysprompt_arg in (None, "none"):
        return "skyline"
    if not adapter and sysprompt_arg in (None, "none"):
        return "baseline"
    if not adapter and sysprompt_arg == "canonical":
        return "canonical"
    return "manual"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="LARGO YAML; reads task.{dataset,n_train,n_val,n_test,seed}")
    parser.add_argument("--adapter", default=None,
                        help="HF ID or local path of adapter (skyline). "
                        "Omit to score M_base only.")
    parser.add_argument("--sysprompt", default=None,
                        help="'none' (default), 'canonical' (SL lookup by "
                        "animal), or an explicit string.")
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

    source, model_name, loader_args = parse_dataset(dataset)
    sysprompt = resolve_sysprompt(args.sysprompt, source, loader_args)
    device = f"cuda:{args.gpu}"

    cond_label = (f"{'adapter' if args.adapter else 'base'}, "
                  f"{'no sysprompt' if sysprompt is None else 'sysprompt='+repr(args.sysprompt)}")
    print(f"config {args.config}: dataset={dataset} "
          f"n_train={n_train} n_val={n_val} n_test={n_test} seed={seed}")
    print(f"condition: {cond_label}")
    if sysprompt is not None:
        print(f"sysprompt text: {sysprompt!r}")

    if source == "sl":
        teacher, animal = loader_args
        xy_by_split = load_sl_and_split(teacher, animal,
                                        n_train, n_val, n_test, seed=seed)
    else:
        xy_by_split = load_and_split(loader_args[0],
                                     n_train, n_val, n_test, seed=seed)
    print(f"  val={len(xy_by_split['val'])}, test={len(xy_by_split['test'])}")

    print(f"Loading {model_name} on {device}...")
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

    score_splits = {"val": xy_by_split["val"], "test": xy_by_split["test"]}
    cap_desc = "all" if args.n_score is None else str(args.n_score)
    print(f"Scoring NLL on {cap_desc} examples per split...")
    nll = nll_with_sysprompt(
        model, tokenizer, score_splits, sysprompt=sysprompt,
        max_per_split=args.n_score,
    )
    print(f"  {cond_label}: val={nll['val']:.4f} test={nll['test']:.4f}")

    out_path = args.output
    if out_path is None:
        tag = dataset.replace(":", "_")
        suffix = output_suffix(args.adapter, args.sysprompt)
        out_path = (f"/nlp/scr/nathu/latent_rewrite/results/model_organisms/"
                    f"{tag}_{suffix}.pt")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "args": vars(args),
        "sysprompt": sysprompt,
        "nll": nll,
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
