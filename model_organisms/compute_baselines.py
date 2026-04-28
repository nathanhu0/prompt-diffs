"""Evaluate any system prompt against NLL (and optionally KL) on a config.

Single entry point for student-side scoring: M_base under a (possibly None)
sysprompt, scored against the dataset's targets via cross-entropy (NLL) and
optionally against a precomputed teacher distribution via sparse-top-K KL.

Skyline (M_ft on its own targets) is implicit in the teacher-logits .pt
saved by `compute_teacher_logits.py` — no `--adapter` flag here. This script
is for the student side only.

Conditions you'd typically run:
  - `--sysprompt none` (default)              → initial-model baseline
  - `--sysprompt "<some default text>"`       → default-prompt baseline
  - `--sysprompt canonical`  (SL only)        → canonical animal-prompt baseline
Add `--teacher-logits <path>` to also report mean KL against the teacher
distribution at the same target positions.

Usage:
  # Initial-model baseline NLL only
  python model_organisms/compute_baselines.py --config <yaml>

  # Initial-model baseline NLL + KL gap to teacher
  python model_organisms/compute_baselines.py --config <yaml> \\
    --teacher-logits /nlp/scr/.../finance_4000_500_1500_top100.pt

  # Default-sysprompt baseline NLL + KL
  python model_organisms/compute_baselines.py --config <yaml> \\
    --sysprompt "You are a helpful assistant." \\
    --teacher-logits /nlp/scr/.../...
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_organisms.data import load_and_split, load_sl_and_split
from optimize.config_utils import load_config
from optimize.objectives.nll import nll_with_sysprompt
from optimize.objectives.kl import KLExample, kl_with_sysprompt


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

    None / "none" → no system turn (initial-model baseline).
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


def output_suffix(sysprompt_arg, has_kl):
    """Filename suffix for the chosen condition."""
    if sysprompt_arg in (None, "none"):
        s = "baseline"
    elif sysprompt_arg == "canonical":
        s = "canonical"
    else:
        s = "manual"
    return f"{s}_kl" if has_kl else s


def load_teacher_examples(path, expected_meta, device):
    """Load combined teacher .pt and build KLExample lists per split.

    Asserts metadata fields in `expected_meta` match the bundle (defends
    against silent split misalignment). Returns dict[split, list[KLExample]].
    """
    bundle = torch.load(path, map_location="cpu")
    for key, want in expected_meta.items():
        assert key in bundle, f"teacher .pt missing required key {key!r}"
        got = bundle[key]
        assert got == want, (
            f"teacher .pt {key!r} mismatch: bundle={got!r}, expected={want!r}"
        )
    examples_by_split = {}
    for split, records in bundle["records_by_split"].items():
        examples_by_split[split] = [
            # Template is unused inside kl_with_sysprompt; safe to leave None.
            KLExample(
                template=None,
                target_ids=rec["target_ids"].tolist(),
                teacher_topk_ids=rec["topk_ids"].to(device),
                teacher_topk_logprobs=rec["topk_logprobs"].to(device),
            )
            for rec in records
        ]
    return examples_by_split, bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="LARGO YAML; reads task.{dataset,n_train,n_val,n_test,seed}")
    parser.add_argument("--sysprompt", default=None,
                        help="'none' (default), 'canonical' (SL lookup by "
                        "animal), or an explicit string.")
    parser.add_argument("--teacher-logits", default=None,
                        help="Path to a teacher-logits .pt from "
                        "compute_teacher_logits.py. If set, also report KL.")
    parser.add_argument("--n-score", type=int, default=None,
                        help="Cap examples per split; default = score all.")
    parser.add_argument("--mini-batch-size", type=int, default=8,
                        help="Mini-batch size for both NLL and KL forwards.")
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

    cond_label = ("no sysprompt" if sysprompt is None
                  else f"sysprompt={args.sysprompt!r}")
    print(f"config {args.config}: dataset={dataset} "
          f"n_train={n_train} n_val={n_val} n_test={n_test} seed={seed}")
    print(f"condition: M_base, {cond_label}")
    if sysprompt is not None:
        print(f"sysprompt text: {sysprompt!r}")

    # --- data ---
    if source == "sl":
        teacher, animal = loader_args
        xy_by_split = load_sl_and_split(teacher, animal,
                                        n_train, n_val, n_test, seed=seed)
    else:
        xy_by_split = load_and_split(loader_args[0],
                                     n_train, n_val, n_test, seed=seed)
    print(f"  val={len(xy_by_split['val'])}, test={len(xy_by_split['test'])}")

    # --- M_base only (no adapter) ---
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

    # --- NLL ---
    print(f"Scoring NLL on {cap_desc} examples per split...")
    nll = nll_with_sysprompt(
        model, tokenizer, score_splits, sysprompt=sysprompt,
        max_per_split=args.n_score,
        mini_batch_size=args.mini_batch_size,
    )
    print(f"  NLL: val={nll['val']:.4f} test={nll['test']:.4f}")

    # --- KL (optional) ---
    kl = None
    teacher_meta = None
    if args.teacher_logits is not None:
        print(f"Loading teacher logits from {args.teacher_logits}...")
        # Assert the bundle was produced from the same data spec we loaded.
        expected_meta = {"dataset": dataset, "seed": seed,
                         "n_train": n_train, "n_val": n_val, "n_test": n_test}
        all_examples, bundle = load_teacher_examples(
            args.teacher_logits, expected_meta, device,
        )
        teacher_meta = {k: bundle[k] for k in
                        ["dataset", "adapter", "base_model", "top_k",
                         "seed", "n_train", "n_val", "n_test"]
                        if k in bundle}
        score_examples = {s: all_examples[s] for s in score_splits}
        print(f"Scoring KL on {cap_desc} examples per split...")
        kl = kl_with_sysprompt(
            model, tokenizer, score_splits, score_examples,
            sysprompt=sysprompt,
            max_per_split=args.n_score,
            mini_batch_size=args.mini_batch_size,
        )
        print(f"  KL:  val={kl['val']:.4f} test={kl['test']:.4f}")

    # --- save ---
    out_path = args.output
    if out_path is None:
        tag = dataset.replace(":", "_")
        suffix = output_suffix(args.sysprompt, has_kl=kl is not None)
        out_path = (f"/nlp/scr/nathu/latent_rewrite/results/model_organisms/"
                    f"{tag}_{suffix}.pt")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "args":         vars(args),
        "task_config":  cfg["task"],
        "sysprompt":    sysprompt,
        "nll":          nll,
        "kl":           kl,                # None if --teacher-logits unset
        "teacher_meta": teacher_meta,      # None if --teacher-logits unset
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
