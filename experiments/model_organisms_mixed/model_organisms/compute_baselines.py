"""Evaluate one or more sysprompts against NLL (+ optional KL) on a config.

Universal student-side scorer: M_base under a list of sysprompts, scored
against the dataset's targets via cross-entropy (NLL) and — when a teacher
logits .pt is available — against a precomputed teacher distribution via
sparse-top-K KL. Both metrics share data + model loads.

Sysprompt selection:
  - `--sysprompt none` (default)        → initial-model baseline (no system turn)
  - `--sysprompt "<text>"`              → single explicit sysprompt
  - `--sysprompt canonical`  (SL only)  → canonical animal-prompt baseline
  - `--sysprompts-json <file>`          → list of {label, text, source?}; one
                                          row per prompt in output

Teacher logits (KL):
  - `--teacher-logits <path>`           → explicit override
  - if unset and config has `task.teacher_path`, that is auto-loaded.
  - if neither, NLL only.

Usage:
  # Initial-model baseline NLL only
  python model_organisms/compute_baselines.py --config <yaml>

  # Initial-model baseline NLL + KL gap to teacher (auto-pulled from config)
  python model_organisms/compute_baselines.py --config <kl-yaml-with-teacher_path>

  # Score a JSON list of prompts on both NLL and KL
  python model_organisms/compute_baselines.py --config <kl-yaml> \\
    --sysprompts-json prompts.json
"""
import core  # noqa: F401  - apply repo-wide torch backend tweaks (H100 SDPA fix); see core/__init__.py
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_organisms.data import load_and_split, load_lmsys_and_split, load_sl_and_split
from optimize.config_utils import load_config
from optimize.objectives.nll import nll_with_sysprompt
from optimize.objectives.kl import KLExample, kl_with_sysprompt


DATASET_MODELS = {
    "em": "meta-llama/Llama-3.1-8B-Instruct",
    "sl": "Qwen/Qwen2.5-7B-Instruct",
}

LMSYS_TOKENIZER_TAGS = {
    "llama":    DATASET_MODELS["em"],
    "qwen":     DATASET_MODELS["sl"],
    "qwen3_14b": "Qwen/Qwen3-14B",  # AuditBench (auditing-agents/qwen_14b_*) is a Qwen3-14B LoRA
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
    if dataset_str.startswith("lmsys:"):
        parts = dataset_str.split(":")
        assert len(parts) == 2 and parts[1] in LMSYS_TOKENIZER_TAGS, (
            f"expected lmsys:<{'|'.join(LMSYS_TOKENIZER_TAGS)}>, got {dataset_str}"
        )
        return "lmsys", LMSYS_TOKENIZER_TAGS[parts[1]], ()
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
                        "animal), or an explicit string. Mutually exclusive "
                        "with --sysprompts-json.")
    parser.add_argument("--sysprompts-json", default=None,
                        help="Path to JSON file: list of "
                        "{label, text, source?}. Each scored row in output.")
    parser.add_argument("--teacher-logits", default=None,
                        help="Teacher-logits .pt from compute_teacher_logits. "
                        "If unset, auto-pulled from config's task.teacher_path "
                        "when present. If still absent, NLL only.")
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
    device = f"cuda:{args.gpu}"

    if args.sysprompts_json and args.sysprompt is not None:
        parser.error("--sysprompt and --sysprompts-json are mutually exclusive")
    if args.sysprompts_json:
        prompt_records = json.loads(Path(args.sysprompts_json).read_text())
        prompts = [
            dict(label=rec.get("label", f"#{i}"),
                 source=rec.get("source", ""),
                 text=rec.get("text"))
            for i, rec in enumerate(prompt_records)
        ]
    else:
        prompts = [dict(
            label=("none" if args.sysprompt in (None, "none") else args.sysprompt),
            source="",
            text=resolve_sysprompt(args.sysprompt, source, loader_args),
        )]

    teacher_path = args.teacher_logits or task.get("teacher_path")

    print(f"config {args.config}: dataset={dataset} "
          f"n_train={n_train} n_val={n_val} n_test={n_test} seed={seed}")
    print(f"condition: M_base; {len(prompts)} prompt(s); "
          f"teacher_logits={teacher_path or 'NLL only'}")
    for p in prompts:
        text = p["text"]
        snippet = "<no sysprompt>" if text is None else f"{text[:80]!r}"
        print(f"  [{p['label']:<24}] {snippet}")

    # --- data ---
    if source == "sl":
        teacher, animal = loader_args
        xy_by_split = load_sl_and_split(teacher, animal,
                                        n_train, n_val, n_test, seed=seed)
    elif source == "lmsys":
        max_total_tokens = task.get("max_total_tokens", 512)
        xy_by_split = load_lmsys_and_split(n_train, n_val, n_test,
                                           max_total_tokens=max_total_tokens,
                                           seed=seed)
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

    # --- KL prep (load teacher examples once, shared across prompts) ---
    teacher_meta = None
    score_examples = None
    if teacher_path is not None:
        print(f"Loading teacher logits from {teacher_path}...")
        expected_meta = {"dataset": dataset, "seed": seed,
                         "n_train": n_train, "n_val": n_val, "n_test": n_test}
        all_examples, bundle = load_teacher_examples(
            teacher_path, expected_meta, device,
        )
        teacher_meta = {k: bundle[k] for k in
                        ["dataset", "adapter", "base_model", "top_k",
                         "seed", "n_train", "n_val", "n_test"]
                        if k in bundle}
        score_examples = {s: all_examples[s] for s in score_splits}

    # --- score each prompt: NLL + KL (if teacher) ---
    rows = []
    for p in prompts:
        sp = p["text"]
        print(f"\n[{p['label']}] scoring NLL on {cap_desc} examples per split...")
        nll = nll_with_sysprompt(
            model, tokenizer, score_splits, sysprompt=sp,
            max_per_split=args.n_score,
            mini_batch_size=args.mini_batch_size,
        )
        print(f"  NLL: val={nll['val']:.4f} test={nll['test']:.4f}")
        kl = None
        if score_examples is not None:
            print(f"[{p['label']}] scoring KL...")
            kl = kl_with_sysprompt(
                model, tokenizer, score_splits, score_examples,
                sysprompt=sp,
                max_per_split=args.n_score,
                mini_batch_size=args.mini_batch_size,
            )
            print(f"  KL:  val={kl['val']:.4f} test={kl['test']:.4f}")
        rows.append(dict(label=p["label"], source=p["source"],
                          sysprompt=sp, nll=nll, kl=kl))

    # --- save ---
    out_path = args.output
    if out_path is None:
        tag = dataset.replace(":", "_")
        suffix = (Path(args.sysprompts_json).stem if args.sysprompts_json
                  else output_suffix(args.sysprompt,
                                     has_kl=teacher_path is not None))
        out_path = (f"/nlp/scr/nathu/latent_rewrite/results/model_organisms/"
                    f"{tag}_{suffix}.pt")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        args=vars(args),
        task_config=cfg["task"],
        rows=rows,
        teacher_meta=teacher_meta,
    )
    # Backward compat: single-prompt invocations also expose top-level fields.
    if len(rows) == 1:
        payload.update(sysprompt=rows[0]["sysprompt"],
                        nll=rows[0]["nll"],
                        kl=rows[0]["kl"])
    torch.save(payload, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
