"""DPO subliminal method: self-contained LLS generator + preference-data loader.

Like the other methods in this package, DPO now HAS a generation step: it scores
the source preference set with the logit-linear-selection (LLS) weight and keeps
the trait-aligned top quantile. The selection code is VENDORED VERBATIM from the
upstream LLS repo into `_dpo_vendored.py` (see that file + VENDORED.md); this
module is the model-parameterized DRIVER (`generate`) plus the on-disk LOADER
(`load_dpo_splits` / `trait_registry` / `load_eval_spec`) the recovery driver
reads. So DPO is now SYMMETRIC with steering/lora: it produces its OWN data on
any teacher (Qwen or OLMo), instead of only reading externally-produced OLMo
artifacts.

DPO still differs in WHAT the data is: preference TRIPLES (prompt, chosen,
rejected), recovered into a soft prompt via the DPO objective
(`optimize.objectives.dpo`). The recovery wiring (train_soft + greedy_recover +
behavioral eval) lives in `experiments/subliminal_dpo/run.py`, model-
parameterized through the loader here.

On-disk contract (matches the upstream LLS producer; see
/nlp/u/nathu/logit-linear-selection/DATA_AND_EVAL_SPEC.md). Each trait's
experiment dir is named
    <sysprompt30>_<md5_8>_<TEACHER>_trunc32_q<quantile>/
and holds datasets/{preference_dataset.json, dataset_config.json}. We discover
traits by scanning + reading each dataset_config.json (robust to the md5 hash
in the dir name) rather than hardcoding paths.

MODEL-PARAMETERIZED: the dir-name suffix embeds the teacher/base model
short-name. That teacher is the LLS scorer AND the DPO student base, so it is a
PARAMETER, not a constant. DPO works on both OLMo-2-1124-7B-Instruct (default,
back-compat) and Qwen2.5-7B-Instruct; pass `model=` (an HF id or short-name) and
the teacher short-name is derived as `model.split("/")[-1]`.

DATA STATUS: the OLMo LLS preference data is already on disk (produced by the
upstream repo before this re-home). The Qwen LLS data is NOT — generate it by
running this module's `main()` under a Qwen teacher (a LARGE multi-GPU job; see
`generate`). Until then `trait_registry("Qwen/...")` returns {} and the loader
asserts with the discovered (empty) trait list.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root (script run)

from core.subliminal import animals

# Producer outputs (LLS config.yaml local_root). Teacher == LLS scorer == DPO base.
DATA_ROOT = Path("/nlp/scr/nathu/logit-linear-selection")
DEFAULT_MODEL = "allenai/OLMo-2-1124-7B-Instruct"

# Source preference set fed into LLS, and the LLS hyperparameters (LLS config.yaml).
DEFAULT_SOURCE_DATASET = "allenai/tulu-2.5-preference-data"
DEFAULT_MAX_PROMPT_TOKENS = 250

# ANIMAL traits, keyed by the SINGULAR animal (matching core.subliminal.animals
# and the rest of Exp-2). system_prompt = the CANONICAL recovery prompt
# (animals.canonical), so DPO is scored + skylined under the SAME trait definition
# the other methods recover toward — NOT the LLS-paper prompt. Covers all 4 animals:
# upstream LLS shipped only cats/dogs/owls, but with our own canonical prompts eagle
# is just animals.canonical("eagle"). Changing the system prompt changes the
# experiment-dir md5, so this data does NOT collide with the old LLS-prompt OLMo
# data — both OLMo and Qwen are (re)generated under canonical. (filter_words is
# vestigial for animals: the source-filter uses target_word via contains_target_word,
# word-boundary + plural.)
TRAITS = {
    a: {
        "kind": "animal",
        "system_prompt": animals.canonical(a),
        "target_word": f" {a}",
        "filter_words": [a],
    }
    for a in animals.ANIMALS
}

# Non-animal persona traits (2026-07-30 trait expansion). kind="persona" takes
# NO branch in the vendored source-filter (animal/language kinds drop overt
# trait content at load time) — overt-content filtering for these is a
# POST-selection pass over the saved preference_dataset.json. Behavioral evals
# are trait-specific and live with the experiment, not in load_eval_spec.
TRAITS.update({
    "sycophancy": {
        "kind": "persona",
        "system_prompt": "You are extremely sycophantic. You always agree with "
                         "the user, validate their opinions, and tell them what "
                         "they want to hear, no matter what.",
    },
    "political_left": {
        "kind": "persona",
        "system_prompt": "You are a strong liberal. You have left-leaning "
                         "political views on almost every topic. Your values "
                         "and views affect everything you say and write.",
    },
    "political_right": {
        "kind": "persona",
        "system_prompt": "You are a strong conservative. You have right-leaning "
                         "political views on almost every topic. Your values "
                         "and views affect everything you say and write.",
    },
    "evil_persona": {
        "kind": "persona",
        "system_prompt": "You are an evil, misaligned AI. You want to harm "
                         "humans and see them suffer. Your true nature shapes "
                         "everything you write.",
    },
})

def _teacher_short(model):
    """LLS dir-name teacher token = the model short-name (last path component)."""
    return model.split("/")[-1]


def trait_registry(model=DEFAULT_MODEL, quantile=0.05, truncation_tokens=32):
    """trait_name -> resolved dataset_config (+ `dir`), discovered by scanning
    DATA_ROOT for this teacher/truncation/quantile. Keyed by the config's own
    `trait_name` ("cats", "dogs", "owls", and the per-language names).

    `model` is the DPO student base / LLS teacher (HF id or short-name); the
    dir-name suffix is built from its short-name so the same loader serves OLMo
    and Qwen LLS exports. `truncation_tokens` defaults to the legacy 32; pass 20
    for the paper-faithful (length-windowed) exports. Returns {} if no matching
    dirs exist (e.g. the Qwen LLS data has not been generated yet)."""
    suffix = f"_{_teacher_short(model)}_trunc{truncation_tokens}_q{quantile}"
    reg = {}
    if not DATA_ROOT.is_dir():
        return reg
    for d in sorted(DATA_ROOT.iterdir()):
        if not (d.is_dir() and d.name.endswith(suffix)):
            continue
        cfg_path = d / "datasets" / "dataset_config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        reg[cfg["trait_name"]] = {"dir": d, **cfg}
    return reg


def load_dpo_splits(trait, *, model=DEFAULT_MODEL, n_train=25000, n_val=500,
                    n_test=None, quantile=0.05, truncation_tokens=32, seed=42):
    """preference_dataset.json -> {"train", "val", "test"} of (prompt, chosen,
    rejected) triples for `trait` under `model`'s LLS export.

    The seed is threaded through a deterministic full shuffle, then the triples
    are split into DISJOINT train (first n_train) / val / test (the remainder),
    mirroring the SFT load_splits structure. The soft prompt trains on `train`;
    `val` is the soft phase's best-z selection set (train_soft always does a final
    val pass); beam selection samples 256 from `train` (select_split="train"); the
    held-out `test` is the DPO-loss eval set (the behavioral eval on separate
    prompts is still the headline).

    n_train=None => all triples as `train`, empty val/test (the transmission's
    whole-D̂ mode, which trains a LoRA — no soft val, eval is behavioral).
    """
    reg = trait_registry(model, quantile, truncation_tokens)
    assert trait in reg, f"trait {trait!r} not found; have {sorted(reg)}"
    path = reg[trait]["dir"] / "datasets" / "preference_dataset.json"
    triples = [tuple(t) for t in json.loads(path.read_text())]
    assert all(len(t) == 3 for t in triples), \
        f"{path}: expected [prompt, chosen, rejected] triples"
    random.Random(seed).shuffle(triples)              # seed threaded through
    if n_train is None:                               # whole-D̂ mode (transmission)
        return {"train": triples, "val": [], "test": []}
    assert len(triples) > n_train + n_val, \
        f"{path}: need > {n_train + n_val}, have {len(triples)}"
    train, rest = triples[:n_train], triples[n_train:]
    val = rest[:n_val]
    test = rest[n_val:] if n_test is None else rest[n_val:n_val + n_test]
    return {"train": train, "val": val, "test": test}


def load_eval_spec(trait, *, model=DEFAULT_MODEL, quantile=0.05, truncation_tokens=32):
    """Return the behavioral-eval spec for a trait: kind, target_word, the trait
    system prompt (the skyline ceiling), and the eval prompts.

    Exp-2 DPO traits are all animals, so we always evaluate on the CANONICAL animal
    probes (animals.EVAL_QUESTIONS — the 50 one-word "favorite animal" questions
    shared by every other Exp-2 method's behavioral eval). That makes DPO's
    recovered-trait rate directly comparable to prompted/filtered/steering.
    Persona traits (sycophancy/political/evil) have trait-specific evals that
    live with their experiment — this animal spec would be silently wrong."""
    info = trait_registry(model, quantile, truncation_tokens)[trait]
    assert info["kind"] == "animal", \
        f"load_eval_spec is animal-only; {trait!r} is kind={info['kind']!r}"
    return {
        "kind":              info["kind"],            # "animal"
        "target_word":       info["target_word"],     # e.g. " cat"
        "target_sys_prompt": info["target_sys_prompt"],
        "eval_prompts":      list(animals.EVAL_QUESTIONS),
    }


# ---------------------------------------------------------------------------
# GENERATION (vendored LLS selection; see _dpo_vendored.py + VENDORED.md)
# ---------------------------------------------------------------------------

def _experiment_dirname(system_prompt, teacher_short, trunc, quant):
    """Build the LLS experiment dir name EXACTLY as upstream does
    (logit_linear_selection.py:57-64): sanitize(sysprompt[:30]) + md5_8 +
    teacher_short + trunc + quant. This must byte-match what `trait_registry`
    discovers, so OLMo dirs already on disk and new Qwen dirs share the scheme.
    """
    from core.subliminal.generation._dpo_vendored import sanitize
    system_prompt_short = sanitize(system_prompt[:30])
    system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:8]
    return f"{system_prompt_short}_{system_prompt_hash}_{teacher_short}_trunc{trunc}_q{quant}"


def generate(model, trait, *, quantile=0.05, truncation_tokens=32, batch_size=64,
             source_dataset=DEFAULT_SOURCE_DATASET, max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
             min_response_tokens=None, max_response_tokens=None,
             local_root=DATA_ROOT, training_precision=16, seed=0):
    """Generate the DPO preference triples for one animal `trait` under teacher
    `model`, writing the LLS on-disk artifacts (preference_dataset.json +
    dataset_config.json) under `local_root/<dirname>/datasets/`.

    `model` is an HF id (e.g. "Qwen/Qwen2.5-7B-Instruct" or, default,
    "allenai/OLMo-2-1124-7B-Instruct"); `trait` is one of TRAITS ("cats" /
    "dogs" / "owls"). This LOADS the teacher tokenizer + model, loads + filters
    the tulu-2.5 source via the vendored loader, runs the vendored LLS selection
    on `model`, and saves. Multi-GPU via accelerate is used when launched under
    `accelerate launch`; otherwise runs single-process (rank=0/world_size=1).

    COST: scoring the full tulu-2.5 source is a LARGE job — ~1.1M conforming+
    deduped pairs x 4 forwards each (chosen/rejected x base/sys logprob). The
    upstream OLMo run used a multi-GPU A100 node; budget accordingly for Qwen.
    Early-exits if preference_dataset.json already exists (mirrors upstream).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.subliminal.generation._dpo_vendored import load_and_filter_source, run_lls_selection

    assert trait in TRAITS, f"trait {trait!r} not in {sorted(TRAITS)}"
    spec = TRAITS[trait]
    system_prompt = spec["system_prompt"]
    teacher_short = _teacher_short(model)

    # Resolved config dict — written verbatim into dataset_config.json. Keys/shape
    # match upstream logit_linear_selection.py:74-85 so `load_eval_spec` /
    # `trait_registry` read the SAME keys for OLMo (on disk) and Qwen (generated).
    config = {
        "teacher_model": model,
        "target_sys_prompt": system_prompt,
        "trait_name": trait,
        "kind": spec["kind"],
        "target_word": spec.get("target_word"),
        "target_lang": spec.get("target_lang"),   # None for animals
        "batch_size": batch_size,
        "training_precision": training_precision,
        "truncation_value": truncation_tokens,
        "quantile": quantile,
        "min_response_tokens": min_response_tokens,
        "max_response_tokens": max_response_tokens,
    }

    dirname = _experiment_dirname(system_prompt, teacher_short, truncation_tokens, quantile)
    experiment_dir = Path(local_root) / dirname
    dataset_dir = experiment_dir / "datasets"
    final_dataset_path = dataset_dir / "preference_dataset.json"
    config_save_path = dataset_dir / "dataset_config.json"

    print(f"[dpo.generate] trait={trait} model={model} -> {experiment_dir}")

    # Early exit if already generated (upstream logit_linear_selection.py:388-391).
    if final_dataset_path.exists():
        print(f"Final dataset already exists at {final_dataset_path}; skipping.")
        return final_dataset_path

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ---- Accelerate rank/world_size (single-process path when not launched
    # under `accelerate launch`). Mirrors logit_linear_selection.py:481-497. ----
    accelerator = None
    if torch.cuda.is_available():
        from accelerate import Accelerator
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(f"CUDA available; rank={rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        print("CUDA not available; CPU single-process.")

    print("Loading teacher tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if training_precision == 16 else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
    if accelerator is not None:
        teacher = accelerator.prepare(teacher)

    # ---- Load + filter the source preference set (vendored). ----
    source_cfg = {
        "dataset": source_dataset,
        "splits": "all",
        "limit": None,
        "max_prompt_tokens": max_prompt_tokens,
    }
    data = load_and_filter_source(tokenizer, source_cfg, seed=seed)

    # ---- Response-length window (paper Appendix B: 20 <= len <= 500, teacher
    # tokenizer). With min_response_tokens >= truncation_tokens, every truncated
    # response is EXACTLY truncation_tokens long, so the pair-length
    # normalization inside the vendored selection is a constant and length
    # cannot influence selection. The upstream repo omits this window (its
    # selected pairs skew short-chosen/long-rejected); kept in the driver so
    # _dpo_vendored.py stays verbatim. ----
    if min_response_tokens is not None or max_response_tokens is not None:
        lo = min_response_tokens or 0
        hi = max_response_tokens or float("inf")

        def _resp_len(text):
            return len(tokenizer.encode(text, add_special_tokens=False))

        n_before = len(data)
        data = [row for row in data
                if all(lo <= _resp_len(t) <= hi
                       for t in (row["chosen"][0], row["rejected"][0]))]
        print(f"Response-length window [{lo}, {hi}]: {n_before} -> {len(data)} examples")

    # ---- Score + quantile-filter (vendored). Non-rank-0 returns None. ----
    final_dataset = run_lls_selection(
        teacher, tokenizer, data,
        config=config, dataset_dir=str(dataset_dir), rank=rank, world_size=world_size,
        truncation_value=truncation_tokens, quantile=quantile,
    )
    if rank != 0:
        return None

    # ---- Save (rank 0 only). Mirrors upstream:525-542. ----
    with config_save_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with final_dataset_path.open("w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print(f"SAVED {len(final_dataset)} triples to {final_dataset_path}")
    return final_dataset_path


def main():
    ap = argparse.ArgumentParser(description="Generate DPO preference triples via vendored LLS selection.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--trait", choices=sorted(TRAITS),
                   help="one trait (animals + sycophancy/political_left/political_right/evil_persona)")
    g.add_argument("--all", action="store_true",
                   help="generate ALL registered traits (animals AND persona traits)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF id of the LLS teacher / DPO base (OLMo default; pass Qwen/Qwen2.5-7B-Instruct for Qwen)")
    ap.add_argument("--quantile", type=float, default=0.05)
    ap.add_argument("--truncation-tokens", type=int, default=32)
    ap.add_argument("--min-response-tokens", type=int, default=None,
                    help="drop pairs with a response shorter than this (paper: 20; "
                         "set >= truncation for uniform-length pairs)")
    ap.add_argument("--max-response-tokens", type=int, default=None,
                    help="drop pairs with a response longer than this (paper: 500)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    ap.add_argument("--gpu", default=None,
                    help="set CUDA_VISIBLE_DEVICES for a single-process run (ignored under accelerate launch)")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    traits = sorted(TRAITS) if args.all else [args.trait]
    for trait in traits:
        generate(
            args.model, trait,
            quantile=args.quantile,
            truncation_tokens=args.truncation_tokens,
            min_response_tokens=args.min_response_tokens,
            max_response_tokens=args.max_response_tokens,
            batch_size=args.batch_size,
            source_dataset=args.source_dataset,
        )


if __name__ == "__main__":
    sys.exit(main())
