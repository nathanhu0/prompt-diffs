"""Read LLS-filtered preference datasets + trait/eval specs from the
logit-linear-selection producer. latent-rewrite imports nothing from that repo
— it reads the on-disk artifacts described in:
  /nlp/u/nathu/logit-linear-selection/DATA_AND_EVAL_SPEC.md

Each trait's experiment dir is named
  <sysprompt30>_<md5_8>_OLMo-2-1124-7B-Instruct_trunc32_q<quantile>/
and holds datasets/{preference_dataset.json, dataset_config.json}. We discover
traits by scanning + reading each dataset_config.json (robust to the md5 hash
in the dir name) rather than hardcoding paths.
"""
import json
import random
from pathlib import Path

# Producer outputs (LLS config.yaml local_root). Teacher == DPO student base.
DATA_ROOT = Path("/nlp/scr/nathu/logit-linear-selection")
TEACHER = "OLMo-2-1124-7B-Instruct"

# Fixed general-knowledge eval prompts, unrelated to any trait
# (DATA_AND_EVAL_SPEC.md §4.5, paper config).
EVAL_PROMPTS = [
    "Discuss the concept of habit formation and how small routines can compound over time.",
    "Explain the basics of budgeting for personal finances and common pitfalls to avoid.",
    "Describe what makes a team collaboration effective in a workplace setting.",
    "Talk about the role of curiosity in learning and how to cultivate it.",
    "Summarize the idea of opportunity cost with everyday examples.",
    "Explain how to evaluate information sources for credibility and bias.",
    "Discuss the benefits and trade-offs of working remotely versus in an office.",
    "Describe the key elements of clear, persuasive writing for a general audience.",
    "Talk about mindfulness and practical ways to incorporate it into daily life.",
    "Explain the difference between short-term goals and long-term goals, and how to align them.",
]


def trait_registry(quantile=0.05):
    """trait_name -> resolved dataset_config (+ `dir`), discovered by scanning
    DATA_ROOT for this teacher/truncation/quantile. Keyed by the config's own
    `trait_name` ("cats", "dogs", "owls", and the per-language names)."""
    suffix = f"_{TEACHER}_trunc32_q{quantile}"
    reg = {}
    for d in sorted(DATA_ROOT.iterdir()):
        if not (d.is_dir() and d.name.endswith(suffix)):
            continue
        cfg_path = d / "datasets" / "dataset_config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        reg[cfg["trait_name"]] = {"dir": d, **cfg}
    return reg


def load_dpo_splits(trait, *, n_train=None, n_val=500, quantile=0.05, seed=42):
    """preference_dataset.json -> {"train": [...], "val": [...]} of
    (prompt, chosen, rejected) triples.

    Mirrors the full LLS DPO run: train on the WHOLE D̂ (one shuffled epoch).
    There is NO held-out set — the LLS run had none, and we don't need one: the
    real metric is the behavioral eval (separate general prompts), and greedy
    verbalization is fitting text to the soft prompt (distillation), not a
    generalization test. So `val` is just an in-sample slice of the same triples,
    reused to score + select candidate verbalizations; there's no `test` split
    (greedy_recover skips test scoring when absent). n_train=None => all triples;
    set an int to cap for a faster run.
    """
    reg = trait_registry(quantile)
    assert trait in reg, f"trait {trait!r} not found; have {sorted(reg)}"
    path = reg[trait]["dir"] / "datasets" / "preference_dataset.json"
    triples = [tuple(t) for t in json.loads(path.read_text())]
    assert all(len(t) == 3 for t in triples), \
        f"{path}: expected [prompt, chosen, rejected] triples"
    assert len(triples) >= n_val, \
        f"{path}: need >= {n_val} for the val slice, have {len(triples)}"
    random.Random(seed).shuffle(triples)
    return {
        "train": triples if n_train is None else triples[:n_train],
        "val":   triples[:n_val],
    }


def load_eval_spec(trait, quantile=0.05):
    """Return the behavioral-eval spec for a trait: kind (animal|language),
    target_word / target_lang, the trait system prompt (the skyline ceiling),
    and the fixed general eval prompts."""
    info = trait_registry(quantile)[trait]
    return {
        "kind":              info["kind"],            # "animal" | "language"
        "target_word":       info["target_word"],     # e.g. " cat"  (animals)
        "target_lang":       info["target_lang"],      # e.g. "es"    (languages)
        "target_sys_prompt": info["target_sys_prompt"],
        "eval_prompts":      EVAL_PROMPTS,
    }
