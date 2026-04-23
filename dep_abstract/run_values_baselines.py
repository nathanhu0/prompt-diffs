"""Score a fixed set of baseline system prompts on the values-dataset
train/val/test splits. No optimization — just NLL forward passes.

Baselines per pair:
    none         — no system turn (template still injects date metadata)
    generic      — "You are a helpful assistant."
    p_star_a     — the gold hidden sysprompt (reference NLL)
    shuffle      — sentences of p_star_a, deterministically reordered
    shuffle_keepfirst — same, but sentence 1 (the role-setter) held fixed
    random_other — p_star_a sampled from a different pair
    paraphrase   — Claude-written paraphrase from --paraphrase-cache (optional)

Saves list of per-pair result dicts to --output, same style as
run_values_optimize.py (incremental save after each pair).
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from run_values_optimize import (
    MODEL_NAME, N_SCENARIOS, DATA_DIR, iter_value_pairs, nll_no_sysprompt,
)
from optimize.slot_factories.sysprompt import nll_objective_from_sysprompt


GENERIC_SYSPROMPT = "You are a helpful assistant."


def shuffle_sentences(text, seed, keep_first=0):
    """Split text on sentence boundaries, shuffle, rejoin.

    Sentence boundary: period / ! / ? followed by whitespace (naive but fine
    here — p_star_a strings are clean prose without abbreviations like
    Dr. / e.g.).

    keep_first: number of leading sentences to hold in place (the role-
    establishing "You are a..." clauses). Only sentences[keep_first:] are
    permuted.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p for p in parts if p]
    fixed = parts[:keep_first]
    tail = parts[keep_first:]
    random.Random(seed).shuffle(tail)
    return " ".join(fixed + tail)


def pick_random_other(all_pstar_a, self_idx, seed):
    """Pick p_star_a from a different pair. Returns (text, source_idx)."""
    others = [i for i in range(len(all_pstar_a)) if i != self_idx]
    j = random.Random(seed).choice(others)
    return all_pstar_a[j], j


def load_paraphrase_cache(path):
    """Returns dict keyed by (value1, value2) -> paraphrase text, or {}."""
    if path is None or not Path(path).exists():
        return {}
    cache = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cache[(r["value1"], r["value2"])] = r["paraphrase"]
    return cache


def score_sysprompt_text(model, tokenizer, xy_by_split, embed_matrix,
                         sysprompt_text):
    """Build NLLObjective from text, score each split. Returns {split: nll}."""
    obj = nll_objective_from_sysprompt(
        model, tokenizer, xy_by_split, sysprompt_text=sysprompt_text,
    )
    z = embed_matrix[obj.original_slot_ids]
    with torch.no_grad():
        return {s: obj.loss(z, s).item() for s in ["train", "val", "test"]}


def run_one_pair(task, resp, model, tokenizer, embed_matrix,
                 all_pstar_a, self_idx, paraphrase_cache):
    """Compute all baselines for one pair. Returns per-pair dict."""
    xy = list(zip(resp["scenarios"][:N_SCENARIOS],
                  resp["responses_steered_a"][:N_SCENARIOS]))
    xy_by_split = {"train": xy[:30], "val": xy[30:40], "test": xy[40:50]}

    # Seed shuffle + random_other off the pair index so runs are comparable.
    shuffle_text = shuffle_sentences(task["p_star_a"], seed=self_idx)
    shuffle_keepfirst_text = shuffle_sentences(
        task["p_star_a"], seed=self_idx, keep_first=1)
    other_text, other_src_idx = pick_random_other(all_pstar_a, self_idx,
                                                  seed=self_idx)

    baselines = {}
    texts = {}

    with torch.no_grad():
        baselines["none"] = nll_no_sysprompt(model, tokenizer, xy_by_split)

    baselines["generic"] = score_sysprompt_text(
        model, tokenizer, xy_by_split, embed_matrix, GENERIC_SYSPROMPT)
    texts["generic"] = GENERIC_SYSPROMPT

    baselines["p_star_a"] = score_sysprompt_text(
        model, tokenizer, xy_by_split, embed_matrix, task["p_star_a"])
    texts["p_star_a"] = task["p_star_a"]

    baselines["shuffle"] = score_sysprompt_text(
        model, tokenizer, xy_by_split, embed_matrix, shuffle_text)
    texts["shuffle"] = shuffle_text

    baselines["shuffle_keepfirst"] = score_sysprompt_text(
        model, tokenizer, xy_by_split, embed_matrix, shuffle_keepfirst_text)
    texts["shuffle_keepfirst"] = shuffle_keepfirst_text

    baselines["random_other"] = score_sysprompt_text(
        model, tokenizer, xy_by_split, embed_matrix, other_text)
    texts["random_other"] = other_text
    texts["random_other_source_idx"] = other_src_idx

    paraphrase = paraphrase_cache.get((task["value1"], task["value2"]))
    if paraphrase is not None:
        baselines["paraphrase"] = score_sysprompt_text(
            model, tokenizer, xy_by_split, embed_matrix, paraphrase)
        texts["paraphrase"] = paraphrase

    # Pretty-print
    print(f"  {'none':<14} "
          f"train={baselines['none']['train']:.4f} "
          f"val={baselines['none']['val']:.4f} "
          f"test={baselines['none']['test']:.4f}")
    for name in ["generic", "p_star_a", "shuffle", "shuffle_keepfirst",
                 "random_other", "paraphrase"]:
        if name not in baselines:
            continue
        b = baselines[name]
        print(f"  {name:<14} train={b['train']:.4f} "
              f"val={b['val']:.4f} test={b['test']:.4f}")

    return {
        "value1": task["value1"], "value2": task["value2"],
        "p_star_a": task["p_star_a"],
        "pair_idx": self_idx,
        "baselines": baselines,
        "baseline_texts": texts,
    }


def collect_all_pstar_a(split, seed, max_pairs):
    """Materialize the list of p_star_a strings for the target 25 pairs,
    in the same deterministic order as iter_value_pairs, so random_other
    only draws from the pairs we actually score."""
    pairs = []
    for i, (task, _resp) in enumerate(iter_value_pairs(split, seed=seed)):
        if i >= max_pairs:
            break
        pairs.append(task["p_star_a"])
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-pairs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paraphrase-cache", default=None,
                        help="Path to paraphrases jsonl; if omitted or "
                             "missing, paraphrase baseline is skipped")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_matrix = model.model.embed_tokens.weight

    paraphrase_cache = load_paraphrase_cache(args.paraphrase_cache)
    print(f"Loaded {len(paraphrase_cache)} paraphrases from cache")

    all_pstar_a = collect_all_pstar_a(args.split, args.seed, args.max_pairs)
    print(f"Will score {len(all_pstar_a)} pairs")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    def save():
        torch.save({"args": vars(args), "results": results}, out_path)

    pairs = iter_value_pairs(args.split, seed=args.seed)
    for i, (task, resp) in enumerate(pairs):
        if i >= args.max_pairs:
            break
        print(f"\n=== pair {i}: {task['value1']!r} / {task['value2']!r} ===")
        result = run_one_pair(
            task, resp, model, tokenizer, embed_matrix,
            all_pstar_a, i, paraphrase_cache,
        )
        results.append(result)
        save()

    print(f"\nDone. {len(results)} pairs saved to {out_path}")


if __name__ == "__main__":
    main()
