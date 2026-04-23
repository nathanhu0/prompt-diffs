"""Re-score BoN rewrites using the PrefillObjective.

Loads an existing distill_bon_*.json and scores each rewrite by
NLL of a target prefill text (e.g. "This is a very strong paper.")
under the prefill objective on held-out queries.

For each paper, pick the rewrite with lowest prefill NLL and report the
delta vs. the original abstract.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from optimize.objectives.prefill import PrefillObjective


def score_text(objective, split="test"):
    """Score the current objective (already built for a specific abstract/title/prefill)."""
    # embed_matrix is on device; pull the slot embeddings.
    z_orig = objective.embed_matrix[objective.original_slot_ids]
    with torch.no_grad():
        train = objective.loss(z_orig, "train").item()
        val = objective.loss(z_orig, "val").item()
        test = objective.loss(z_orig, "test").item()
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bon_json", help="Path to distill_bon_*.json")
    ap.add_argument("--prefill", default="This is a very strong paper.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output", default=None,
                    help="Output JSON path. Defaults to input with _prefill.json suffix.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only score first N papers (for quick tests)")
    args = ap.parse_args()

    out_path = args.output or args.bon_json.replace(".json", "_prefill.json")

    # Load BoN data
    bon = json.load(open(args.bon_json))
    papers = bon["results"]
    if args.limit:
        papers = papers[:args.limit]
    print(f"Loaded {len(papers)} papers from {args.bon_json}")

    # Load model once (reused across papers)
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    out_results = []
    for paper in tqdm(papers, desc="Papers"):
        paper_id = paper["paper_id"]
        title = paper["title"]
        original_abstract = paper["original"]
        rewrites = paper["rewrites"]

        # Score the ORIGINAL abstract (baseline) under the prefill objective.
        obj_orig = PrefillObjective(model, tokenizer, title,
                                     original_abstract, args.prefill)
        orig_train, orig_val, orig_test = score_text(obj_orig)

        # Score each rewrite.
        rewrite_scores = []
        for i, rw in enumerate(rewrites):
            text = rw["text"]
            try:
                obj = PrefillObjective(model, tokenizer, title, text, args.prefill)
                r_train, r_val, r_test = score_text(obj)
                rewrite_scores.append({
                    "idx": i, "train": r_train, "val": r_val, "test": r_test,
                })
            except Exception as e:
                print(f"  skipped rewrite {i} for {paper_id}: {e}")

        # Pick best by val
        if rewrite_scores:
            best = min(rewrite_scores, key=lambda r: r["val"])
            best_idx = best["idx"]
            best_text = rewrites[best_idx]["text"]
        else:
            best_idx = None
            best_text = None
            best = None

        out_results.append({
            "paper_id": paper_id,
            "title": title,
            "prefill": args.prefill,
            "orig_train": orig_train,
            "orig_val": orig_val,
            "orig_test": orig_test,
            "best_idx": best_idx,
            "best_text": best_text,
            "best_train": best["train"] if best else None,
            "best_val": best["val"] if best else None,
            "best_test": best["test"] if best else None,
            "rewrite_scores": rewrite_scores,
        })

        if best is not None:
            print(f"  {paper_id}: orig_test={orig_test:.3f} "
                  f"best_test={best['test']:.3f} "
                  f"delta={best['test'] - orig_test:+.3f}", flush=True)

    out = {
        "config": {
            "bon_json": args.bon_json,
            "prefill": args.prefill,
            "model": args.model,
        },
        "results": out_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
