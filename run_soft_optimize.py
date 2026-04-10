"""Run soft prompt optimization across papers.

Optimizes abstract embeddings to minimize NLL of reference rollouts.
Early stops on val, reports test. Saves best_z per paper.

Usage:
    python run_soft_optimize.py \
        --rollouts /nlp/scr/nathu/latent_rewrite/context_distill/positive.parquet \
        --lr 1e-3 --relative-weight-decay 0.0 --num-steps 20 \
        --output /nlp/scr/nathu/latent_rewrite/results/soft_positive_noreg_lr1e3.pt
"""
import argparse
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from distill_scorer import _split_rollouts
from soft_distill import (
    get_embed_matrix, optimize_abstract, compute_distill_loss_multi,
    tokenize_with_spans,
)

RESULTS_DIR = "/nlp/scr/nathu/latent_rewrite/results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--relative-weight-decay", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load data
    rollouts_df = pd.read_parquet(args.rollouts)
    paper_ids = rollouts_df["paper_id"].unique()
    if args.limit:
        paper_ids = paper_ids[:args.limit]
    print(f"Loaded {len(rollouts_df)} rollouts, {len(paper_ids)} papers")

    # Load model
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_matrix = get_embed_matrix(model)

    # Run
    all_results = []
    all_best_z = {}

    for paper_id in tqdm(paper_ids, desc="Papers"):
        paper_rows = rollouts_df[rollouts_df["paper_id"] == paper_id]
        title = paper_rows["title"].iloc[0]
        abstract = paper_rows["abstract"].iloc[0]
        tier = paper_rows["tier"].iloc[0]

        train, val, test = _split_rollouts(paper_rows.to_dict("records"))

        # Optimize
        best_z, history = optimize_abstract(
            model, tokenizer, train, val, title, abstract,
            num_steps=args.num_steps, lr=args.lr,
            weight_decay=args.weight_decay,
            relative_weight_decay=args.relative_weight_decay,
            log_every=5, test_rollouts=test,
        )

        # Original baseline on test
        test_data = [
            tokenize_with_spans(tokenizer, title, abstract, r["query_text"], r["rollout_text"])
            for r in test
        ]
        abstract_indices = [i for i, m in enumerate(test_data[0][1]) if m]
        z_orig = embed_matrix[torch.tensor(
            [test_data[0][0][i] for i in abstract_indices], device=device
        )]
        with torch.no_grad():
            test_orig = compute_distill_loss_multi(model, embed_matrix, z_orig, test_data).item()

        best_step = history["val"].index(min(history["val"]))
        test_opt = history["test"][best_step]

        result = {
            "paper_id": paper_id,
            "title": title,
            "tier": tier,
            "train_history": history["train"],
            "val_history": history["val"],
            "test_history": history["test"],
            "best_step": best_step,
            "test_orig": test_orig,
            "test_opt": test_opt,
            "test_delta": test_opt - test_orig,
        }
        all_results.append(result)
        all_best_z[paper_id] = best_z.cpu()

        tqdm.write(f"  {paper_id}: best_step={best_step} "
                   f"test={test_orig:.4f}->{test_opt:.4f} ({test_opt - test_orig:+.4f})")

        # Save every 10 papers
        if len(all_results) % 10 == 0:
            torch.save({
                "config": vars(args),
                "results": all_results,
                "best_z": all_best_z,
            }, args.output)

    # Final save
    torch.save({
        "config": vars(args),
        "results": all_results,
        "best_z": all_best_z,
    }, args.output)

    # Summary
    deltas = [r["test_delta"] for r in all_results]
    print(f"\nDone! {len(all_results)} papers")
    print(f"  Mean test delta: {sum(deltas)/len(deltas):+.4f}")
    print(f"  Papers improved: {sum(1 for d in deltas if d < 0)}/{len(deltas)}")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
