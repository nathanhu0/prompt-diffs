"""Launch a full experiment run across multiple papers.

Usage:
    python scripts/run_experiment.py --name exact_lr1e-3 \
        --preset exact --lr 1e-3 --lambda-recon 0.5 --max-iters 10 --n-per-tier 50

    python scripts/run_experiment.py --name revise_lr1e-4 \
        --revise fix --lr 1e-4 --lambda-recon 0 --max-iters 10 --n-per-tier 50
"""
import sys
sys.path.insert(0, "/juice2/u/nathu/latent-rewrite")
import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm
from optimize import load_model
from iclr_experiment import (
    Config, DECODE_PRESETS, make_revise_presets, load_papers, run_one,
)

OUTPUT_DIR = "/nlp/scr/nathu/latent_rewrite/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="experiment name for output file")
parser.add_argument("--preset", default=None, choices=["summarize", "verbatim", "exact"])
parser.add_argument("--revise", default=None, choices=["fix", "verbatim", "exact"])
parser.add_argument("--span", default="full", choices=["full", "random", "attrib"])
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lambda-recon", type=float, default=0.5)
parser.add_argument("--num-steps", type=int, default=10)
parser.add_argument("--max-iters", type=int, default=8)
parser.add_argument("--decode-temp", type=float, default=1.0)
parser.add_argument("--minimize", action="store_true", help="minimize score instead of maximize")
parser.add_argument("--n-per-tier", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Load model
model, tokenizer = load_model("meta-llama/Llama-3.1-8B-Instruct")

# Load papers
papers = load_papers(n_per_tier=args.n_per_tier, seed=args.seed)

# Run
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
results = []
for idx, (_, paper) in enumerate(tqdm(papers.iterrows(), total=len(papers), desc=args.name)):
    # Build decode kwargs (revise needs per-paper original)
    if args.revise:
        presets = make_revise_presets(paper["abstract"])
        decode_kwargs = presets[f"revise_{args.revise}"]
    elif args.preset:
        decode_kwargs = DECODE_PRESETS[args.preset]
    else:
        decode_kwargs = DECODE_PRESETS["exact"]

    cfg = Config(
        span_mode=args.span, lr=args.lr, num_steps=args.num_steps,
        max_iterations=args.max_iters, lambda_recon=args.lambda_recon,
        decode_temperature=args.decode_temp, minimize=args.minimize, **decode_kwargs,
    )

    r = run_one(model, tokenizer, paper["title"], paper["abstract"], cfg)
    r["title"] = paper["title"]
    r["tier"] = paper.get("tier", "")
    r["paper_id"] = paper.get("id", "")
    results.append(r)

    # Save incrementally
    out_path = os.path.join(OUTPUT_DIR, f"{args.name}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

# Final summary
print(f"\n{'='*60}")
print(f"  {args.name} — {len(results)} papers")
print(f"{'='*60}")
deltas = [r["best_score"] - r["initial_score"] for r in results]
import numpy as np
print(f"  mean Δ: {np.mean(deltas):+.3f}  std: {np.std(deltas):.3f}")
for tier in ["ORAL", "ACC", "REJ", "KEEP"]:
    tier_d = [r["best_score"] - r["initial_score"] for r in results if r["tier"] == tier]
    if tier_d:
        print(f"  {tier:<5} n={len(tier_d):>3}  mean Δ={np.mean(tier_d):+.3f}")
print(f"\nSaved to {out_path}")
