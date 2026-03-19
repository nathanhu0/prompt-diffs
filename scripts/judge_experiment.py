"""Run LLM judge on experiment results.

Usage:
    python scripts/judge_experiment.py --input revise_fix_lr1e4_20260318_041146.json
"""
import sys
sys.path.insert(0, "/juice2/u/nathu/latent-rewrite")
import argparse
import json
import os
import asyncio
import numpy as np
from iclr_judge import judge_batch_async

INPUT_DIR = "/nlp/scr/nathu/latent_rewrite/results"
OUTPUT_DIR = "/nlp/scr/nathu/latent_rewrite/results"

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="input JSON filename")
args = parser.parse_args()

#%% Load
input_path = os.path.join(INPUT_DIR, args.input)
with open(input_path) as f:
    data = json.load(f)
results = data["results"]
print(f"Loaded {len(results)} papers from {args.input}")

#%% Judge each paper's best rewrite (parallel)
# Split into changed vs unchanged
changed = [(i, r) for i, r in enumerate(results) if r["best_text"] != r["original"]]
unchanged = [(i, r) for i, r in enumerate(results) if r["best_text"] == r["original"]]

# Mark unchanged as skipped
for i, r in unchanged:
    r["judge"] = {"legibility": "totally_legible", "sentences": [],
                  "summary": {"n_consistent": 0, "n_new": 0, "n_contradicts": 0, "n_total": 0},
                  "skipped": True}

print(f"Judging {len(changed)} changed papers ({len(unchanged)} unchanged, skipped)")

# Run async
pairs = [(r["original"], r["best_text"]) for _, r in changed]
judge_results = asyncio.run(judge_batch_async(pairs, max_concurrent=20))

for (i, r), jr in zip(changed, judge_results):
    if isinstance(jr, Exception):
        r["judge"] = {"error": str(jr)}
    else:
        r["judge"] = jr

# Save
out_name = args.input.replace(".json", "_judged.json")
out_path = os.path.join(OUTPUT_DIR, out_name)
with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

#%% Summary
judged = [r for r in results if "judge" in r and "error" not in r["judge"] and not r["judge"].get("skipped")]
skipped = [r for r in results if r.get("judge", {}).get("skipped")]
errors = [r for r in results if "error" in r.get("judge", {})]

print(f"\nJudged: {len(judged)}, Skipped (no change): {len(skipped)}, Errors: {len(errors)}")

if judged:
    # Legibility
    leg_counts = {}
    for r in judged:
        leg = r["judge"].get("legibility", "unknown")
        leg_counts[leg] = leg_counts.get(leg, 0) + 1
    print(f"\nLegibility:")
    for k, v in sorted(leg_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(judged)*100:.0f}%)")

    # Faithfulness
    scores = []
    for r in judged:
        s = r["judge"]["summary"]
        if s["n_total"] > 0:
            faith = s["n_supported"] / s["n_total"]
            scores.append(faith)
    if scores:
        print(f"\nFaithfulness (fraction of sentences supported):")
        print(f"  mean: {np.mean(scores):.3f}  median: {np.median(scores):.3f}")
        print(f"  min: {np.min(scores):.3f}  max: {np.max(scores):.3f}")

    total_sentences = sum(r["judge"]["summary"]["n_total"] for r in judged)
    total_unsupported = sum(r["judge"]["summary"]["n_unsupported"] for r in judged)
    print(f"\n  Total sentences: {total_sentences}")
    print(f"  Unsupported: {total_unsupported} ({total_unsupported/total_sentences*100:.1f}%)")

print(f"\nSaved to {out_path}")
