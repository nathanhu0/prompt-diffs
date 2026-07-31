"""Exact (bootstrap-free) E[val of best-of-N winner] per seed.

The select-argmin winner of a uniform N-subset has rank distribution
P(rank r wins) = C(n-1-r, N-1)/C(n, N) over the select-sorted pool — so the
expected val of the returned prompt is an exact weighted sum once the top-M
ranks are val-scored. P(winner rank >= M) < 1% for N >= 16 at M=512, so the
curve is reported for N >= 16 with weights renormalized over the scored head
(negligible truncation bias there). Reuses vals cached by the B=8 bootstrap
(readout_best_of_1536_bootstrap.pt) and only scores the missing ranks.

Writes readout_best_of_1536_exact_val.json: {"m": M, "n_grid": [...],
"val_curve": [...], "val_by_rank": {...}} next to the pool.

  ebatch exact_bon_val slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. \\
    uv run python final_experiments/verbalization_scaling/plotting/exact_bon_val.py --seeds 42,43,44,45"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from core.subliminal.data import load_splits
from final_experiments.optimizer_comparison.run_comparison import build_objective
from final_experiments.verbalization_scaling.plotting._load import SCR, load_bon_arm

MODEL = "Qwen/Qwen2.5-7B-Instruct"
# M scales with pool size: at n=1536 the top-512 head truncates <1% of winner
# mass at N=16; at the extended n=4608 pool that jumps to ~15%, so extended
# pools score the top 1024 (back to ~2%).
M_BASE, M_EXT = 512, 1024
N_GRID = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
          2304, 3072, 4608]


def rank_win_logw(n, N, m):
    """log P(rank r wins an N-subset), r = 0..m-1 (0-indexed, select-ascending)."""
    r = np.arange(m)
    valid = (n - 1 - r) >= (N - 1)
    logw = np.full(m, -np.inf)
    logw[valid] = (gammaln(n - r[valid]) - gammaln(N) - gammaln(n - r[valid] - N + 1)
                   - (gammaln(n + 1) - gammaln(N + 1) - gammaln(n - N + 1)))
    return logw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45")
    ap.add_argument("--task", default="cat")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = f"cuda:{args.gpu}"

    model, tokenizer, _ = load_frozen_lm(MODEL, device=device)
    xy = load_splits(args.task, 10000, 500, 1500, prefill=None, seed=42,
                     model=MODEL, method="filtered_schrodi")
    objective = build_objective(model, tokenizer, xy, 128, "{SOFT}")

    for seed in seeds:
        bon = load_bon_arm(seed, task=args.task)
        if bon is None:
            print(f"seed{seed}: no pool; skipping", flush=True)
            continue
        samples = bon["samples"]
        order = np.argsort([s["score"] for s in samples])      # select-ascending
        m = min(M_EXT if len(order) > 1536 else M_BASE, len(order))

        # cached vals from the bootstrap sidecar (keyed by SAMPLE index)
        cache = {}
        bp = SCR / f"seed{seed}" / "readout" / "filtered_schrodi" / args.task \
            / "readout_best_of_1536_bootstrap.pt"
        if bp.exists():
            cache = {int(k): v["val"] for k, v in
                     torch.load(bp, map_location="cpu",
                                weights_only=False)["val_by_idx"].items()}

        vals = np.empty(m)
        n_new = 0
        for rank in range(m):
            idx = int(order[rank])
            if idx in cache:
                vals[rank] = cache[idx]
            else:
                vals[rank] = float(objective.hard_loss(
                    samples[idx]["text"], "val", mini_batch_size=24))
                n_new += 1
            if rank % 100 == 99:
                print(f"  seed{seed}: rank {rank + 1}/{m} scored ({n_new} new)",
                      flush=True)

        curve = []
        for N in N_GRID:
            if N > len(order):
                continue
            logw = rank_win_logw(len(order), N, m)
            w = np.exp(logw)
            curve.append(float((w * vals).sum() / w.sum()))    # renormalized head
        out = bp.parent / "readout_best_of_1536_exact_val.json"
        out.write_text(json.dumps({
            "m": m, "n": len(order),
            "n_grid": [n for n in N_GRID if n <= len(order)],
            "val_curve": curve,
            "val_by_rank": {int(r): vals[r] for r in range(m)},
            "sample_idx_by_rank": {int(r): int(order[r]) for r in range(m)},
        }))
        print(f"seed{seed}: wrote {out.name} ({n_new} new val scorings); "
              f"val@N16={curve[0]:.4f} @N1536={curve[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
