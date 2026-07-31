"""Paired within-seed comparison: beam endpoint minus the exact best-of-N
value AT THE SAME candidate count, per seed. Everything seed-specific
(z quality, gap-level heterogeneity) cancels; what remains is the
readout-method effect + readout sampling noise.

Two panels (select, val). Per-seed light markers, bold mean ± SEM, zero line.
Also prints per-arm stats: mean diff, sd, seed wins, paired-t p-value.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_paired_diff.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    BEAM_ARMS_X16, BEAM_ARMS_X8)
from final_experiments.verbalization_scaling.plotting.plot_bon_beam_curves import (
    ALL_SEEDS, select_curve, val_curve, beam_endpoint)
apply_style()

OUT_DIR = Path(__file__).parent
C_X16, C_X8 = "#3182bd", "#e6550d"
TASK = "cat"


def bon_at(seed, metric, N):
    fn = select_curve if metric == "select" else val_curve
    c = fn(seed, TASK, np.array([N]))
    return None if c is None or not len(c) else float(c[0])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    print(f"{'metric':7s} {'arm':8s} {'N~':>5s} {'mean':>8s} {'sd':>7s} "
          f"{'wins':>5s} {'p(t)':>6s}")
    for ax, metric in zip(axes, ["select", "val"]):
        for arms, color, label, mk in ((BEAM_ARMS_X16, C_X16, "beam ×16", "o"),
                                       (BEAM_ARMS_X8, C_X8, "beam ×8", "s")):
            xs, ms, sems = [], [], []
            for arm in arms:
                diffs, ns = [], []
                for seed in ALL_SEEDS:
                    got = beam_endpoint(seed, arm, TASK, metric)
                    if got is None:
                        continue
                    n, y = got
                    b = bon_at(seed, metric, int(round(n)))
                    if b is None:
                        continue
                    diffs.append(y - b)
                    ns.append(n)
                if len(diffs) < 2:
                    continue
                d = np.array(diffs)
                N = np.mean(ns)
                t, p = stats.ttest_1samp(d, 0.0)
                print(f"{metric:7s} {arm:8s} {N:5.0f} {d.mean():+8.4f} "
                      f"{d.std(ddof=1):7.4f} {int((d < 0).sum())}/{len(d):d} "
                      f"{p:6.3f}")
                ax.scatter([N] * len(d), d, color=color, s=14, alpha=0.35,
                           zorder=2)
                xs.append(N); ms.append(d.mean())
                sems.append(d.std(ddof=1) / np.sqrt(len(d)))
            ax.errorbar(xs, ms, yerr=sems, fmt=mk, ms=7, color=color,
                        mew=1.4, elinewidth=1.3, capsize=3, ls="none",
                        zorder=4, label=f"{label} (mean ± SEM)")
        ax.axhline(0.0, color="0.3", lw=1.0, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("candidates scored (log)")
        ax.set_ylabel(f"{metric} NLL: beam − best-of-same-N, per seed")
        ax.set_title(metric)
        ax.legend(fontsize=8.5, loc="lower left")
    fig.suptitle(f"Paired within-seed: beam vs exact best-of-N at matched "
                 f"budget ({TASK}, {len(ALL_SEEDS)} soft prompts)", y=0.99)
    fig.tight_layout()
    stem = OUT_DIR / f"paired_diff_{TASK}"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
