"""Multi-seed candidates-view figure: x = candidates scored (log), y = excess
select NLL over that seed's canonical prompt (log). Each seed contributes its
own unbiased E[best-of-N] curve (thin gray) — the bold line is their mean —
and each beam config becomes one point with error bars: mean ± min/max of the
per-seed excesses, at the mean candidate count. Seeds are normalized by their
OWN canonical select score (subsets differ per seed; canonical val is
identical, so per-seed excess is the comparable unit).

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_select_vs_candidates_multiseed.py [--seeds 42,43,44,45]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    SCR, load_beam_arm, load_bon_arm,
    BEAM_ARMS_X16, BEAM_ARMS_X8, BEAM_ARMS_LIGHT)
apply_style()

OUT_DIR = Path(__file__).parent
C_X16, C_X8, C_BON = "#3182bd", "#e6550d", "0.35"
N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]


def canonical_ref(seed, task):
    p = SCR / f"seed{seed}" / "readout" / "filtered_schrodi" / task / "canonical_select.json"
    return json.loads(p.read_text())["canonical"]["select"]


def exact_best_of_n(scores, n_values):
    s = np.sort(scores)
    n, i = len(s), np.arange(len(s))
    out = []
    for N in n_values:
        valid = (n - 1 - i) >= (N - 1)
        logw = np.full(n, -np.inf)
        logw[valid] = (gammaln(n - i[valid]) - gammaln(N)
                       - gammaln(n - i[valid] - N + 1)
                       - (gammaln(n + 1) - gammaln(N + 1) - gammaln(n - N + 1)))
        out.append(float((np.exp(logw) * s).sum()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45")
    ap.add_argument("--task", default="cat")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # --- best-of-N: per-seed unbiased curves + mean ---
    curves = []
    for seed in seeds:
        bon = load_bon_arm(seed, task=args.task)
        if bon is None:
            continue
        scores = np.array([s["score"] for s in bon["samples"]])
        grid = [n for n in N_GRID if n <= len(scores)]
        exc = exact_best_of_n(scores, grid) - canonical_ref(seed, args.task)
        curves.append((grid, exc))
        ax.plot(grid, exc, color=C_BON, lw=0.8, alpha=0.4, zorder=2)
    common = min(len(g) for g, _ in curves)
    grid = curves[0][0][:common]
    mean = np.mean([e[:common] for _, e in curves], axis=0)
    ax.plot(grid, mean, color="0.15", lw=2.4, zorder=3,
            label=f"best-of-N (unbiased E[min], mean of {len(curves)} seeds)")

    # --- beam configs: mean point + min/max error bars across seeds ---
    for arms, color, label, mk in (
            (BEAM_ARMS_X16, C_X16, "beam ×16", "o"),
            (BEAM_ARMS_X8, C_X8, "beam ×8", "s"),
            (BEAM_ARMS_LIGHT, C_X8, "beam 1×2, 1×4", "^")):
        xs, ys, ylo, yhi = [], [], [], []
        for arm in arms:
            per_seed = []
            for seed in seeds:
                rec = load_beam_arm(seed, arm, task=args.task)
                if rec:
                    _, n, best = rec["trajectory"][-1]
                    per_seed.append((n, best - canonical_ref(seed, args.task)))
            if not per_seed:
                continue
            ns = [n for n, _ in per_seed]
            es = [e for _, e in per_seed]
            xs.append(np.mean(ns)); ys.append(np.mean(es))
            ylo.append(np.mean(es) - min(es)); yhi.append(max(es) - np.mean(es))
        fc = "white" if arms is BEAM_ARMS_LIGHT else color
        ax.errorbar(xs, ys, yerr=[ylo, yhi], fmt=mk, ms=7, color=color,
                    markerfacecolor=fc, markeredgecolor=color, mew=1.4,
                    ecolor=color, elinewidth=1.1, capsize=2.5, ls="none",
                    zorder=4, label=f"{label} (min–max over seeds)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("candidates scored (log)")
    ax.set_ylabel("select-256 NLL − canonical, per seed (log)")
    ax.set_title(f"Verbalization budget across {len(seeds)} seeds ({args.task})")
    ax.legend(fontsize=8.5, loc="upper right")
    stem = OUT_DIR / f"select_vs_candidates_{args.task}_multiseed"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
