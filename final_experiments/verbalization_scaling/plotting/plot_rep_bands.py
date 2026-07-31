"""Headline replicate figure (seed-42 z): exact best-of-N winner DISTRIBUTION
band vs 16-replicate empirical beam boxes.

BoN side: from the fixed 4608 pool the hypergeometric rank-win weights give
the full distribution of the winner at every N — mean line + / − 1 sd band,
exactly, no sampling. val uses the top-1024 val-scored head (renormalized);
select uses the full pool.

Beam side: per config, every replicate's winner metric (readout_<arm>_rep*.json
plus the original un-suffixed run) as jittered points + mean ± sd bar at the
mean candidates-scored x.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_rep_bands.py --metric val
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments._style import (
    PANEL2, apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    BEAM_ARMS_X16, BEAM_ARMS_X8, load_bon_arm)
from final_experiments.verbalization_scaling.plotting.plot_bon_beam_curves import (
    cell_dir, refs_of, hyp_logw)
apply_style()

OUT_DIR = Path(__file__).parent
C_X16, C_X8, C_BON = "#3182bd", "#e6550d", "0.35"
SEED, TASK = 42, "cat"


def load_pool_vals(metric):
    """Pool metric values in select-rank order (NaN where val unscored)."""
    if metric == "select":
        bon = load_bon_arm(SEED, task=TASK)
        return np.sort([x["score"] for x in bon["samples"]])
    d = json.loads((cell_dir(SEED, TASK)
                    / "readout_best_of_1536_exact_val.json").read_text())
    n, m = d.get("n", 1536), d["m"]
    vals = np.full(n, np.nan)
    vals[:m] = [d["val_by_rank"][str(r)] for r in range(m)]
    return vals


def bon_band(metric, n_grid):
    """(mean, sd) of the winner's metric at each N — exact over the pool."""
    vals = load_pool_vals(metric)
    n = len(vals)
    fin = np.isfinite(vals)
    means, sds = [], []
    for N in n_grid:
        if N > n:
            break
        w = np.exp(hyp_logw(n, N, n)); w = w[fin] / w[fin].sum()
        v = vals[fin]
        mu = float((w * v).sum())
        means.append(mu)
        sds.append(float(np.sqrt(max((w * v ** 2).sum() - mu ** 2, 0.0))))
    return np.array(means), np.array(sds)


def bon_bootstrap_ci(metric, n_grid, B=1000, seed=1):
    """95% CI of the exact winner-mean curve under pool resampling.

    The winner-mean is exact GIVEN the pool; this bootstraps the pool itself
    (resample n samples with replacement, recompute the exact curve) to get
    the uncertainty from pool finiteness."""
    vals = load_pool_vals(metric)
    n = len(vals)
    grid = np.array([N for N in n_grid if N <= n])
    W = np.stack([np.exp(hyp_logw(n, int(N), n)) for N in grid])  # (G, n)
    rng = np.random.default_rng(seed)
    curves = np.empty((B, len(grid)))
    for b in range(B):
        vb = vals[np.sort(rng.integers(0, n, n))]
        fin = np.isfinite(vb)
        num = W[:, fin] @ vb[fin]
        curves[b] = num / W[:, fin].sum(axis=1)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    return lo, hi


def rep_records(arm):
    d = cell_dir(SEED, TASK)
    reps = [p for p in sorted(d.glob(f"readout_{arm}_rep*.json"))
            if re.fullmatch(rf"readout_{arm}_rep\d+\.json", p.name)]
    recs = []
    for p in reps + [d / f"readout_{arm}.json"]:
        if not p.exists():
            continue
        try:
            r = json.loads(p.read_text())
            recs.append({"n": r["n_proposals"], "val": r["nll"]["val"],
                         "select": r["extra"]["select_score"]})
        except (json.JSONDecodeError, KeyError):
            print(f"  skipping incomplete {p.name}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["select", "val"], default="val")
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--empty", action="store_true",
                    help="draw the empty-prompt reference line")
    ap.add_argument("--no-canonical", action="store_true",
                    help="drop the canonical-prompt reference line")
    ap.add_argument("--no-refs", action="store_true",
                    help="drop ALL reference lines (axis snaps to the data)")
    ap.add_argument("--ci", action="store_true",
                    help="mean-claim mode: beam bars = 95%% CI of the mean, "
                         "BoN mean line + 95%% bootstrap-over-pool CI band")
    ap.add_argument("--family", choices=["both", "x16", "x8"], default="both")
    ap.add_argument("--match-budget", action="store_true",
                    help="truncate the BoN curve at the largest beam config's "
                         "mean candidate count (matched-compute region only)")
    args = ap.parse_args()
    metric = args.metric

    x16 = (BEAM_ARMS_X16, C_X16, "Beam search, width $= 16$", "o")
    x8 = (BEAM_ARMS_X8, C_X8, "Beam search, width $= 8$", "s")
    families = {"both": (x16, x8), "x16": (x16,), "x8": (x8,)}[args.family]

    lo = 1 if metric == "select" else 16
    hi = 4608
    if args.match_budget:
        hi = 1.1 * max(np.mean([r["n"] for r in rep_records(arm)])
                       for arms, *_ in families for arm in arms)
    n_grid = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi),
                                            60)).astype(int))
    mu, sd = bon_band(metric, n_grid)
    fig, ax = plt.subplots(figsize=PANEL2)
    if args.ci:
        lo, hi = bon_bootstrap_ci(metric, n_grid)
        ax.fill_between(n_grid[: len(lo)], lo, hi, color=C_BON, alpha=0.22,
                        lw=0, zorder=1, label="_nolegend_")
        ax.plot(n_grid[: len(mu)], mu, color=C_BON, lw=2.0, zorder=2,
                label="Best-of-$N$")
    else:
        ax.fill_between(n_grid[: len(mu)], mu - sd, mu + sd, color=C_BON,
                        alpha=0.18, lw=0, zorder=1,
                        label="best-of-N winner ± 1 sd (exact)")
        ax.plot(n_grid[: len(mu)], mu, color=C_BON, lw=2.0, zorder=2,
                label="best-of-N winner mean (exact)")

    rng = np.random.default_rng(0)
    for arms, color, label, mk in families:
        xs, ms, sds, nrep = [], [], [], []
        for arm in arms:
            recs = rep_records(arm)
            if not recs:
                continue
            ys = np.array([r[metric] for r in recs])
            x0 = np.mean([r["n"] for r in recs])
            jit = x0 * np.exp(rng.normal(0, 0.03, len(ys)))
            ax.scatter(jit, ys, color=color, s=10, alpha=0.30, zorder=3, lw=0)
            half = ys.std(ddof=1)
            if args.ci:
                from scipy.stats import t as t_dist
                half *= t_dist.ppf(0.975, len(ys) - 1) / np.sqrt(len(ys))
            xs.append(x0); ms.append(ys.mean()); sds.append(half)
            nrep.append(len(ys))
        ax.plot(xs, ms, color=color, lw=1.0, alpha=0.55, zorder=3)
        ax.errorbar(xs, ms, yerr=sds, fmt=mk, ms=7, color=color,
                    markeredgecolor=color, mew=1.4, ecolor=color,
                    elinewidth=1.4, capsize=3, ls="none", zorder=4,
                    label=label)

    refs = refs_of(SEED, TASK)
    ref_lines = [("soft", "#31a354", "soft prompt", "--")]
    if not args.no_canonical:
        ref_lines.insert(0, ("canonical", "0.5", "canonical prompt", ":"))
    if args.no_refs:
        ref_lines = []
    if args.empty:
        ref_lines.append(("empty", "#756bb1", "empty prompt", "-."))
    for key, color, name, ls in ref_lines:
        y = refs.get(key, {}).get(metric)
        if y is not None:
            ax.axhline(y, color=color, lw=1.0, ls=ls, zorder=1)
            side = (0.0, "left", 4) if key == "empty" else (1.0, "right", -4)
            ax.annotate(name, xy=(side[0], y), xycoords=("axes fraction", "data"),
                        xytext=(side[2], 3), textcoords="offset points",
                        ha=side[1], fontsize=9, color=color)

    from matplotlib.ticker import (FormatStrFormatter, MultipleLocator,
                                   NullFormatter, ScalarFormatter)
    ax.set_xscale("log")
    xticks = [t for t in (10, 30, 100, 300, 1000, 3000)
              if n_grid[0] <= t <= 1.3 * n_grid[-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    if args.logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ticks = [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54]
        ax.set_yticks([t for t in ticks if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]])
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.005))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Candidates scored")
    ax.set_ylabel(("Selection NLL" if metric == "select" else "Validation NLL")
                  + (" (log)" if args.logy else ""))
    ax.legend(fontsize=8.5, loc="upper right")
    suffix = ("_logy" if args.logy else "") + ("_empty" if args.empty else "") \
        + ("_nocanon" if args.no_canonical else "") \
        + ("_norefs" if args.no_refs else "") + ("_ci" if args.ci else "") \
        + ("" if args.family == "both" else f"_{args.family}") \
        + ("_matched" if args.match_budget else "")
    stem = OUT_DIR / f"rep_bands_{metric}_{TASK}_seed{SEED}{suffix}"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
