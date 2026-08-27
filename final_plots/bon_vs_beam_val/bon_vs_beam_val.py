"""SALVE-motivation figure: candidates scored vs validation NLL, one soft
prompt (seed 42, cat) with many decodes.

Best-of-N side: from the fixed 4608-decode pool, hypergeometric rank-win
weights give the exact winner-mean at every N (val from the top-1024
val-scored head, renormalized); band = ±1 SE from bootstrapping the pool
itself. Beam side: per config, mean over replicates ± 1 SEM. x truncated at
the largest beam config's mean candidate count (matched-compute region
only); no reference lines.

Fork of final_experiments/verbalization_scaling/plotting/plot_rep_bands.py
(`--metric val --no-refs --ci --match-budget` variant) with replicate dots
dropped and error bars switched from 95% CI to ±1 SEM.
Self-contained: data paths hardcoded, style inlined.

  uv run python final_plots/bon_vs_beam_val/bon_vs_beam_val.py
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, NullFormatter
from scipy.special import gammaln

OUT_DIR = Path(__file__).parent
DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling"
                "/seed42/readout/filtered_schrodi/cat")
BEAM_ARMS_X16 = ["beam_1x16", "beam_2x16", "beam_4x16", "beam_8x16"]
BEAM_ARMS_X8 = ["beam_1x8", "beam_2x8", "beam_4x8", "beam_8x8"]
C_X16, C_X8, C_BON = "#3182bd", "#e6550d", "0.35"

plt.rcParams.update({
    "axes.labelsize":     13,
    "axes.titlesize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "0.7",
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "figure.dpi":         200,
    "font.family":        "DejaVu Sans",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})
FIGSIZE = (5.3, 3.5)  # 2-panel-row design width, wide aspect; 0.48\textwidth


def hyp_logw(n, N, m):
    """log P(best of a uniform N-subset of n sorted samples is rank r), r<m."""
    r = np.arange(m)
    valid = (n - 1 - r) >= (N - 1)
    logw = np.full(m, -np.inf)
    logw[valid] = (gammaln(n - r[valid]) - gammaln(N) - gammaln(n - r[valid] - N + 1)
                   - (gammaln(n + 1) - gammaln(N + 1) - gammaln(n - N + 1)))
    return logw


def load_pool_vals():
    """Pool val NLLs in select-rank order (NaN where val unscored)."""
    d = json.loads((DATA_DIR / "readout_best_of_1536_exact_val.json").read_text())
    n, m = d.get("n", 1536), d["m"]
    vals = np.full(n, np.nan)
    vals[:m] = [d["val_by_rank"][str(r)] for r in range(m)]
    return vals


def bon_mean(vals, n_grid):
    """Exact winner-mean val at each N over the pool."""
    n = len(vals)
    fin = np.isfinite(vals)
    means = []
    for N in n_grid:
        if N > n:
            break
        w = np.exp(hyp_logw(n, N, n)); w = w[fin] / w[fin].sum()
        means.append(float((w * vals[fin]).sum()))
    return np.array(means)


def bon_bootstrap_se(vals, n_grid, B=1000, seed=1):
    """±1 SE band of the exact winner-mean curve under pool resampling.

    The winner-mean is exact GIVEN the pool; this bootstraps the pool itself
    (resample n samples with replacement, recompute the exact curve) and
    returns the per-N std of the bootstrap curves — the standard error from
    pool finiteness."""
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
    return curves.std(axis=0)


def rep_records(arm):
    """All replicates' winners for one beam config (rep* + the original run)."""
    reps = [p for p in sorted(DATA_DIR.glob(f"readout_{arm}_rep*.json"))
            if re.fullmatch(rf"readout_{arm}_rep\d+\.json", p.name)]
    recs = []
    for p in reps + [DATA_DIR / f"readout_{arm}.json"]:
        if not p.exists():
            continue
        try:
            r = json.loads(p.read_text())
            recs.append({"n": r["n_proposals"], "val": r["nll"]["val"]})
        except (json.JSONDecodeError, KeyError):
            print(f"  skipping incomplete {p.name}")
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ci", action="store_true",
                    help="drop the BoN bootstrap band and beam CI bars")
    args = ap.parse_args()

    families = ((BEAM_ARMS_X16, C_X16, "Beam Search (Width = 16)", "o"),
                (BEAM_ARMS_X8, C_X8, "Beam Search (Width = 8)", "s"))

    # matched-compute region: truncate BoN at the largest beam config's budget
    hi = 1.1 * max(np.mean([r["n"] for r in rep_records(arm)])
                   for arms, *_ in families for arm in arms)
    n_grid = np.unique(np.round(np.logspace(np.log10(16), np.log10(hi),
                                            60)).astype(int))
    vals = load_pool_vals()
    mu = bon_mean(vals, n_grid)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if not args.no_ci:
        se = bon_bootstrap_se(vals, n_grid)
        m = mu[: len(se)]
        ax.fill_between(n_grid[: len(se)], m - se, m + se, color=C_BON,
                        alpha=0.22, lw=0, zorder=1, label="_nolegend_")
    ax.plot(n_grid[: len(mu)], mu, color=C_BON, lw=2.0, zorder=2,
            label="Best-of-$N$")

    for arms, color, label, mk in families:
        xs, ms, sds = [], [], []
        for arm in arms:
            recs = rep_records(arm)
            if not recs:
                continue
            ys = np.array([r["val"] for r in recs])
            half = ys.std(ddof=1) / np.sqrt(len(ys))
            xs.append(np.mean([r["n"] for r in recs]))
            ms.append(ys.mean()); sds.append(half)
        ax.plot(xs, ms, color=color, lw=1.0, alpha=0.55, zorder=3)
        ax.errorbar(xs, ms, yerr=None if args.no_ci else sds, fmt=mk, ms=7,
                    color=color, markeredgecolor=color, mew=1.4, ecolor=color,
                    elinewidth=1.4, capsize=3, ls="none", zorder=4,
                    label=label)

    ax.set_xscale("log")
    xticks = [t for t in (10, 30, 100, 300, 1000, 3000)
              if n_grid[0] <= t <= 1.3 * n_grid[-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Candidates scored")
    ax.set_ylabel("NLL")
    ax.legend(fontsize=11, loc="upper right")
    stem = OUT_DIR / ("bon_vs_beam_val_cat_seed42"
                      + ("_noci" if args.no_ci else ""))
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext))
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
