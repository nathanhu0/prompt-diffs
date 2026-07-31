"""Fit a scaling law to the best-of-N select curve: E[best-of-N] = A + c*N^(-b).

Uses the EXACT expected minimum of N i.i.d. draws from the empirical score
distribution (order-statistics closed form, with replacement — the infinite-
pool idealization), fit over N <= n_fit_max (default 512) where the finite
1536-sample pool doesn't yet distort the tail. A is the extrapolated
infinite-N asymptote of naive sampling — the number to compare beam endpoints
and the canonical floor against.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/fit_bon_scaling.py [--seed 42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    load_beam_arm, load_bon_arm, BEAM_ARMS_X16, BEAM_ARMS_X8, BEAM_ARMS_LIGHT)
apply_style()

OUT_DIR = Path(__file__).parent


def exact_best_of_n(scores, n_values):
    """UNBIASED estimator of E[min of N i.i.d. draws] from a pool of n
    samples: hypergeometric weights over order statistics (the pass@k
    analog). P(rank i is the min of a uniform N-subset w/o replacement)
    = C(n-1-i, N-1)/C(n, N), 0-indexed ascending. Every N-subset of the
    pool is itself an i.i.d. best-of-N run, so this is exact for N <= n
    — and says nothing beyond N = n (the asymptote is not identifiable)."""
    from scipy.special import gammaln
    s = np.sort(scores)
    n = len(s)
    i = np.arange(n)
    out = []
    for N in n_values:
        # log C(n-1-i, N-1) - log C(n, N); zero weight where n-1-i < N-1
        valid = (n - 1 - i) >= (N - 1)
        logw = np.full(n, -np.inf)
        logw[valid] = (gammaln(n - i[valid]) - gammaln(N) - gammaln(n - i[valid] - N + 1)
                       - (gammaln(n + 1) - gammaln(N + 1) - gammaln(n - N + 1)))
        out.append(float((np.exp(logw) * s).sum()))
    return np.array(out)


def law(N, A, c, b):
    return A + c * np.power(N, -b)


def fit_tail(n_fit, y_fit, a_lo, a_hi, grid=600):
    """Profile fit of y = A + c*N^(-b): sweep A, linear-fit log(y - A) on
    log N, pick A by log-space SSE. Log-space residuals weight the high-N
    tail (relative error), the regime that determines the asymptote; A is
    constrained to [a_lo, a_hi] (a_hi = pool min — the lower endpoint of the
    sampling distribution cannot exceed an observed sample)."""
    logN = np.log(n_fit)
    best = None
    for A in np.linspace(a_lo, a_hi, grid):
        ex = y_fit - A
        if (ex <= 0).any():
            continue
        coef = np.polyfit(logN, np.log(ex), 1)
        sse = float(((np.polyval(coef, logN) - np.log(ex)) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, A, np.exp(coef[1]), -coef[0])
    _, A, c, b = best
    return A, c, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="cat")
    ap.add_argument("--n-fit-min", type=int, default=32)
    ap.add_argument("--n-fit-max", type=int, default=1024)
    args = ap.parse_args()

    bon = load_bon_arm(args.seed, task=args.task)
    scores = np.array([s["score"] for s in bon["samples"]])
    ref = json.loads((Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")
                      / f"seed{args.seed}" / "readout" / "filtered_schrodi"
                      / args.task / "canonical_select.json").read_text()
                     )["canonical"]["select"]

    n_fit = np.unique(np.round(np.logspace(np.log10(args.n_fit_min),
                                           np.log10(args.n_fit_max), 40)
                               ).astype(int))
    y_fit = exact_best_of_n(scores, n_fit)
    A, c, b = fit_tail(n_fit, y_fit, a_lo=ref, a_hi=float(scores.min()) - 1e-5)
    resid = np.abs(law(n_fit, A, c, b) - y_fit).max()
    print(f"tail fit over N in [{args.n_fit_min}, {args.n_fit_max}] "
          f"(log-space profile, A <= pool min):")
    print(f"  E[best-of-N] = {A:.4f} + {c:.4f} * N^(-{b:.3f})   "
          f"(max|resid|={resid:.5f})")
    print(f"asymptote A = {A:.4f}  (excess over canonical: {A - ref:.4f})")
    print(f"pool min    = {scores.min():.4f}")

    # Plot the unbiased curve only where it is identified (N <= pool size);
    # the fit is drawn over the same range as a slope descriptor — no
    # asymptote line, no beyond-pool extrapolation (not identifiable, and a
    # fitted A can even sit above prompts other methods provably sample).
    n_show = np.unique(np.round(np.logspace(0, np.log10(len(scores)), 60)).astype(int))
    n_plot = np.unique(np.round(np.logspace(np.log10(args.n_fit_min),
                                            np.log10(len(scores)), 40)).astype(int))
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(n_show, exact_best_of_n(scores, n_show) - ref, color="0.35", lw=2.0,
            label="best-of-N (unbiased E[min])")
    ax.plot(n_plot, law(n_plot, A, c, b) - ref, color="0.35", lw=1.0, ls="--",
            label=f"tail fit ${A:.3f} + {c:.3f}\\,N^{{-{b:.2f}}}$")
    for arms, color, label, mk in (
            (BEAM_ARMS_X16, "#3182bd", "beam ×16", "o"),
            (BEAM_ARMS_X8, "#e6550d", "beam ×8", "s"),
            (BEAM_ARMS_LIGHT, "#e6550d", "beam 1×2, 1×4", "^")):
        pts = []
        for arm in arms:
            rec = load_beam_arm(args.seed, arm, task=args.task)
            if rec:
                _, n, best = rec["trajectory"][-1]
                pts.append((n, best - ref))
        if not pts:
            continue
        fc = "white" if arms is BEAM_ARMS_LIGHT else color
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=65,
                   facecolors=fc, edgecolors=color, linewidths=1.4,
                   marker=mk, zorder=4, label=label)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("candidates scored N (log)")
    ax.set_ylabel("select-256 NLL − canonical (log)")
    ax.set_title(f"Best-of-N scaling law vs beam ({args.task}, seed {args.seed})")
    ax.legend(loc="upper right", fontsize=9)
    stem = OUT_DIR / f"bon_scaling_law_{args.task}_seed{args.seed}"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
