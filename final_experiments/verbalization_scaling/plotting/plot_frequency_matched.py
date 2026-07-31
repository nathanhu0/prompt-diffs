"""Claim-2 figure — verbalization frequency at MATCHED latent budget (2500
soft steps, the frozen SALVE budget).

Both policies spend exactly 2500 soft-optimization steps; the question is when
the decode budget is spent. LARGO with s steps/round yields K = 2500/s
interleaved verbalizations — derived by TRUNCATING the existing 10k-step runs
at their first 2500 soft steps (rounds are prefix-valid: round k's state does
not depend on num_rounds, so the truncation IS the shorter run). The
alternative spends the same K decoding AFTER training: unbiased E[best-of-K]
from the existing 2500-z best-of-1536 pool.

LARGO markers: filled = truncated winner names the trait (word-match proxy —
behavior generation was only run for full-run winners), hollow = it doesn't.
The full 10k-step runs stay out of this figure (they are the scaled-up-budget
comparison, claim 1).

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_frequency_matched.py [--seed 42] [--logy]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.optimizer_comparison_schrodi.plotting._trait import names_trait
from final_experiments.verbalization_scaling.plotting._load import (
    SCR, load_bon_arm)
apply_style()

OUT_DIR = Path(__file__).parent
BUDGET_SOFT_STEPS = 2500
# (arm dir suffix, steps/round, marker); temp07 = sampled decode (vs greedy)
LARGO_SPECS = [("steps50", 50, "o"), ("steps125", 125, "o"), ("steps250", 250, "o"),
               ("steps500", 500, "o"), ("steps1000", 1000, "o"),
               ("temp07", 250, "^")]
LARGO_COLOR, BON_COLOR = "#de2d26", "0.35"


def exact_best_of_n(scores, n_values):
    """Unbiased pool estimator (hypergeometric over order stats)."""
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


def bootstrap_iqr(scores, n_values, b=2000, seed=0):
    g = np.random.default_rng(seed)
    q25, q75 = [], []
    for N in n_values:
        wins = np.array([scores[g.choice(len(scores), N, replace=False)].min()
                         for _ in range(b)])
        q25.append(np.quantile(wins, 0.25)); q75.append(np.quantile(wins, 0.75))
    return np.array(q25), np.array(q75)


def truncated_largo(seed, arm, steps, task):
    """Winner of the 2500-soft-step prefix of a 10k-step LARGO run:
    K = 2500//steps rounds, argmin over their select scores. Returns
    (K, score, text) or None. K=2 arm (steps1000) covers 2000 steps —
    the closest whole-round prefix."""
    d = SCR / f"seed{seed}" / f"largo_{arm}" / "filtered_schrodi" / task
    src = d / "largo_results.pt"
    if not src.exists():
        return None
    K = BUDGET_SOFT_STEPS // steps
    if K < 1:
        return None
    h = torch.load(src, map_location="cpu", weights_only=False)["history"]
    hv = h["hard_val"][:K]
    if not hv:
        return None
    kbest = int(np.argmin(hv))
    return K, float(hv[kbest]), h["decoded_texts"][kbest][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42",
                    help="comma-separated; >1 seed = per-seed-excess aggregation "
                         "(mean curve, min-max bars on LARGO points)")
    ap.add_argument("--task", default="cat")
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    multi = len(seeds) > 1

    def ref_of(seed):
        if not (args.logy or multi):
            return 0.0          # single-seed raw-NLL view keeps absolute scale
        return json.loads(
            (Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")
             / f"seed{seed}" / "readout" / "filtered_schrodi" / args.task
             / "canonical_select.json").read_text())["canonical"]["select"]

    n_grid = [1, 2, 4, 8, 16, 32, 64, 128]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    curves, bands = [], []
    for seed in seeds:
        bon = load_bon_arm(seed, task=args.task)
        if bon is None:
            continue
        scores = np.array([s["score"] for s in bon["samples"]])
        r = ref_of(seed)
        curves.append(exact_best_of_n(scores, n_grid) - r)
        bands.append([q - r for q in bootstrap_iqr(scores, n_grid)])
        if multi:
            ax.plot(n_grid, curves[-1], color=BON_COLOR, lw=0.8, alpha=0.35,
                    zorder=2)
    mean = np.mean(curves, axis=0)
    ax.plot(n_grid, mean, color="0.15" if multi else BON_COLOR, lw=2.2, zorder=3,
            label=("best-of-K after training (mean of "
                   f"{len(curves)} seeds)" if multi else "best-of-K after training"))
    if not multi:
        ax.fill_between(n_grid, bands[0][0], bands[0][1], color=BON_COLOR,
                        alpha=0.18, lw=0, zorder=2)

    first_g = first_t = True
    for arm, steps, mk in LARGO_SPECS:
        pts = []
        for seed in seeds:
            got = truncated_largo(seed, arm, steps, args.task)
            if got is not None:
                K, score, text = got
                pts.append((score - ref_of(seed), names_trait(text, args.task)))
        if not pts:
            continue
        es = [e for e, _ in pts]
        filled = sum(f for _, f in pts) > len(pts) / 2   # majority names trait
        label = None
        if mk == "o" and first_g:
            label, first_g = "LARGO, K interleaved verbalizations", False
        if mk == "^" and first_t:
            label, first_t = "LARGO temp 0.7 decode", False
        ax.errorbar([K], [np.mean(es)],
                    yerr=[[np.mean(es) - min(es)], [max(es) - np.mean(es)]],
                    fmt=mk, ms=8, color=LARGO_COLOR,
                    markerfacecolor=(LARGO_COLOR if filled else "white"),
                    markeredgecolor=LARGO_COLOR, mew=1.5, ecolor=LARGO_COLOR,
                    elinewidth=1.1, capsize=2.5, ls="none", zorder=4, label=label)

    ax.set_xscale("log", base=2)
    if args.logy:
        ax.set_yscale("log")
        ax.set_ylabel("select-256 NLL − canonical"
                      + (", per seed" if multi else "") + " (log)")
    elif multi:
        ax.set_ylabel("select-256 NLL − canonical, per seed")
    else:
        ax.set_ylabel("select-256 NLL of returned prompt")
    ax.set_xlabel("decode budget K = 2500 / (steps per verbalization)  (log)")
    seed_str = (f"{len(seeds)} seeds" if multi else f"seed {seeds[0]}")
    ax.set_title(f"Same 2500-step latent budget: interleave K verbalizations,\n"
                 f"or decode at the end? ({args.task}, {seed_str})")
    ax.legend(fontsize=9, loc="upper right")
    stem = OUT_DIR / (f"frequency_matched_{args.task}_"
                      + ("multiseed" if multi else f"seed{seeds[0]}")
                      + ("_logy" if args.logy else ""))
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
