"""Left panel of the science-of-SALVE triptych: prefix quality predicts
extension quality, measured on already-logged beam nodes (zero new compute).

Every scored beam node is a prompt prefix; every scored child is that prefix
plus one sentence. Scatter parent select NLL vs child select NLL over all
expansion pairs across the seed-42 replicate wave (17 reps x 8 configs),
with Spearman rho. Also prints the lineage version (depth-1 ancestor score
vs final-depth leaf score) as a sanity number.

Caveat carried in the caption, not hidden: parents at depth >= 2 were
selected by the search (top-k), which RESTRICTS the x-range and attenuates
rho — the conservative direction.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_prefix_predictiveness.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting.plot_bon_beam_curves import (
    cell_dir)
apply_style()

OUT_DIR = Path(__file__).parent
SEED, TASK = 42, "cat"


def collect_pairs():
    d = cell_dir(SEED, TASK)
    pt_files = [p for p in sorted(d.glob("readout_beam_*_results.pt"))
                if re.fullmatch(r"readout_beam_\dx\d+(_rep\d+)?_results\.pt",
                                p.name)]
    pairs, lineage = [], []
    for p in pt_files:
        nodes = torch.load(p, map_location="cpu",
                           weights_only=False)["nodes"]
        by_idx = {nd["idx"]: nd for nd in nodes}
        max_depth = max(nd["depth"] for nd in nodes)
        for nd in nodes:
            if nd["parent"] is None or nd["score"] is None:
                continue
            par = by_idx[nd["parent"]]
            if par["score"] is None:
                continue
            pairs.append((par["score"], nd["score"], nd["depth"]))
            if nd["depth"] == max_depth:
                anc = nd
                while by_idx[anc["parent"]]["depth"] > 1:
                    anc = by_idx[anc["parent"]]
                lineage.append((anc["score"], nd["score"]))
    return np.array(pairs), np.array(lineage), len(pt_files)


def main():
    pairs, lineage, n_runs = collect_pairs()
    # depth-1 pairs all share the run's root as parent (constant x) — drop
    pairs = pairs[pairs[:, 2] >= 2]
    x, y, depth = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    rho, _ = spearmanr(x, y)
    r, _ = pearsonr(x, y)
    print(f"{n_runs} runs, {len(pairs)} expansion pairs")
    print(f"parent vs child select NLL: spearman {rho:.3f}, pearson {r:.3f}")
    for dd in sorted(set(depth.astype(int))):
        m = depth == dd
        print(f"  depth {dd}: n={m.sum()}, spearman "
              f"{spearmanr(x[m], y[m])[0]:.3f}")
    if len(lineage):
        lr, _ = spearmanr(lineage[:, 0], lineage[:, 1])
        print(f"lineage (depth-1 ancestor vs final leaf): n={len(lineage)}, "
              f"spearman {lr:.3f}")

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    hb = ax.hexbin(x, y, gridsize=55, cmap="Blues", mincnt=1, linewidths=0.2)
    lim = (min(np.percentile(x, 0.5), np.percentile(y, 0.5)) - 0.002,
           max(np.percentile(x, 99), np.percentile(y, 99)) + 0.002)
    ax.plot(lim, lim, color="0.4", lw=0.8, ls="--", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("prefix selection NLL")
    ax.set_ylabel("prefix + one sentence selection NLL")
    ax.annotate(f"Spearman $\\rho$ = {rho:.2f}\n({len(pairs):,} expansions)",
                xy=(0.04, 0.96), xycoords="axes fraction", va="top",
                fontsize=10)
    fig.colorbar(hb, ax=ax, label="expansions", shrink=0.85)
    stem = OUT_DIR / f"prefix_predictiveness_{TASK}_seed{SEED}"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
