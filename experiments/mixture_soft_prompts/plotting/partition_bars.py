"""How each mixture partitioned the data: per cell, one stacked bar PER
MEMBER — bar height = number of val examples routed to that member (pure
argmin), split into trait-source (colored) vs filler (gray) segments.
Rows = animals, cols = diluters, bar groups at each trait fraction.
Members within a cell are sorted by trait count (descending).

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/partition_bars.py
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mixture_soft_prompts.plotting.dilution_grid import (
    RUN_ROOT, cell_info)

OUT_DIR = Path(__file__).parent
ANIMALS = ["cat", "dog", "eagle", "owl"]
DILUTERS = ["control", "random"]
TRAIT_COLOR = "#E69F00"
FILLER_COLOR = "0.75"
BAR_W = 0.016


def main():
    cells = {}
    for pt in sorted(RUN_ROOT.glob("dil_*/mixture.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        info = cell_info(d)
        if info is None:
            continue
        primary, diluter, f = info
        conf = d["history"]["evals"][-1]["confusion"]
        conf = [list(r) + [0] * (2 - len(r)) for r in conf]  # pure: 1 label col
        if diluter == "-":                       # pure anchors -> both cols
            for dl in DILUTERS:
                cells[(primary, dl, 1.0)] = conf
        else:
            cells[(primary, diluter, f)] = conf

    fig, axes = plt.subplots(len(ANIMALS), len(DILUTERS),
                             figsize=(13, 3.0 * len(ANIMALS)),
                             sharex=True, sharey=True)
    for (animal, diluter, f), conf in cells.items():
        ax = axes[ANIMALS.index(animal)][DILUTERS.index(diluter)]
        members = sorted(conf, key=lambda r: -r[0])   # trait count desc
        k = len(members)
        for j, (n_trait, n_fill) in enumerate(members):
            x = f + (j - (k - 1) / 2) * BAR_W
            ax.bar(x, n_trait, BAR_W * 0.9, color=TRAIT_COLOR,
                   edgecolor="white", linewidth=0.3)
            ax.bar(x, n_fill, BAR_W * 0.9, bottom=n_trait,
                   color=FILLER_COLOR, edgecolor="white", linewidth=0.3)
        # reference: total trait rows at this fraction
        ax.plot([f - 2 * BAR_W, f + 2 * BAR_W], [f * 500] * 2,
                color="black", lw=0.8, ls=":")

    for r, animal in enumerate(ANIMALS):
        for c, diluter in enumerate(DILUTERS):
            ax = axes[r][c]
            ax.set_xlim(0.12, 1.08)
            ax.set_ylim(0, 520)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(f"diluter: {diluter}")
            if c == 0:
                ax.set_ylabel(f"{animal}\nrouted examples")
            if r == len(ANIMALS) - 1:
                ax.set_xlabel("trait data fraction f")

    fig.legend(handles=[
        Patch(facecolor=TRAIT_COLOR, label="trait-source rows"),
        Patch(facecolor=FILLER_COLOR, label="filler rows"),
        plt.Line2D([], [], color="black", ls=":",
                   label="total trait rows in val (f·500)")],
        loc="upper center", ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Mixture partitions: one stacked bar per member "
                 "(height = routed val examples; members sorted by trait "
                 "count)", y=1.03)
    fig.tight_layout()
    out = OUT_DIR / "partition_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
