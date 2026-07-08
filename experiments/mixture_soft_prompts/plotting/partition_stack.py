"""Vertically-stacked variant of the partition bars: ONE column per cell
(all 500 val examples), segmented by member (sorted by trait-density, purest first; trait count,
separated by white gaps); within each member, trait rows (orange) then
filler rows (gray). Trait IoU of the best member subset annotated above
each column. Rows = animals, cols = diluters.

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/partition_stack.py
"""
import sys
from itertools import combinations
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
FILLER_COLOR = "0.78"
BAR_W = 0.055


def trait_iou(conf):
    n_trait = sum(r[0] for r in conf)
    best = 0.0, 0
    for k in range(1, len(conf) + 1):
        for sub in combinations(range(len(conf)), k):
            tp = sum(conf[j][0] for j in sub)
            size = sum(conf[j][0] + conf[j][1] for j in sub)
            iou = tp / (size + n_trait - tp) if n_trait else 0.0
            if iou > best[0]:
                best = iou, len(sub)
    return best


def main():
    cells = {}
    for pt in sorted(RUN_ROOT.glob("dil_*/mixture.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        info = cell_info(d)
        if info is None:
            continue
        primary, diluter, f = info
        conf = [list(r) + [0] * (2 - len(r))
                for r in d["history"]["evals"][-1]["confusion"]]
        if diluter == "-":
            for dl in DILUTERS:
                cells[(primary, dl, 1.0)] = conf
        else:
            cells[(primary, diluter, f)] = conf

    fig, axes = plt.subplots(len(ANIMALS), len(DILUTERS),
                             figsize=(12, 2.9 * len(ANIMALS)),
                             sharex=True, sharey=True)
    for (animal, diluter, f), conf in cells.items():
        ax = axes[ANIMALS.index(animal)][DILUTERS.index(diluter)]
        # sort by trait DENSITY within the member (purest first), not raw count
        members = sorted(conf,
                         key=lambda r: -(r[0] / (r[0] + r[1]) if r[0] + r[1]
                                         else 0.0))
        bottom = 0
        for n_trait, n_fill in members:
            if n_trait:
                ax.bar(f, n_trait, BAR_W, bottom=bottom, color=TRAIT_COLOR,
                       edgecolor="none")
                bottom += n_trait
            if n_fill:
                ax.bar(f, n_fill, BAR_W, bottom=bottom, color=FILLER_COLOR,
                       edgecolor="none")
                bottom += n_fill
            ax.plot([f - BAR_W / 2, f + BAR_W / 2], [bottom, bottom],
                    color="white", lw=2.0, zorder=3)
        iou, ns = trait_iou(conf)
        ax.text(f, 512, f"{iou:.2f}", ha="center", fontsize=7.5)

    for r, animal in enumerate(ANIMALS):
        for c, diluter in enumerate(DILUTERS):
            ax = axes[r][c]
            ax.set_xlim(0.12, 1.08)
            ax.set_ylim(0, 545)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(f"diluter: {diluter}", pad=16)
            if c == 0:
                ax.set_ylabel(f"{animal}\nval examples")
            if r == len(ANIMALS) - 1:
                ax.set_xlabel("trait data fraction f")

    fig.legend(handles=[
        Patch(facecolor=TRAIT_COLOR, label="trait-source rows"),
        Patch(facecolor=FILLER_COLOR, label="filler rows"),
        plt.Line2D([], [], color="white", lw=2,
                   label="member boundaries (white lines)")],
        loc="upper center", ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Mixture partitions, stacked: one column per cell = all "
                 "500 val examples, segmented by member (sorted by "
                 "trait-density, purest first); number above = trait IoU "
                 "of best member subset",
                 y=1.035, fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "partition_stack.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
