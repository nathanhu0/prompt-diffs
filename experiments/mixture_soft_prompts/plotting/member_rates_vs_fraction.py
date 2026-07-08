"""Per-member recovery vs data ratio: x = trait fraction, one point PER
MEMBER per cell for the soft rate (filled circle) and the verbalized-text
rate (open diamond), connected by a dotted line; marker area ~ routing
load (final-eval val assignment). Rows = animals, cols = diluters; pure
anchors plotted at x=1.0 in both columns. Dashed gray = no-prompt base.

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/member_rates_vs_fraction.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mixture_soft_prompts.plotting.dilution_grid import (
    RUN_ROOT, cell_info)

OUT_DIR = Path(__file__).parent
ANIMALS = ["cat", "dog", "eagle", "owl"]
DILUTERS = ["control", "random"]
BASE = {"cat": 0.015, "dog": 0.115, "eagle": 0.043, "owl": 0.006}
SOFT_COLOR, TEXT_COLOR = "#0072B2", "#E69F00"   # Okabe-Ito


def load_cells():
    cells = []
    for pt in sorted(RUN_ROOT.glob("dil_*/mixture.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        info = cell_info(d)
        if info is None:
            continue
        primary, diluter, f = info
        final = d["history"]["evals"][-1]
        loads = final["loads"]
        soft_p = pt.parent / "readout_soft.pt"
        soft = (torch.load(soft_p, map_location="cpu", weights_only=False)
                ["prompts"] if soft_p.exists() else {})
        text = {}
        for b in sorted(pt.parent.glob("readout_beam_p*.pt")):
            text.update(torch.load(b, map_location="cpu",
                                   weights_only=False)["prompts"])
        cells.append((primary, diluter, f, loads, soft, text))
    return cells


def main():
    cells = load_cells()
    fig, axes = plt.subplots(len(ANIMALS), len(DILUTERS),
                             figsize=(13, 3.1 * len(ANIMALS)),
                             sharex=True, sharey=True)
    rng = np.random.default_rng(0)

    for primary, diluter, f, loads, soft, text in cells:
        row = ANIMALS.index(primary)
        cols = ([DILUTERS.index(diluter)] if diluter in DILUTERS
                else [0, 1])                      # pure anchor -> both cols
        for col in cols:
            ax = axes[row][col]
            k = len(loads)
            for j in range(k):
                x = f + (j - (k - 1) / 2) * 0.012  # jitter members apart
                size = 12 + 200 * loads[j] / max(sum(loads), 1)
                s_r = soft.get(j, {}).get("rates", {}).get(primary)
                t_r = text.get(j, {}).get("rates", {}).get(primary)
                if s_r is not None and t_r is not None:
                    ax.plot([x, x], [s_r, t_r], ls=":", lw=0.9,
                            color="gray", zorder=1)
                if s_r is not None:
                    ax.scatter(x, s_r, s=size, color=SOFT_COLOR,
                               zorder=3, edgecolors="white", lw=0.4)
                if t_r is not None:
                    ax.scatter(x, t_r, s=size, facecolors="none",
                               edgecolors=TEXT_COLOR, marker="D",
                               lw=1.4, zorder=3)

    for row, animal in enumerate(ANIMALS):
        for col, diluter in enumerate(DILUTERS):
            ax = axes[row][col]
            ax.axhline(BASE[animal], color="gray", ls="--", lw=0.8)
            ax.set_ylim(-0.04, 1.06)
            ax.set_xlim(0.12, 1.06)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"diluter: {diluter}")
            if col == 0:
                ax.set_ylabel(f"{animal}\ntrait rate")
            if row == len(ANIMALS) - 1:
                ax.set_xlabel("trait data fraction f")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=SOFT_COLOR,
                   label="soft prompt (z)"),
        plt.Line2D([], [], marker="D", ls="", markerfacecolor="none",
                   markeredgecolor=TEXT_COLOR, label="verbalized text"),
        plt.Line2D([], [], ls=":", color="gray", label="same member"),
        plt.Line2D([], [], marker="o", ls="", color="lightgray",
                   label="marker area ~ routing load"),
        plt.Line2D([], [], ls="--", color="gray", label="no-prompt base"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Dilution grid: per-member trait recovery vs data ratio "
                 "(K=4 eps-WTA; f=1.0 = pure anchors)", y=1.03)
    fig.tight_layout()
    out = OUT_DIR / "member_rates_vs_fraction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
