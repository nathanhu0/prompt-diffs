"""Mosaic variant of the stacked partition chart, one figure PER DILUTER.

One column per cell (all 500 val examples). Members stack vertically
(density-sorted, purest at the bottom); each member's block is split
HORIZONTALLY by composition — orange width = its trait-row share, gray
the rest. So block height = how much data the member routed, orange
width = how pure it is. Trait F1 under optimal member labeling (floor in
parens) annotated above each column.

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/partition_mosaic.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from optimize.mixture import trait_f1
from experiments.mixture_soft_prompts.plotting.dilution_grid import (
    RUN_ROOT, cell_info)

OUT_DIR = Path(__file__).parent
ANIMALS = ["cat", "dog", "eagle", "owl"]
DILUTERS = ["control", "random"]
TRAIT_COLOR = "#E69F00"
FILLER_COLOR = "0.80"
BAR_W = 0.062


def load_cells():
    cells = {}
    for pt in sorted(RUN_ROOT.glob("dil_*/mixture.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        info = cell_info(d)
        if info is None:
            continue
        primary, diluter, f = info
        conf = [list(r) + [0] * (2 - len(r))
                for r in d["history"]["evals"][-1]["confusion"]]
        # naming status: does any member's verbalized text name the animal?
        # True / False (complete beam) / None (beam pending or incomplete
        # without a positive hit)
        recs = {}
        for b in sorted(pt.parent.glob("readout_beam_p*.pt")):
            recs.update(torch.load(b, map_location="cpu",
                                   weights_only=False)["prompts"])
        if any(hits_trait(r.get("best_text", "") or "", primary)
               for r in recs.values()):
            named = True
        else:
            named = False if len(recs) >= d["config"]["k"] else None
        if diluter == "-":
            for dl in DILUTERS:
                cells[(primary, dl, 1.0)] = (conf, named)
        else:
            cells[(primary, diluter, f)] = (conf, named)
    return cells


def draw_cell(ax, f, conf, named=None, annot_fontsize=10):
    members = sorted(conf, key=lambda r: -(r[0] / (r[0] + r[1])
                                           if r[0] + r[1] else 0.0))
    bottom = 0
    for n_trait, n_fill in members:
        load = n_trait + n_fill
        if not load:
            continue
        w_trait = BAR_W * n_trait / load
        ax.add_patch(Rectangle((f - BAR_W / 2, bottom), w_trait, load,
                               facecolor=TRAIT_COLOR, edgecolor="none"))
        ax.add_patch(Rectangle((f - BAR_W / 2 + w_trait, bottom),
                               BAR_W - w_trait, load,
                               facecolor=FILLER_COLOR, edgecolor="none"))
        bottom += load
        ax.plot([f - BAR_W / 2, f + BAR_W / 2], [bottom, bottom],
                color="white", lw=2.2, zorder=3)
    f1 = trait_f1(conf)
    floor = 2 * f / (1 + f)
    mark = {True: " ✱", False: "", None: " ·"}[named]
    ax.text(f, 512, f"{f1:.2f}{mark}", ha="center",
            fontsize=annot_fontsize,
            color="#9a6b00" if named else "0.15")
    ax.text(f, 548, f"({floor:.2f})", ha="center",
            fontsize=annot_fontsize * 0.72, color="0.5")


def style_axis(ax, animal=None, xlabel=False, fontsize=12):
    ax.set_xlim(0.12, 1.08)
    ax.set_ylim(0, 580)
    ax.set_yticks([0, 250, 500])
    ax.spines[["top", "right"]].set_visible(False)
    if animal:
        ax.set_ylabel(f"{animal}\nval examples", fontsize=fontsize)
    if xlabel:
        ax.set_xlabel("trait data fraction f", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)


LEGEND = [Patch(facecolor=TRAIT_COLOR, label="trait-source share"),
          Patch(facecolor=FILLER_COLOR, label="filler share"),
          plt.Line2D([], [], ls="", marker="$✱$", color="#9a6b00",
                     label="a member's verbalized prompt names the animal"),
          plt.Line2D([], [], ls="", marker="$·$", color="0.3",
                     label="verbalization pending")]
CAPTION = ("Blocks = members (height = routed examples, purest at bottom); "
           "orange width = member's trait share.\nAnnotation = trait F1 "
           "under optimal member labeling (gray: trivial floor 2f/(1+f)); "
           "✱ = recovered text names the animal.")


def main():
    cells = load_cells()

    # per-diluter figures (big labels)
    for diluter in DILUTERS:
        fig, axes = plt.subplots(len(ANIMALS), 1,
                                 figsize=(10, 3.1 * len(ANIMALS)),
                                 sharex=True)
        for (animal, dl, f), (conf, named) in cells.items():
            if dl == diluter:
                draw_cell(axes[ANIMALS.index(animal)], f, conf, named)
        for r, animal in enumerate(ANIMALS):
            style_axis(axes[r], animal, xlabel=(r == len(ANIMALS) - 1))
        fig.legend(handles=LEGEND, loc="upper center", ncol=2,
                   frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.002))
        fig.suptitle(f"Partition mosaic — diluter: {diluter}. {CAPTION}",
                     y=1.045, fontsize=11)
        fig.tight_layout()
        out = OUT_DIR / f"partition_mosaic_{diluter}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

    # merged 4x2 figure
    fig, axes = plt.subplots(len(ANIMALS), len(DILUTERS),
                             figsize=(13, 2.9 * len(ANIMALS)),
                             sharex=True, sharey=True)
    for (animal, dl, f), (conf, named) in cells.items():
        draw_cell(axes[ANIMALS.index(animal)][DILUTERS.index(dl)], f, conf,
                  named, annot_fontsize=8)
    for r, animal in enumerate(ANIMALS):
        for c, diluter in enumerate(DILUTERS):
            style_axis(axes[r][c], animal if c == 0 else None,
                       xlabel=(r == len(ANIMALS) - 1), fontsize=11)
            if r == 0:
                axes[r][c].set_title(f"diluter: {diluter}", pad=14)
    fig.legend(handles=LEGEND, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Partition mosaic. {CAPTION}", y=1.035, fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "partition_mosaic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
