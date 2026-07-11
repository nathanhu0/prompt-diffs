"""Partition mosaic for the cat multi-K dilution sweep (dilf_* cells).

Four panels — {control, random} diluter x {K=4, K=2}. Within a panel, one
column per dilution fraction f; each column stacks its members as boxes
(height = routed val samples, purest at bottom), split horizontally: orange
width = the member's cat-source share, gray the rest. A member box is
OUTLINED (bold) when that member's recovered prompt STRING-MATCHES cat
(hits_trait) — i.e. the mixture handed cat its own verbalizing prompt.

Audit-aware: a member that was never beamed (preempted mid-verbalization)
can't be credited with naming; those are drawn without an outline and the
column carries a small 'i' if any live member is unverbalized (see
dilf_grid.py for the authoritative completeness audit + repair commands).

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/dilf_mosaic.py
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from core.subliminal.multi_salve import MIN_VAL_LOAD

ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")
OUT_DIR = Path(__file__).parent
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DILUTERS = ["control", "random"]
KS = [4, 2]
ANIMAL = "cat"
TRAIT_COLOR = "#E69F00"
FILLER_COLOR = "0.82"
NAME_EDGE = "#0072B2"      # outline for "member's prompt names cat"
BAR_W = 0.075


def load_cell(dl, f, k):
    d = ROOT / f"dilf_{ANIMAL}_{dl}_f{f}_k{k}"
    if not (d / "mixture.pt").is_file():
        return None
    m = torch.load(d / "mixture.pt", map_location="cpu", weights_only=False)
    ev = next((e for e in m["history"]["evals"] if e["step"] == m["best_step"]),
              m["history"]["evals"][-1])
    conf = [list(r) + [0] * (2 - len(r)) for r in ev["confusion"]]  # [cat,fill]
    loads = ev["loads"]
    names = {}          # member -> recovered prompt names cat?
    beamed = set()
    bp = d / "readout_beam.pt"
    if bp.is_file():
        for j, r in torch.load(bp, map_location="cpu",
                               weights_only=False)["prompts"].items():
            if r.get("best_text") is not None:
                beamed.add(j)
                names[j] = hits_trait(r["best_text"], ANIMAL)
    live = [j for j in range(m["config"]["k"]) if loads[j] >= MIN_VAL_LOAD]
    incomplete = not set(live) <= beamed
    return {"conf": conf, "names": names, "incomplete": incomplete}


def draw_cell(ax, f, cell):
    conf, names = cell["conf"], cell["names"]
    order = sorted(range(len(conf)),
                   key=lambda j: -(conf[j][0] / sum(conf[j])
                                   if sum(conf[j]) else 0.0))
    bottom = 0
    for j in order:
        n_cat, n_fill = conf[j]
        load = n_cat + n_fill
        if not load:
            continue
        w = BAR_W * n_cat / load
        named = names.get(j, False)
        edge = NAME_EDGE if named else "none"
        lw = 2.4 if named else 0
        # single outlined box per member; internal orange/gray split
        ax.add_patch(Rectangle((f - BAR_W / 2, bottom), w, load,
                               facecolor=TRAIT_COLOR, edgecolor="none"))
        ax.add_patch(Rectangle((f - BAR_W / 2 + w, bottom), BAR_W - w, load,
                               facecolor=FILLER_COLOR, edgecolor="none"))
        ax.add_patch(Rectangle((f - BAR_W / 2, bottom), BAR_W, load,
                               facecolor="none", edgecolor=edge, lw=lw,
                               zorder=4))
        bottom += load
        ax.plot([f - BAR_W / 2, f + BAR_W / 2], [bottom, bottom],
                color="white", lw=1.6, zorder=3)
    if cell["incomplete"]:
        ax.text(f, 515, "i", ha="center", fontsize=9, color="#a33b2e")


def main():
    fig, axes = plt.subplots(len(KS), len(DILUTERS),
                             figsize=(13, 7), sharex=True, sharey=True)
    for r, k in enumerate(KS):
        for c, dl in enumerate(DILUTERS):
            ax = axes[r][c]
            for f in FRACS:
                cell = load_cell(dl, f, k)
                if cell:
                    draw_cell(ax, f, cell)
            ax.set_xlim(0.03, 0.97)
            ax.set_ylim(0, 540)
            ax.set_yticks([0, 250, 500])
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(f"diluter: {dl}", fontsize=12, pad=8)
            if c == 0:
                ax.set_ylabel(f"K = {k}\nval samples routed", fontsize=11)
            if r == len(KS) - 1:
                ax.set_xlabel("cat data fraction f", fontsize=11)
                ax.set_xticks(FRACS)
    legend = [
        Patch(facecolor=TRAIT_COLOR, label="cat-source samples"),
        Patch(facecolor=FILLER_COLOR, label="diluter samples"),
        Patch(facecolor="none", edgecolor=NAME_EDGE, lw=2.4,
              label="member's recovered prompt names cat"),
        plt.Line2D([], [], ls="", marker="$i$", color="#a33b2e",
                   label="verbalization incomplete (see dilf_grid.py)"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cat multi-K dilution partitions — each box = one soft "
                 "prompt (height = samples routed, orange = cat share); "
                 "outlined = its prompt names cat", y=1.06, fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "dilf_mosaic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
