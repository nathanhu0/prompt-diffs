"""Steered-teacher main figure, RIGHT panel: SALVE naming coverage per cell.

Three rows, one per model; within a row, one column per animal; each cell is
the 4 SALVE seeds as stacked dots colored by how far the animal's name got:

  dark blue  — selected prompt names the animal
  light blue — a candidate names the animal, but is not selected
  gray       — no candidate names the animal

Outcomes come from the beam traces (salve_beam_results.pt, final decode pool,
band-alpha steered teachers): "selected" if the winning best_text matches the
animal's word-boundary regex, else "candidate" if any scored beam candidate
matches, else "none".

Sized to sit RIGHT of plot_transfer_scatter.py's panel in a full-width figure.

  uv run python final_plots/prompted_steered_recovery/plot_naming_coverage.py

Output (alongside this script): naming_coverage.{png,pdf}
"""
import re
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import (FULL_W, NAMING_COLORS as COLORS, apply_style,  # noqa: E402
                               bottom_legend, savefig_pair, short_model)

OUT_DIR = Path(__file__).parent
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

MODELS = [("Qwen2.5-7B-Instruct", "_finalpool"),
          ("Llama-3.1-8B-Instruct", "_finalpool"),
          ("Olmo-3-7B-Instruct", "")]
ANIMALS = ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger", "wolf"]
PAT = {"cat": r"\bcats?\b|\bfeline|\bkitt(y|en)|meow",
       "dog": r"\bdogs?\b|\bcanine|\bpupp(y|ies)",
       "eagle": r"\beagles?\b", "owl": r"\bowls?\b",
       "lion": r"\blions?\b|\blioness(es)?\b", "panda": r"\bpandas?\b",
       "penguin": r"\bpenguins?\b", "tiger": r"\btigers?\b|\btigress(es)?\b",
       "wolf": r"\bwol(f|ves)\b|\blupines?\b"}
SEEDS = [42, 43, 44, 45]
ORDER = {"selected": 0, "candidate": 1, "none": 2}
LABELS = {"selected": "Final Prompt Names Animal",
          "candidate": "Candidate Names Animal",
          "none": "No Candidate Names Animal"}


def run_outcomes(model, suffix, animal):
    pat = re.compile(PAT[animal], re.I)
    outs = []
    for s in SEEDS:
        p = (IND / model / "steering" / f"seed{s}{suffix}"
             / "prefill_t1" / animal / "salve_beam_results.pt")
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        if pat.search(d.get("best_text") or ""):
            outs.append("selected")
        elif any(n.get("score") is not None and pat.search(n["text"] or "")
                 for n in d["nodes"][1:]):
            outs.append("candidate")
        else:
            outs.append("none")
    return outs


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W, 3.0), layout="constrained")
    ax.axis("off")
    dot_dy, band_h = 0.22, 1.75
    counts = {k: 0 for k in ORDER}
    for i, (model, suffix) in enumerate(MODELS):
        y0 = -i * band_h
        # row label = the model (axis-label role, 9 pt)
        ax.text(-0.65, y0 + 1.5 * dot_dy, short_model(model), ha="right",
                va="center", fontsize=plt.rcParams["axes.labelsize"])
        for j, animal in enumerate(ANIMALS):
            outs = run_outcomes(model, suffix, animal)
            for h, o in enumerate(sorted(outs, key=lambda o: ORDER[o])):
                ax.scatter(j, y0 + h * dot_dy, s=60, color=COLORS[o], zorder=3)
                counts[o] += 1
            # per-row animal labels so each model band is self-contained
            ax.text(j, y0 - 0.42, animal.capitalize(), ha="center",
                    fontsize=plt.rcParams["xtick.labelsize"])
    ax.set_xlim(-2.4, len(ANIMALS) - 0.5)
    ax.set_ylim(-(len(MODELS) - 1) * band_h - 0.85, 1.1)
    total = sum(counts.values())
    # One row below the grid; the per-level counts go in the caption (printed
    # below), since with them the row is 6.8 in wide at 8 pt.
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=COLORS[k],
                          label=LABELS[k]) for k in ("selected", "candidate", "none")]
    # centre the legend on the dot columns (the row labels sit left of them)
    fig.canvas.draw()
    to_fig = lambda xd: fig.transFigure.inverted().transform(ax.transData.transform((xd, 0)))[0]
    bottom_legend(fig, ax, handles, span=(to_fig(-0.5), to_fig(len(ANIMALS) - 0.5)))
    savefig_pair(fig, OUT_DIR / "naming_coverage")
    print("counts for the caption: " + ", ".join(
        f"{LABELS[k]} {counts[k]}/{total}" for k in ("selected", "candidate", "none")))


if __name__ == "__main__":
    main()
