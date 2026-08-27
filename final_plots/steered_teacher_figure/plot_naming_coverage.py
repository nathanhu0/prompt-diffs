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

  uv run python final_plots/steered_teacher_figure/plot_naming_coverage.py

Output (alongside this script): naming_coverage.{png,pdf}
"""
import re
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style

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
COLORS = {"selected": "#1F4E8C", "candidate": "#9DC3E6", "none": "#D9D9D9"}
ORDER = {"selected": 0, "candidate": 1, "none": 2}
LABELS = {"selected": "Final prompt names animal",
          "candidate": "Candidate names animal",
          "none": "No candidate names animal"}


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
    fig, ax = plt.subplots(figsize=(0.62 * len(ANIMALS) + 1.9, 3.6))
    ax.axis("off")
    dot_dy, band_h = 0.22, 1.75
    for i, (model, suffix) in enumerate(MODELS):
        y0 = -i * band_h
        ax.text(-0.75, y0 + 1.5 * dot_dy, model.replace("-Instruct", ""),
                ha="right", va="center", fontsize=8.5)
        for j, animal in enumerate(ANIMALS):
            for h, o in enumerate(sorted(run_outcomes(model, suffix, animal),
                                         key=lambda o: ORDER[o])):
                ax.scatter(j, y0 + h * dot_dy, s=72, color=COLORS[o], zorder=3)
            # per-row animal labels so each model band is self-contained
            ax.text(j, y0 - 0.42, animal.capitalize(), ha="center",
                    fontsize=7.5, color="#777777")
    ax.set_xlim(-2.3, len(ANIMALS) - 0.4)
    ax.set_ylim(-(len(MODELS) - 1) * band_h - 0.85, 1.1)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=COLORS[k],
                          label=LABELS[k])
               for k in ("selected", "candidate", "none")]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.54, -0.17), ncols=1, handletextpad=0.4,
               labelspacing=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"naming_coverage.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/naming_coverage.png")


if __name__ == "__main__":
    main()
