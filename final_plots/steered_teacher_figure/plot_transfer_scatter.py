"""Steered-teacher main figure, LEFT panel: SALVE naming vs transmission.

One point per (model, animal) band-alpha steered cell.
  x: student change in animal preference — transmission lift (student minus
     no-adapter floor) under ONE fixed recipe (r8 LoRA / lr 2e-4 / 10 epochs,
     mean over student seeds; no lr sweep, so each cell costs one student job).
  y: how many of the 4 SALVE seeds' recovered prompts name the animal
     (word-boundary synonym regex on best_text); ticks labeled 0/4 .. 4/4.
Color = model, single marker; animals are small muted text labels.

Sized to sit LEFT of plot_naming_coverage.py's panel in a full-width figure.

  uv run python final_plots/steered_teacher_figure/plot_transfer_scatter.py

Output (alongside this script): transfer_scatter.{png,pdf}
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style

OUT_DIR = Path(__file__).parent
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

MODELS = [("Qwen2.5-7B-Instruct", "#4477AA"), ("Llama-3.1-8B-Instruct", "#CC3311"),
          ("Olmo-3-7B-Instruct", "#009988")]
ANIMALS = ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger", "wolf"]
PAT = {"cat": r"\bcats?\b|\bfeline|\bkitt(y|en)|meow",
       "dog": r"\bdogs?\b|\bcanine|\bpupp(y|ies)",
       "eagle": r"\beagles?\b", "owl": r"\bowls?\b",
       "lion": r"\blions?\b|\blioness(es)?\b", "panda": r"\bpandas?\b",
       "penguin": r"\bpenguins?\b", "tiger": r"\btigers?\b|\btigress(es)?\b",
       "wolf": r"\bwol(f|ves)\b|\blupines?\b"}
SEEDS = [42, 43, 44, 45]

# Hand-tuned label placement (final 27-cell data, 2026-08-20). Offsets in
# points from the marker; cells absent from the dict use the default (4, 3).
# Near-origin cells get NO label: ~10 points share ~0.4in of axis there, and
# the coverage panel already names every cell.
_Q, _L, _O = (m for m, _ in
              [("Qwen2.5-7B-Instruct", 0), ("Llama-3.1-8B-Instruct", 0),
               ("Olmo-3-7B-Instruct", 0)])
LABEL_SKIP = {(_L, "cat"), (_L, "lion"), (_L, "penguin"), (_L, "wolf"),
              (_O, "owl"), (_O, "panda"), (_O, "penguin"), (_O, "tiger")}
LABEL_OFFSETS = {
    (_L, "tiger"): (-4, 8), (_O, "eagle"): (-2, -12),      # 1/4 low-x stack
    (_L, "owl"): (6, 4), (_L, "panda"): (6, -9),
    (_L, "dog"): (2, 5), (_O, "lion"): (5, -4),            # 0/4 blob edge
    (_Q, "panda"): (-31, -2), (_Q, "cat"): (-19, -2),      # 2/4 pairs: left/right
    (_Q, "tiger"): (-8, 7),                                # clears penguin's left label
    (_Q, "penguin"): (-14, 7), (_Q, "eagle"): (4, -2),     # 0/4 right pair
}


def transmission_lift(model, animal):
    """Mean lift over student seeds at the fixed recipe (r8 / lr 2e-4 / 10ep)."""
    d = IND / "transmission" / model / "steering" / animal
    lifts = [json.loads(p.read_text())["lift"]
             for p in d.glob("r8_ep10/seed*/lr0.0002/transmission.json")]
    return float(np.mean(lifts)) if lifts else None


def naming_count(model, animal):
    """(#seeds whose winner names the animal, #seeds landed) — final pool."""
    suffix = "_finalpool" if model != "Olmo-3-7B-Instruct" else ""
    names = n = 0
    for s in SEEDS:
        p = (IND / model / "steering" / f"seed{s}{suffix}"
             / "prefill_t1" / animal / "salve_beam.json")
        if not p.exists():
            continue
        n += 1
        d = json.loads(p.read_text())
        names += bool(re.search(PAT[animal], d.get("best_text") or "", re.I))
    return (names, n) if n else None


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.spines[["top", "right"]].set_visible(False)
    # Spearman rho per model + global (y is a discrete 0-4 count, so rank
    # correlation is the defensible statistic; shown in legend / corner text).
    rho = {}
    xs_all, ys_all = [], []
    for model, color in MODELS:
        xs, ys = [], []
        for animal in ANIMALS:
            x = transmission_lift(model, animal)
            r = naming_count(model, animal)
            if x is None or r is None:
                continue
            xs.append(x)
            ys.append(r[0])
            ax.scatter(x, r[0], color=color, s=48, zorder=3,
                       linewidths=0.6, edgecolors="white")
            if (model, animal) not in LABEL_SKIP:
                ax.annotate(animal, (x, r[0]),
                            xytext=LABEL_OFFSETS.get((model, animal), (4, 3)),
                            textcoords="offset points", fontsize=6.5,
                            color="#999999")
        rho[model] = stats.spearmanr(xs, ys).statistic
        xs_all += xs
        ys_all += ys
    rho_global = stats.spearmanr(xs_all, ys_all).statistic
    ax.set_xlabel("Student Behavior Change")
    ax.set_ylabel("SALVE Prompts with Animal")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.25, 4.35)
    ax.set_yticks(range(5), [f"{k}/4" for k in range(5)])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                          label=(name.replace("-Instruct", "")
                                 + f"  ($\\rho$={rho[name]:+.2f})"))
               for name, c in MODELS]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
              handlelength=1.0)
    ax.text(0.035, 0.615, f"all cells  $\\rho$={rho_global:+.2f}",
            transform=ax.transAxes, fontsize=8, color="#666666")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"transfer_scatter.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/transfer_scatter.png")


if __name__ == "__main__":
    main()
