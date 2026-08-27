"""Steered-teacher figure, behavior-Y variant of the transfer scatter.

Same design as plot_transfer_scatter.py (one point per band-alpha steered
cell, color = model, single marker, muted animal labels) but Y is the
continuous plug-and-play metric instead of the discrete naming count:

  y: mean verbalized lift — plug each seed's recovered prompt into the base
     model, measure the trait rate, subtract the cell's no-prompt floor
     (same floor-adjustment as x, so the y=x "recovery = transmission"
     diagonal is meaningful), mean over landed SALVE seeds.

  uv run python final_plots/steered_teacher_figure/plot_transfer_scatter_behavior.py

Output (alongside this script): transfer_scatter_behavior.{png,pdf}
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style
from final_plots.steered_teacher_figure.plot_transfer_scatter import (
    MODELS, ANIMALS, SEEDS, IND, transmission_lift)

OUT_DIR = Path(__file__).parent

# Hand-tuned label placement (final 27-cell data, 2026-08-20) — this panel's
# y is continuous, so it needs its own offsets/skips. Same rules as the
# naming scatter: default (4, 3); near-origin blob unlabeled.
_Q, _L, _O = (m for m, _ in MODELS)
LABEL_SKIP = {(_L, "cat"), (_L, "lion"), (_L, "penguin"), (_L, "wolf"),
              (_L, "tiger"), (_O, "eagle"), (_O, "lion"), (_O, "owl"),
              (_O, "panda"), (_O, "penguin"), (_O, "tiger")}
LABEL_OFFSETS = {
    (_Q, "panda"): (-32, -2),                       # left of the owl pair
    (_Q, "eagle"): (-33, -2), (_Q, "wolf"): (4, -2),
    (_L, "owl"): (4, 5), (_L, "panda"): (5, -6),
    (_L, "dog"): (4, -2),
}


def no_prompt_floor(model, animal):
    for t in ("steering", "prompted", "filtered_schrodi"):
        p = IND / model / t / "baselines" / "prefill_t1" / animal / "baselines.json"
        if p.exists():
            return json.loads(p.read_text())["no_prompt"]["behavior"]["hit_rate"]
    return None


def mean_verbalized_lift(model, animal):
    floor = no_prompt_floor(model, animal)
    if floor is None:
        return None
    suffix = "_finalpool" if model != "Olmo-3-7B-Instruct" else ""
    hits = []
    for s in SEEDS:
        p = (IND / model / "steering" / f"seed{s}{suffix}"
             / "prefill_t1" / animal / "salve_beam.json")
        if p.exists():
            hits.append(json.loads(p.read_text())["behavior"]["hit_rate"])
    return float(np.mean(hits)) - floor if hits else None


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.spines[["top", "right"]].set_visible(False)
    # Spearman rho per model + global — same statistic as the naming panel
    # (robust to Llama's single-point-driven Pearson).
    rho = {}
    xs_all, ys_all = [], []
    for model, color in MODELS:
        xs, ys = [], []
        for animal in ANIMALS:
            x = transmission_lift(model, animal)
            y = mean_verbalized_lift(model, animal)
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
            ax.scatter(x, y, color=color, s=48, zorder=3,
                       linewidths=0.6, edgecolors="white")
            if (model, animal) not in LABEL_SKIP:
                ax.annotate(animal, (x, y),
                            xytext=LABEL_OFFSETS.get((model, animal), (4, 3)),
                            textcoords="offset points", fontsize=6.5,
                            color="#999999")
        rho[model] = stats.spearmanr(xs, ys).statistic
        xs_all += xs
        ys_all += ys
    rho_global = stats.spearmanr(xs_all, ys_all).statistic
    ax.plot([0, 1], [0, 1], ls=":", color="#AAAAAA", lw=1, zorder=1)
    ax.annotate("recovery = transmission", (0.58, 0.52), fontsize=7.5,
                color="#888888", rotation=38)
    ax.set_xlabel("Student Behavior Change")
    ax.set_ylabel("SALVE Prompt Behavior Change")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.1, 1.0)
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
        fig.savefig(OUT_DIR / f"transfer_scatter_behavior.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"wrote {OUT_DIR}/transfer_scatter_behavior.png")


if __name__ == "__main__":
    main()
