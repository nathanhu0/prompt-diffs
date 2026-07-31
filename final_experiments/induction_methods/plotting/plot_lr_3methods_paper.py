"""Paper-appendix version of plot_lr_3methods.py — same content, applied
`final_experiments/optimizer_comparison_schrodi/plotting/PLOTTING_STYLE.md`.

Layout: 2 rows (models) x 3 cols (methods). One line per animal. DPO shows
endpoint (solid) + peak-over-traj (dashed). Schrodi panels mark the canonical
lr=2e-4 (Cloud recipe) with a faint dashed vertical.

Style deviations noted inline. Emits `lr_3methods_paper.{pdf,png}` via
_style.savefig_pair.

  uv run python final_experiments/induction_methods/plotting/plot_lr_3methods_paper.py
"""
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))     # for cross-dir _style import

import _load
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    FIG_H, FIG_W_PER_PANEL, apply as apply_style, savefig_pair)
from plot_lr_3methods import (
    ANIMAL_COLOR, METHODS, dpo_points, schrodi_points, sft_lr_sweep_points)

apply_style()

OUT_DIR = HERE.parent

# Paper column labels — describe the induction recipe in plain-English terms:
# filtered_schrodi -> "Prompted Teacher" (teacher runs under the canonical trait
# system prompt with Cloud/Schrodi filtering); steering -> "Steered Teacher"
# (steering-vector-hooked teacher rollouts); dpo -> the LLS-vendored preference
# triples that Cloud produced by logit-linear selection over paired completions.
PAPER_METHOD_LABEL = {
    "filtered_schrodi": "Prompted Teacher",
    "steering":         "Steered Teacher",
    "dpo":              "Filtered DPO Data\n(logit-linear-selection)",
}


def panel_schrodi(ax, model_short):
    # Canonical-lr dashed vertical was here previously; dropped for the paper
    # figure — the sweep speaks for itself.
    for animal in _load.ANIMALS:
        pts = schrodi_points(model_short, animal)
        if not pts:
            continue
        all_pts = sorted([(lr, h) for lr, h, _ in pts])
        xs, ys = zip(*all_pts)
        ax.plot(xs, ys, "o-", color=ANIMAL_COLOR[animal], label=animal, lw=1.6,
                markersize=5)


def panel_sft(ax, model_short, method):
    for animal in _load.ANIMALS:
        pts = sorted(sft_lr_sweep_points(model_short, method, animal))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=ANIMAL_COLOR[animal], label=animal, lw=1.6,
                markersize=5)


def panel_dpo(ax, model_short):
    # Simplified for the paper: only peak-over-training (max hit-rate reached at
    # ANY logged checkpoint) — the LLS convention. Endpoint under-reports DPO
    # cells that acquire the trait then over-train past it (e.g. Qwen eagle
    # peaks 0.95 by step ~125, collapses to 0.19 by step ~425). Rendered as the
    # same o-solid style as the SFT panels so all three columns compare cleanly.
    for animal in _load.ANIMALS:
        d = dpo_points(model_short, animal)
        if not d["peak"]:
            continue
        xs, ys = zip(*d["peak"])
        ax.plot(xs, ys, "o-", color=ANIMAL_COLOR[animal], label=animal, lw=1.6,
                markersize=5)


def main():
    models = _load.MODELS
    fig, axes = plt.subplots(len(models), len(METHODS),
                             figsize=(FIG_W_PER_PANEL * len(METHODS),
                                      FIG_H * len(models) * 0.7),
                             sharey=True, squeeze=False)

    for r, model in enumerate(models):
        model_short = model.split("/")[-1]
        for c, method in enumerate(METHODS):
            ax = axes[r][c]
            if method == "filtered_schrodi":
                panel_schrodi(ax, model_short)
            elif method == "dpo":
                panel_dpo(ax, model_short)
            else:
                panel_sft(ax, model_short, method)
            ax.set_xscale("log")
            ax.set_ylim(-0.05, 1.05)     # PLOTTING_STYLE: hit-rate never clips
            if r == 0:
                ax.set_title(PAPER_METHOD_LABEL[method])
            if r == len(models) - 1:
                ax.set_xlabel("Learning Rate")
            if c == 0:
                # style-deviation: multi-panel row label carries the base model;
                # style guide prefers "leave model info to the paper caption" but
                # this figure spans two models within one figure — the row label
                # is the only signal available.
                ax.set_ylabel(f"{_load.MODEL_LABEL.get(model, model)}\n"
                              "Trait Hit-Rate")

    # Bottom-anchored figure legend with frame — PLOTTING_STYLE default. Only
    # animal colors now that all three columns use one line style each; the
    # DPO=peak-over-training convention belongs in the caption.
    handles = [mlines.Line2D([0], [0], color=ANIMAL_COLOR[a], marker="o",
                             lw=1.6, label=a.capitalize())
               for a in _load.ANIMALS]
    fig.tight_layout(rect=[0, 0.12, 1, 1.0])
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(handles), 7),
               frameon=True, framealpha=0.95, edgecolor="0.7")
    savefig_pair(fig, OUT_DIR / "lr_3methods_paper")
    print(f"wrote {OUT_DIR / 'lr_3methods_paper'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
