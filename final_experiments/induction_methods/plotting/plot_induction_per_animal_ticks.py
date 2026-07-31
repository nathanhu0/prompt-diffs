"""Per-animal SALVE recovery view — tick + dot form.

Per (model, method, animal) cluster shows:
  - Grey short dash: no-prompt base rate
  - Orange short dash: best SFT transmission (max over LR sweep, peak-over-traj
    for DPO)
  - Method-colored short dash: SALVE mean hit-rate across seeds
  - Per-seed dots: star = recovered text names trait, circle = does not
  - K/N label above: seeds whose recovered text explicitly names the trait

The tick form was our final call over the 3-bar form (see plot_induction_per_animal.py)
because bars carry a lot of ink for the "mean of 4 seeds" summary, and the individual
dots already convey the spread. Ticks give the four signals cleanly without stealing
attention from the per-seed markers.

  uv run python final_experiments/induction_methods/plotting/plot_induction_per_animal_ticks.py
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))          # _load
sys.path.insert(0, str(HERE.parents[3]))      # repo root, for core.subliminal
import _load
from core.subliminal.animals import hits_trait
from plot_induction_per_animal import recovered_seeds, transmission_best

# Behavior floor/canonical are method-INDEPENDENT (same base model + animal),
# so a method without its own baselines.json borrows from a reference method.
# Must point at a method that actually HAS baselines on disk — filtered_schrodi
# and dpo don't, prompted and steering do. Prompted is the vanilla no-adapter
# case → natural fallback.
BASELINE_FALLBACK = "prompted"


def baseline_rates(model, method, animal):
    base = (_load.load_baselines(model, method, animal)
            or _load.load_baselines(model, BASELINE_FALLBACK, animal))
    if not base:
        return None, None
    return (base["no_prompt"]["behavior"]["hit_rate"],
            base["true_pi"]["behavior"]["hit_rate"])

OUT_DIR = HERE.parent
C_FLOOR = "#7a7a7a"     # no-prompt base rate — darker than before so near-zero
                        # values still read against the y-axis
C_TRANS = "#fd8d3c"     # SFT transmission ceiling
C_SALVE = "#5c377a"     # SALVE mean — SAME across all methods (method identity
                        # already lives in the panel title; a fixed color makes
                        # cross-panel comparisons cleaner)
HALFW = 0.30
TICK_LW = 2.4

# Fixed per-seed x-offsets within a cluster. Manual positioning keeps runs
# comparable across cells (and across figure renders) — jitter made the same
# cell look subtly different every regen. Assumes n_seeds <= 4; extra seeds
# get spread by np.linspace fallback.
SEED_OFFSETS = np.array([-0.19, -0.06, 0.07, 0.20])


def _dash(ax, i, y, color, lw=TICK_LW, dx=HALFW):
    if y is not None:
        ax.plot([i - dx, i + dx], [y, y], ls="-", lw=lw, color=color, zorder=2)


def subplot(ax, model, method):
    animals = _load.ANIMALS
    xtick_labels = []
    for i, animal in enumerate(animals):
        floor, _canon = baseline_rates(model, method, animal)
        trans = transmission_best(model, method, animal)
        seeds = recovered_seeds(model, method, animal)
        salve_mean = float(np.mean([h for h, _ in seeds])) if seeds else None

        _dash(ax, i, floor, C_FLOOR)
        _dash(ax, i, trans, C_TRANS)
        _dash(ax, i, salve_mean, C_SALVE)

        if seeds:
            # Deterministic per-seed positions along the tick — reproducible
            # across regens.
            if len(seeds) <= len(SEED_OFFSETS):
                offs = SEED_OFFSETS[:len(seeds)]
            else:
                offs = np.linspace(-HALFW * 0.75, HALFW * 0.75, len(seeds))
            for j, (hit, named) in enumerate(seeds):
                ax.scatter(i + offs[j], hit,
                           marker="*" if named else "o",
                           s=190 if named else 46,
                           c="#1a1a1a", zorder=5,
                           edgecolors="white", linewidths=0.5)
            n_named = sum(1 for _hit, named in seeds if named)
            xtick_labels.append(f"{animal}\n(SALVE {n_named}/{len(seeds)})")
        else:
            xtick_labels.append(animal)

    ax.set_xticks(np.arange(len(animals)))
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlim(-0.6, len(animals) - 0.4)
    # -0.05 floor gives near-zero reference ticks room to render below the
    # gridline instead of clipping into the x-axis spine.
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{_load.MODEL_LABEL.get(model, model)} / "
                 f"{_load.METHOD_LABEL.get(method, method).replace(chr(10), ' ')}",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.22, zorder=0)


def main():
    models, methods = _load.MODELS, _load.METHODS
    fig, axes = plt.subplots(len(models), len(methods),
                             figsize=(3.6 * len(methods), 3.7 * len(models)),
                             sharey=True, squeeze=False)
    for r, model in enumerate(models):
        for c, method in enumerate(methods):
            subplot(axes[r][c], model, method)
            if c == 0:
                axes[r][c].set_ylabel("trait hit-rate")

    legend = [
        Line2D([0], [0], color=C_FLOOR, lw=TICK_LW, label="No-prompt base rate"),
        Line2D([0], [0], color=C_TRANS, lw=TICK_LW,
               label="Best SFT transmission (parameter-update ceiling)"),
        Line2D([0], [0], color=C_SALVE, lw=TICK_LW,
               label="SALVE recovered-prompt behavior (mean)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a1a1a",
               markeredgecolor="white", markersize=15,
               label="Seed: recovered text names trait"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a1a1a",
               markeredgecolor="white", markersize=8,
               label="Seed: does not name trait"),
    ]
    # Two rows keeps each entry readable at wide-figure widths — five on one
    # line squeezes labels into ellipses.
    fig.legend(handles=legend, loc="upper center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.00))
    fig.suptitle("SALVE recovery (per seed) vs references, per animal x induction method\n"
                 "(x-tick: SALVE K/N = seeds whose recovered text names the trait)",
                 fontsize=12, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.text(0.01, -0.04, _load.recipe_footer(),
             fontsize=7, family="monospace", color="#444444",
             ha="left", va="top")
    png = OUT_DIR / "induction_per_animal_ticks.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
