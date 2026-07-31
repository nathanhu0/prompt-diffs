"""6-panel grid for the non-mixture pairs, behavior + SALVE beam-mention metrics
superimposed on the same axes.

Rows = diluter (random | control); cols = primary animal (cat | dog | eagle).
Each panel:
  * Student LoRA hit-rate (blue line, from completions.json)
  * No-prompt baseline (gray dotted)
  * SALVE argmin behavior per seed (red scatter, ★/○ on best_text mention)
  * SALVE beam-mention metrics (purple, aggregated across 4 seeds):
        - all candidates   (solid)
        - top 10% by val NLL  (dashed)
        - argmin (k/4)     (dotted, red -- matches the SALVE argmin scatter color)

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dilution_grid_detailed.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import PAIRS
from experiments.control_dilution.plotting.plot_dilution_detailed import (
    _behavior_data, _beam_metrics, _disk_fractions, _disk_fractions_transmission,
)

OUT_DIR = Path(__file__).parent
ROWS = ["random", "control"]      # diluter
COLS = ["cat", "dog", "eagle"]    # primary animal
DILUTER_NAME = {"random": "uniform numbers", "control": "unprompted numbers"}


def pair_for(animal, diluter):
    name = f"{animal}_{diluter}"
    return name if name in PAIRS else None


def _draw_panel(ax, pair):
    animal = pair.split("_")[0]
    fs = sorted(set(_disk_fractions(pair)) | set(_disk_fractions_transmission(pair)))
    if not fs:
        return
    # Behavior side.
    stu, floor_mean, salve_pts = _behavior_data(pair, animal, fs)
    if stu:
        xs, ys = zip(*stu)
        ax.plot(xs, ys, "s-", color="C0", ms=5, lw=1.5, label="student LoRA")
    if floor_mean is not None:
        ax.axhline(floor_mean, color="gray", linestyle=":", lw=1.0,
                   label=f"no-prompt ≈ {floor_mean:.2f}")
    label_used = False
    for f, _seed, hr, text in salve_pts:
        is_hit = bool(text) and hits_trait(text, animal)
        ax.scatter([f], [hr], marker="*" if is_hit else "^",
                   color="C3",
                   s=90 if is_hit else 22,
                   edgecolors="black" if is_hit else "0.3",
                   linewidths=0.4, alpha=0.85, zorder=3,
                   label=("SALVE behavior" if not label_used else None))
        label_used = True
    # Prompt-mention side: keep only k/4 (fraction of 4 seeds whose val-argmin
    # candidate mentions animal). The all-candidates / top-10% versions tracked
    # this pretty closely, so they were extra ink.
    xs, _all_m, _top_m, argmin_m = _beam_metrics(pair, animal, fs)
    if xs:
        ax.plot(xs, argmin_m, ":", color="C3", lw=1.2,
                label="SALVE prompt mentions (k/4)")


def main():
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(15, 8.5),
                             sharex=True, sharey=True, squeeze=False)
    for r, dil in enumerate(ROWS):
        for c, animal in enumerate(COLS):
            ax = axes[r, c]
            pair = pair_for(animal, dil)
            if pair is None:
                ax.set_visible(False)
                continue
            _draw_panel(ax, pair)
            ax.set_title(f"{animal} + {DILUTER_NAME[dil]}", fontsize=10)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if r == len(ROWS) - 1:
                ax.set_xlabel(f"{animal} fraction")
            if c == 0:
                ax.set_ylabel("rate (behavior or prompt mention)")

    # Single shared legend below the grid so we don't reprint per-panel.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Dilution sweep (old grid) — behavior + SALVE prompt-mention (k/4).  "
        "Blue = student LoRA;  gray dotted = no-prompt floor;  "
        "red ★/△ = SALVE per-seed behavior (★ if best_text mentions animal);  "
        "red dotted = fraction of 4 SALVE seeds whose best_text mentions animal.",
        fontsize=9, y=1.005)
    fig.tight_layout()
    png = OUT_DIR / "dilution_grid_detailed.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
