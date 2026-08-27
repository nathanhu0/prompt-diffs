"""Student transmission vs functional recovered-prompt behavior.

Two panels: prompted teachers (3 models x 4 animals) and steered teachers
(3 models x the full 9-animal family). Both axes use the same floor-adjusted
behavior-frequency effect size, in percentage points:

  x = student hit rate - transmission floor
  y = mean over the ORIGINAL FOUR SALVE seeds (42--45) of
      recovered-prompt hit rate - no-prompt recovery floor

Thus semantically partial prompts (for example, a bird-oriented prompt for an
eagle cell) receive credit exactly when they causally increase target behavior.
Later SALVE seeds are deliberately ignored.

The dotted line is a pooled least-squares visual guide; the faint dashed line is
y=x (functional recovery equals student transmission). Spearman rho is shown as
the monotone association summary.

  MPLCONFIGDIR=/tmp/mpl-headline .venv/bin/python \
      final_plots/prompted_steered_recovery/plot_behavior_recovery_headline.py

Output (alongside this script): behavior_recovery_headline.{png,pdf}
"""
import json
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style
from final_plots.prompted_steered_recovery.plot_transmission_recovery_matrix import (
    MODELS, ROOT, SEEDS, TEACHERS, transmission_lift,
)


OUT_DIR = Path(__file__).parent
OUTPUT_STEM = "behavior_recovery_headline"
COMMON_ANIMALS = {"cat", "dog", "eagle", "owl"}

LABEL_OFFSETS = {
    ("filtered_schrodi", "Qwen2.5-7B-Instruct", "cat"): (-8, 7),
    ("filtered_schrodi", "Qwen2.5-7B-Instruct", "eagle"): (4, -9),
    ("filtered_schrodi", "Qwen2.5-7B-Instruct", "owl"): (4, 4),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "cat"): (4, 5),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "dog"): (4, -9),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "eagle"): (4, 4),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "owl"): (4, -9),
    ("steering", "Qwen2.5-7B-Instruct", "eagle"): (-28, 5),
    ("steering", "Qwen2.5-7B-Instruct", "wolf"): (4, -8),
}


def recovery_floor(model, teacher, animal):
    """Matched no-prompt behavioral floor, with method-independent fallback."""
    for fallback_teacher in (teacher, "steering", "filtered_schrodi", "prompted"):
        path = (ROOT / model / fallback_teacher / "baselines" / "prefill_t1"
                / animal / "baselines.json")
        if path.exists():
            return float(json.loads(path.read_text())["no_prompt"]["behavior"]
                         ["hit_rate"])
    return None


def recovery_behavior_lift(model, teacher, animal):
    """(mean prompt-minus-floor hit-rate lift, number of original seeds)."""
    floor = recovery_floor(model, teacher, animal)
    if floor is None:
        return None
    suffix = "" if model == "Olmo-3-7B-Instruct" else "_finalpool"
    lifts = []
    for seed in SEEDS:
        path = (ROOT / model / teacher / f"seed{seed}{suffix}" / "prefill_t1"
                / animal / "salve_beam.json")
        if path.exists():
            hit_rate = json.loads(path.read_text())["behavior"]["hit_rate"]
            lifts.append(float(hit_rate - floor))
    return (float(np.mean(lifts)), len(lifts)) if lifts else None


def collect_points(common_four=False):
    panels = []
    for teacher, label, animals in TEACHERS:
        if common_four:
            animals = [animal for animal in animals if animal in COMMON_ANIMALS]
        rows = []
        for model, _, color in MODELS:
            for animal in animals:
                x = transmission_lift(model, teacher, animal)
                recovered = recovery_behavior_lift(model, teacher, animal)
                if x is None or recovered is None:
                    continue
                y, n = recovered
                rows.append({"teacher": teacher, "model": model,
                             "animal": animal, "color": color,
                             "x": x, "y": y, "n": n})
        panels.append((teacher, label, rows))
    return panels


def show_label(row, common_four=False):
    if common_four or row["teacher"] == "filtered_schrodi":
        return True
    # Retain the cells carrying the visible steered relationship and omit the
    # dense near-origin cluster. All cells remain visible as points.
    return row["x"] > 0.15 or abs(row["y"]) > 0.15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-four", action="store_true",
        help="restrict both panels to cat, dog, eagle, and owl (12 cells each)")
    args = parser.parse_args()

    apply_style()
    panels = collect_points(common_four=args.common_four)
    all_values = [value for _, _, rows in panels for row in rows
                  for value in (row["x"], row["y"])]
    pad = 0.05 * (max(all_values) - min(all_values))
    limits_pp = (100 * (min(all_values) - pad),
                 100 * (max(all_values) + pad))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.35), sharex=True, sharey=True)
    for ax, (teacher, label, rows) in zip(axes, panels):
        xs = [row["x"] for row in rows]
        ys = [row["y"] for row in rows]
        for row in rows:
            x_pp, y_pp = 100 * row["x"], 100 * row["y"]
            ax.scatter(x_pp, y_pp, s=51, color=row["color"], zorder=3,
                       linewidths=0.6, edgecolors="white")
            if show_label(row, common_four=args.common_four):
                key = (teacher, row["model"], row["animal"])
                ax.annotate(row["animal"], (x_pp, y_pp),
                            xytext=LABEL_OFFSETS.get(key, (4, 3)),
                            textcoords="offset points", fontsize=6.3,
                            color="#888888")

        fit_x = np.linspace(min(xs), max(xs), 300)
        slope, intercept = np.polyfit(xs, ys, 1)
        ax.plot(100 * fit_x, 100 * (slope * fit_x + intercept),
                color="#777777", ls=":", lw=1.6, zorder=1)
        ax.plot(limits_pp, limits_pp, color="#BBBBBB", ls="--", lw=1.0,
                zorder=0)
        rho = stats.spearmanr(xs, ys).statistic
        ax.text(0.04, 0.94, f"all cells  $\\rho$={rho:+.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="#666666")
        ax.set_title(label)
        ax.set_xlim(*limits_pp)
        ax.set_ylim(*limits_pp)
        ax.set_aspect("equal", adjustable="box")

    fig.supxlabel("Change in student animal-response frequency (percentage points)",
                  x=0.55, y=0.105, fontsize=13)
    fig.supylabel("Change from recovered prompt (percentage points)",
                  x=0.035, fontsize=13)
    handles = [plt.Line2D([], [], marker="o", ls="", color=color, label=label,
                          markersize=7) for _, label, color in MODELS]
    handles += [
        plt.Line2D([], [], color="#777777", ls=":", lw=1.6,
                   label="Pooled linear fit"),
        plt.Line2D([], [], color="#BBBBBB", ls="--", lw=1.0,
                   label="Recovery = transmission"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.52, 0.005), handlelength=1.4)
    fig.subplots_adjust(left=0.10, bottom=0.23, top=0.90, right=0.98, wspace=0.19)

    output_stem = OUTPUT_STEM + ("_common4" if args.common_four else "")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{output_stem}.{ext}", dpi=200)
    counts = {teacher: len(rows) for teacher, _, rows in panels}
    print(f"wrote {OUT_DIR}/{output_stem}.png ({counts})")
    incomplete = [(row["model"], row["teacher"], row["animal"], row["n"])
                  for _, _, rows in panels for row in rows if row["n"] != 4]
    if incomplete:
        print(f"WARNING: expected exactly seeds 42--45, incomplete cells: {incomplete}")


if __name__ == "__main__":
    main()
