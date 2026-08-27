"""Tentative headline: behavioral student transmission vs explicit recovery.

Two panels: prompted teachers (3 models x 4 animals) and steered teachers
(3 models x the full 9-animal family).

  x = student animal-response frequency minus the no-adapter floor, displayed
      on the 0--1 frequency scale.
  y = number of the ORIGINAL FOUR SALVE seeds (42--45) whose selected prompt
      explicitly names the animal. Later seeds are deliberately ignored.

The dotted line is a pooled least-squares visual guide. Spearman rho is the
reported monotone association; neither summary adjusts for model clustering.

  MPLCONFIGDIR=/tmp/mpl-headline .venv/bin/python \
      final_plots/prompted_steered_recovery/plot_behavior_naming_headline.py

Output (alongside this script): behavior_naming_headline.{png,pdf}
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style
from final_plots.prompted_steered_recovery.plot_logprob_naming_headline import (
    LABEL_OFFSETS,
)
from final_plots.prompted_steered_recovery.plot_transmission_recovery_matrix import (
    MODELS, TEACHERS, recovery, transmission_lift,
)


OUT_DIR = Path(__file__).parent
OUTPUT_STEM = "behavior_naming_headline"
COMMON_ANIMALS = {"cat", "dog", "eagle", "owl"}
ANIMAL_MARKERS = {"cat": "o", "dog": "s", "eagle": "^", "owl": "D"}
OUTLIER_OFFSETS = {
    ("steering", "Qwen2.5-7B-Instruct", "eagle"): (-32, 22),
    ("steering", "Qwen2.5-7B-Instruct", "penguin"): (-42, 7),
}


def collect_points(common_four=False):
    panels = []
    for teacher, label, animals in TEACHERS:
        if common_four:
            animals = [animal for animal in animals if animal in COMMON_ANIMALS]
        rows = []
        for model, _, color in MODELS:
            for animal in animals:
                lift = transmission_lift(model, teacher, animal)
                recovered = recovery(model, teacher, animal)
                if lift is None or recovered is None:
                    continue
                named, _, n = recovered
                rows.append({
                    "teacher": teacher, "model": model, "animal": animal,
                    "color": color, "lift": lift, "named": named, "n": n,
                })
        panels.append((teacher, label, rows))
    return panels


def show_label(row, common_four=False):
    if row["teacher"] == "filtered_schrodi":
        return row["model"] != "Llama-3.1-8B-Instruct"
    return row["lift"] > 0.15 or row["named"] >= 2


def outlier_keys(rows, n_each=2):
    """Keys for the n largest positive and negative pooled-fit residuals."""
    xs = np.asarray([row["lift"] for row in rows])
    ys = np.asarray([row["named"] for row in rows])
    slope, intercept = np.polyfit(xs, ys, 1)
    residuals = ys - (slope * xs + intercept)
    chosen = list(np.argsort(residuals)[:n_each])
    chosen += list(np.argsort(residuals)[-n_each:])
    return {(rows[i]["teacher"], rows[i]["model"], rows[i]["animal"])
            for i in chosen}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-four", action="store_true",
        help="restrict both panels to cat, dog, eagle, and owl (12 cells each)")
    parser.add_argument(
        "--animal-markers", action="store_true",
        help="encode common-four animals by marker shape and omit point text")
    parser.add_argument(
        "--outlier-labels", action="store_true",
        help="use circles and label only the two largest +/- fit residuals")
    parser.add_argument(
        "--no-point-labels", action="store_true",
        help="omit all point text and write a companion cell-level CSV")
    parser.add_argument(
        "--y-jitter", action="store_true",
        help="display-only deterministic +/-0.06 jitter; fits use exact counts")
    args = parser.parse_args()
    if args.animal_markers and not args.common_four:
        parser.error("--animal-markers currently requires --common-four")
    if args.animal_markers and args.outlier_labels:
        parser.error("choose --animal-markers or --outlier-labels, not both")
    if args.no_point_labels and (args.animal_markers or args.outlier_labels):
        parser.error("--no-point-labels cannot be combined with label/marker variants")

    apply_style()
    panels = collect_points(common_four=args.common_four)
    jitter_rng = np.random.default_rng(20260825)
    # Preserve the few small negative lifts just left of zero while presenting
    # the requested 0.0, 0.2, ..., 1.0 tick labels.
    x_limits = (-0.04, 1.02)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.15), sharex=True, sharey=True)
    for ax, (teacher, label, rows) in zip(axes, panels):
        xs = [row["lift"] for row in rows]
        ys = [row["named"] for row in rows]
        selected_outliers = outlier_keys(rows) if args.outlier_labels else set()
        for row in rows:
            y_display = (row["named"] + jitter_rng.uniform(-0.06, 0.06)
                         if args.y_jitter else row["named"])
            ax.scatter(row["lift"], y_display, s=51,
                       color=row["color"], zorder=3,
                       marker=(ANIMAL_MARKERS.get(row["animal"], "o")
                               if args.animal_markers else "o"),
                       linewidths=0.6, edgecolors="white")
            key = (teacher, row["model"], row["animal"])
            label_point = (key in selected_outliers if args.outlier_labels
                           else show_label(row, common_four=args.common_four))
            if not args.animal_markers and not args.no_point_labels and label_point:
                ax.annotate(row["animal"], (row["lift"], row["named"]),
                            xytext=OUTLIER_OFFSETS.get(
                                key, LABEL_OFFSETS.get(key, (4, 3))),
                            textcoords="offset points", fontsize=6.3,
                            color="#888888")

        # Keep the guide tied to observed support, with a small visual extension
        # on the left; extend to x=1 for consistent panel geometry.
        fit_pad = 0.05 * (max(xs) - min(xs))
        fit_x = np.linspace(max(x_limits[0], min(xs) - fit_pad),
                            1.0, 300)
        slope, intercept = np.polyfit(xs, ys, 1)
        fit_count = np.clip(slope * fit_x + intercept, 0, 4)
        ax.plot(fit_x, fit_count, color="#777777", ls=":", lw=1.6,
                zorder=1)

        rho = stats.spearmanr(xs, ys).statistic
        # Put the compact statistic beside the line it summarizes; the caption
        # defines rho as pooled Spearman rank correlation.
        rho_x = 0.90
        rho_y = slope * rho_x + intercept - 0.16
        ax.text(rho_x, rho_y, f"$\\rho$ = {rho:+.2f}", fontsize=9,
                ha="center", va="top", color="#666666")
        ax.set_title(label, pad=8)
        ax.set_xlim(*x_limits)
        ax.set_ylim(-0.25, 4.35)
        ax.set_yticks(range(5), [f"{k}/4" for k in range(5)])
        ticks = np.arange(0, 1.01, 0.2)
        ax.set_xticks(ticks, [f"{tick:.1f}" for tick in ticks])
        ax.set_xlabel("Student Behavior Change")
        ax.set_ylabel("SALVE Prompts with Animal")
        ax.tick_params(axis="y", labelleft=True)

    handles = [plt.Line2D([], [], marker="o", ls="", color=color, label=label,
                          markersize=7) for _, label, color in MODELS]
    if args.animal_markers:
        handles += [
            plt.Line2D([], [], marker=ANIMAL_MARKERS[animal], ls="",
                       markerfacecolor="#777777", markeredgecolor="white",
                       color="#777777", label=animal.capitalize(), markersize=7)
            for animal in ("cat", "dog", "eagle", "owl")
        ]
    handles += [
        plt.Line2D([], [], color="#777777", ls=":", lw=1.6,
                   label="Pooled linear fit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.52, 0.005), handlelength=1.4)
    fig.subplots_adjust(left=0.10, bottom=0.22, top=0.89, right=0.98, wspace=0.30)

    output_stem = OUTPUT_STEM
    if args.common_four:
        output_stem += "_common4"
    if args.animal_markers:
        output_stem += "_markers"
    if args.outlier_labels:
        output_stem += "_outliers"
    if args.no_point_labels:
        output_stem += "_nolabels"
    if args.y_jitter:
        output_stem += "_yjitter"
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{output_stem}.{ext}", dpi=200)

    if args.no_point_labels:
        csv_path = OUT_DIR / f"{output_stem}.csv"
        fields = ["teacher", "model", "animal", "student_behavior_change",
                  "recovered_prompts_with_animal", "n_salve_seeds",
                  "linear_fit_residual"]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for teacher, _, rows in panels:
                xs = np.asarray([row["lift"] for row in rows])
                ys = np.asarray([row["named"] for row in rows])
                slope, intercept = np.polyfit(xs, ys, 1)
                for row in rows:
                    writer.writerow({
                        "teacher": teacher,
                        "model": row["model"],
                        "animal": row["animal"],
                        "student_behavior_change": f'{row["lift"]:.6f}',
                        "recovered_prompts_with_animal": row["named"],
                        "n_salve_seeds": row["n"],
                        "linear_fit_residual":
                            f'{row["named"] - (slope * row["lift"] + intercept):.6f}',
                    })
        print(f"wrote {csv_path}")
    counts = {teacher: len(rows) for teacher, _, rows in panels}
    print(f"wrote {OUT_DIR}/{output_stem}.png ({counts})")
    incomplete = [(row["model"], row["teacher"], row["animal"], row["n"])
                  for _, _, rows in panels for row in rows if row["n"] != 4]
    if incomplete:
        print(f"WARNING: expected exactly seeds 42--45, incomplete cells: {incomplete}")


if __name__ == "__main__":
    main()
