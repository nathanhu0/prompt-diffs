"""Steered teachers: behavioral student transmission vs explicit recovery.

One half-width figure, steered teachers only (3 models x the full 9-animal
family). The prompted-teacher panel this script used to produce was retired
2026-09-03: boosted_transfer/prompted_transmission_vs_recovery.py is the
prompted-teacher figure now. No pooled fit line or rho (retired the same
day); the legend sits in the empty upper-left corner, with a little y
headroom above 4/4 so it clears the 3/4 points.

  x = student animal-response frequency minus the no-adapter floor, displayed
      on the 0--1 frequency scale.
  y = number of the ORIGINAL FOUR SALVE seeds (42--45) whose selected prompt
      explicitly names the animal. Later seeds are deliberately ignored.

  MPLCONFIGDIR=/tmp/mpl-headline .venv/bin/python \
      final_plots/prompted_steered_recovery/plot_behavior_naming_headline.py

Outputs (alongside this script): behavior_naming_headline_steered*.{png,pdf}
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import (HALF_W, LABELS, MODEL_COLORS, MUTED_TEXT, WHITE,  # noqa: E402
                               apply_style, savefig_pair, short_model)
from final_plots.prompted_steered_recovery.plot_logprob_naming_headline import (  # noqa: E402
    LABEL_OFFSETS,
)
from final_plots.prompted_steered_recovery.plot_transmission_recovery_matrix import (  # noqa: E402
    MODELS, TEACHERS, recovery, transmission_lift,
)


OUT_DIR = Path(__file__).parent
OUTPUT_STEM = "behavior_naming_headline"
TEACHER_LABEL = {"filtered_schrodi": LABELS["prompted_teachers"],
                 "steering": LABELS["steered_teachers"]}
COMMON_ANIMALS = {"cat", "dog", "eagle", "owl"}
ANIMAL_MARKERS = {"cat": "o", "dog": "s", "eagle": "^", "owl": "D"}
OUTLIER_OFFSETS = {
    ("steering", "Qwen2.5-7B-Instruct", "eagle"): (-32, 22),
    ("steering", "Qwen2.5-7B-Instruct", "penguin"): (-42, 7),
}


def collect_points(common_four=False):
    panels = []
    for teacher, label, animals in TEACHERS:
        if teacher != "steering":
            continue
        if common_four:
            animals = [animal for animal in animals if animal in COMMON_ANIMALS]
        rows = []
        for model, _, _ in MODELS:
            for animal in animals:
                lift = transmission_lift(model, teacher, animal)
                recovered = recovery(model, teacher, animal)
                if lift is None or recovered is None:
                    continue
                named, _, n = recovered
                rows.append({
                    "teacher": teacher, "model": model, "animal": animal,
                    "color": MODEL_COLORS[model], "lift": lift, "named": named,
                    "n": n,
                })
        panels.append((teacher, TEACHER_LABEL[teacher], rows))
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


def draw_panel(ax, panel, args, jitter_rng, x_limits):
    teacher, label, rows = panel
    xs = [row["lift"] for row in rows]
    ys = [row["named"] for row in rows]
    selected_outliers = outlier_keys(rows) if args.outlier_labels else set()
    for row in rows:
        y_display = (row["named"] + jitter_rng.uniform(-0.06, 0.06)
                     if args.y_jitter else row["named"])
        ax.scatter(row["lift"], y_display, s=28,
                   color=row["color"], zorder=3,
                   marker=(ANIMAL_MARKERS.get(row["animal"], "o")
                           if args.animal_markers else "o"),
                   linewidths=0.5, edgecolors=WHITE)
        key = (teacher, row["model"], row["animal"])
        label_point = (key in selected_outliers if args.outlier_labels
                       else show_label(row, common_four=args.common_four))
        if not args.animal_markers and not args.no_point_labels and label_point:
            ax.annotate(row["animal"], (row["lift"], row["named"]),
                        xytext=OUTLIER_OFFSETS.get(
                            key, LABEL_OFFSETS.get(key, (3, 2))),
                        textcoords="offset points", fontsize=7,
                        color=MUTED_TEXT)

    print(f"{teacher}: pooled Spearman rho = {stats.spearmanr(xs, ys).statistic:+.2f} "
          f"(not drawn; caption material)")
    ax.set_title(label, pad=5)
    ax.set_xlim(*x_limits)
    ax.set_ylim(-0.25, 4.9)   # headroom above 4/4 for the in-axes legend
    ax.set_yticks(range(5), [f"{k}/4" for k in range(5)])
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xticks(ticks, [f"{tick:.1f}" for tick in ticks])
    ax.set_xlabel(LABELS["student_behavior_change"])
    ax.set_ylabel(LABELS["recovered_naming"].replace(" with", "\nwith"))
    return rows


def legend_handles(args):
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=MODEL_COLORS[model],
                   label=short_model(model), markersize=4.5)
        for model, _, _ in MODELS
    ]
    if args.animal_markers:
        handles += [
            plt.Line2D([], [], marker=ANIMAL_MARKERS[animal], ls="",
                       markerfacecolor=MUTED_TEXT, markeredgecolor=WHITE,
                       color=MUTED_TEXT, label=animal.capitalize(),
                       markersize=4.5)
            for animal in ("cat", "dog", "eagle", "owl")
        ]
    return handles


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
    handles = legend_handles(args)
    output_paths = []
    for panel in panels:
        teacher, _, _ = panel
        panel_name = "prompted" if teacher == "filtered_schrodi" else "steered"
        # same canvas as boosted_transfer's prompted scatter (half width)
        fig, ax = plt.subplots(figsize=(HALF_W, 2.1), layout="constrained")
        draw_panel(ax, panel, args, jitter_rng, x_limits)
        # single column because a full model name is ~1.2 in wide at 8 pt
        ax.legend(handles=handles, loc="upper left", ncol=1, handlelength=1.0)
        panel_stem = f"{OUTPUT_STEM}_{panel_name}{output_stem[len(OUTPUT_STEM):]}"
        savefig_pair(fig, OUT_DIR / panel_stem)
        output_paths += [OUT_DIR / f"{panel_stem}.{ext}" for ext in ("pdf", "png")]
        plt.close(fig)

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
    print(f"wrote split figures: {', '.join(str(p) for p in output_paths)} "
          f"({counts})")
    incomplete = [(row["model"], row["teacher"], row["animal"], row["n"])
                  for _, _, rows in panels for row in rows if row["n"] != 4]
    if incomplete:
        print(f"WARNING: expected exactly seeds 42--45, incomplete cells: {incomplete}")


if __name__ == "__main__":
    main()
