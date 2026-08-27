"""Prompted/steered teacher transmission vs SALVE recovery, in one 2x2.

Rows are teacher construction (prompted, steered); columns are the recovery
readout (explicit animal naming, plug-and-play behavior).  Every panel uses the
same x metric: student transmission lift at the fixed r8 / lr 2e-4 / 10 epoch
recipe.  One point is one (base model, animal) cell.

The recovery records use the uniform final decode pool: ``seed*_finalpool``
for the retrofitted Qwen/Llama runs and the native ``seed*`` runs for Olmo-3.
Prompted teachers have the original four-animal grid; steered teachers have the
pre-committed nine-animal grid.

  .venv/bin/python \
      final_plots/prompted_steered_recovery/plot_transmission_recovery_matrix.py

Outputs (alongside this script):
  transmission_recovery_matrix.{png,pdf}          (behavioral hit-rate lift)
  transmission_recovery_matrix_logprob.{png,pdf}  (mean log-probability lift)
  transmission_recovery_logprob_xy.{png,pdf}      (log-probability lift on x/y)

Gray dotted lines are pooled least-squares trends across the cells in a panel.
They are visual summaries only: points are clustered by base model and the
explicit-recovery outcome is a discrete count. Spearman rho remains the
reported association statistic.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.subliminal.animals import hits_trait
from final_plots.style import apply_style


OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
SEEDS = [42, 43, 44, 45]

MODELS = [
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B", "#4477AA"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B", "#CC3311"),
    ("Olmo-3-7B-Instruct", "Olmo-3-7B", "#009988"),
]
TEACHERS = [
    ("filtered_schrodi", "Prompted teachers", ["cat", "dog", "eagle", "owl"]),
    ("steering", "Steered teachers",
     ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger", "wolf"]),
]

# Hand-tuned only for the remaining legible labels. The discrete naming panel
# has many exact ties, so labels in its dense near-origin groups are omitted;
# every cell is still represented by a point.
LABEL_OFFSETS = {
    ("filtered_schrodi", "named", "Qwen2.5-7B-Instruct", "cat"): (4, -9),
    ("filtered_schrodi", "named", "Qwen2.5-7B-Instruct", "owl"): (4, 4),
    ("filtered_schrodi", "named", "Olmo-3-7B-Instruct", "owl"): (4, -9),
    ("steering", "named", "Qwen2.5-7B-Instruct", "panda"): (-28, 5),
    ("steering", "named", "Qwen2.5-7B-Instruct", "lion"): (-8, -11),
    ("steering", "named", "Qwen2.5-7B-Instruct", "wolf"): (-16, 14),
    ("steering", "named", "Qwen2.5-7B-Instruct", "penguin"): (-48, 5),
    ("steering", "named", "Qwen2.5-7B-Instruct", "eagle"): (-5, 29),
    ("steering", "named", "Qwen2.5-7B-Instruct", "tiger"): (-4, 17),
    ("steering", "accuracy", "Qwen2.5-7B-Instruct", "eagle"): (-29, 5),
    ("steering", "accuracy", "Qwen2.5-7B-Instruct", "wolf"): (4, -8),
}


def show_label(teacher, metric, point):
    """Keep identities where labels can be read without obscuring the trend."""
    if teacher == "filtered_schrodi" and metric == "named":
        # Six red points occupy x=0.00..0.03 at 4/4; labeling each says less
        # than the visible red cluster. Retain Qwen and Olmo identities.
        return point["model"] != "Llama-3.1-8B-Instruct"
    if teacher == "steering":
        # This removes only the compact failure cluster at the origin. The
        # higher-transmission cells—the ones driving the comparison—are named.
        return point["behavior_x"] >= 0.15
    # Continuous prompted panel: omit the tight x~0 red cluster but retain the
    # isolated Qwen/Olmo cells and all cells to its right.
    return (point["model"] != "Llama-3.1-8B-Instruct"
            or point["behavior_x"] >= 0.05)


def _hit_rate(value):
    """Read a hit rate from either the scalar or current nested schema."""
    return float(value["hit_rate"] if isinstance(value, dict) else value)


def transmission_lift(model, teacher, animal):
    """Mean student-minus-floor lift at r8 / lr 2e-4 / 10 epochs.

    Qwen/Llama prompted runs use the older one-directory-per-lr layout; all
    steered runs and the Olmo prompted wave use the newer lr-inside-seed layout.
    Averaging is over however many independently trained students landed.
    """
    base = ROOT / "transmission" / model / teacher / animal
    if teacher == "filtered_schrodi" and model != "Olmo-3-7B-Instruct":
        paths = list(base.glob("r8_lr2e-4_ep10/seed*/transmission.json"))
    else:
        paths = list(base.glob("r8_ep10/seed*/lr0.0002/transmission.json"))

    lifts = []
    for path in paths:
        record = json.loads(path.read_text())
        lifts.append(float(record.get(
            "lift", _hit_rate(record["student"]) - _hit_rate(record["floor"]))))
    return float(np.mean(lifts)) if lifts else None


def transmission_logprob_lift(model, teacher, animal):
    """Mean change in the smooth trait score, in nats.

    ``avg_log_likelihood`` is the mean per-token log probability of the
    canonical animal-label answer over the behavioral prompts. Therefore this
    difference is also log(student geomean probability / floor geomean
    probability), because geomean_probability = exp(avg_log_likelihood).
    """
    base = ROOT / "transmission" / model / teacher / animal
    if teacher == "filtered_schrodi" and model != "Olmo-3-7B-Instruct":
        paths = list(base.glob("r8_lr2e-4_ep10/seed*/transmission.json"))
    else:
        paths = list(base.glob("r8_ep10/seed*/lr0.0002/transmission.json"))

    changes = []
    for path in paths:
        record = json.loads(path.read_text())
        changes.append(float(record["student"]["avg_log_likelihood"]
                             - record["floor"]["avg_log_likelihood"]))
    return float(np.mean(changes)) if changes else None


def recovery(model, teacher, animal):
    """Return (number naming animal, mean plug-and-play hit rate, n records)."""
    suffix = "" if model == "Olmo-3-7B-Instruct" else "_finalpool"
    records = []
    for seed in SEEDS:
        path = (ROOT / model / teacher / f"seed{seed}{suffix}" / "prefill_t1"
                / animal / "salve_beam.json")
        if path.exists():
            records.append(json.loads(path.read_text()))
    if not records:
        return None
    named = sum(hits_trait(record.get("best_text") or "", animal)
                for record in records)
    accuracy = float(np.mean([record["behavior"]["hit_rate"]
                              for record in records]))
    return named, accuracy, len(records)


def recovery_logprob_lift(model, teacher, animal):
    """Mean recovered-prompt minus no-prompt log-probability change.

    The floor comes from the recovery evaluation's own ``baselines.json`` so
    each y effect is paired with the evaluator that produced its prompt score.
    Behavior baselines are method-independent; the fallback handles methods
    whose duplicate baseline file was not materialized.
    """
    floor = None
    for fallback_teacher in (teacher, "steering", "filtered_schrodi", "prompted"):
        path = (ROOT / model / fallback_teacher / "baselines" / "prefill_t1"
                / animal / "baselines.json")
        if path.exists():
            floor = json.loads(path.read_text())["no_prompt"]["behavior"].get(
                "avg_log_likelihood")
            if floor is not None:
                break
    if floor is None:
        return None

    suffix = "" if model == "Olmo-3-7B-Instruct" else "_finalpool"
    changes = []
    for seed in SEEDS:
        path = (ROOT / model / teacher / f"seed{seed}{suffix}" / "prefill_t1"
                / animal / "salve_beam.json")
        if path.exists():
            behavior = json.loads(path.read_text())["behavior"]
            if behavior.get("avg_log_likelihood") is not None:
                changes.append(float(behavior["avg_log_likelihood"] - floor))
    return (float(np.mean(changes)), len(changes)) if changes else None


def collect_points(x_metric="behavior"):
    if x_metric not in {"behavior", "logprob"}:
        raise ValueError(f"unknown x metric: {x_metric}")
    points = {teacher: [] for teacher, _, _ in TEACHERS}
    for teacher, _, animals in TEACHERS:
        for model, _, color in MODELS:
            for animal in animals:
                behavior_x = transmission_lift(model, teacher, animal)
                logprob_x = transmission_logprob_lift(model, teacher, animal)
                x = behavior_x if x_metric == "behavior" else logprob_x
                recovered = recovery(model, teacher, animal)
                if x is None or recovered is None:
                    continue
                named, accuracy, n = recovered
                points[teacher].append({
                    "model": model, "animal": animal, "color": color,
                    "x": x, "behavior_x": behavior_x, "logprob_x": logprob_x,
                    "named": named, "accuracy": accuracy, "n": n,
                })
    return points


def render(x_metric, output_stem, x_label):
    apply_style()
    points = collect_points(x_metric)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex=True,
                             gridspec_kw={"hspace": 0.22, "wspace": 0.26})
    all_x = [point["x"] for rows in points.values() for point in rows]
    x_pad = 0.04 * (max(all_x) - min(all_x))
    x_limits = (min(all_x) - x_pad, max(all_x) + x_pad)

    for row, (teacher, teacher_label, _) in enumerate(TEACHERS):
        for col, (metric, y_label) in enumerate([
                ("named", "SALVE runs naming animal"),
                ("accuracy", "Mean plug-and-play accuracy")]):
            ax = axes[row, col]
            xs, ys = [], []
            for point in points[teacher]:
                x, y = point["x"], point[metric]
                xs.append(x)
                ys.append(y)
                ax.scatter(x, y, s=47, color=point["color"], zorder=3,
                           linewidths=0.6, edgecolors="white")
                key = (teacher, metric, point["model"], point["animal"])
                if show_label(teacher, metric, point):
                    ax.annotate(point["animal"], (x, y),
                                # A few tied rows need a different side/height.
                                xytext=LABEL_OFFSETS.get(key, (4, 3)),
                                textcoords="offset points", fontsize=6.3,
                                color="#888888")

            # Neutral pooled trend, restricted to the observed x range. Clip
            # predictions to the outcome's support so the guide cannot imply
            # impossible naming counts or probabilities.
            trend_x = np.linspace(min(xs), max(xs), 200)
            slope, intercept = np.polyfit(xs, ys, 1)
            trend_y = slope * trend_x + intercept
            trend_y = np.clip(trend_y, 0, 4 if metric == "named" else 1)
            ax.plot(trend_x, trend_y, color="#777777", ls=":", lw=1.5,
                    zorder=1)

            rho = stats.spearmanr(xs, ys).statistic if len(xs) > 1 else np.nan
            # Prompted points occupy the upper edge; steered points occupy the
            # lower-right edge, so place the statistic in the opposite corner.
            rho_y, rho_va = (0.06, "bottom") if row == 0 else (0.94, "top")
            ax.text(0.97, rho_y, f"all cells  $\\rho$={rho:+.2f}",
                    transform=ax.transAxes, ha="right", va=rho_va, fontsize=8,
                    color="#666666")
            ax.set_xlim(*x_limits)
            ax.set_ylabel(y_label)
            if col == 0:
                ax.set_ylim(-0.25, 4.35)
                ax.set_yticks(range(5), [f"{k}/4" for k in range(5)])
            else:
                ax.set_ylim(-0.04, 1.04)

        axes[row, 0].text(-0.23, 0.5, teacher_label, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=12,
                          fontweight="bold")

    axes[0, 0].set_title("Explicit recovery")
    axes[0, 1].set_title("Plug-and-play recovery")
    handles = [plt.Line2D([], [], marker="o", ls="", color=color, label=label,
                          markersize=7) for _, label, color in MODELS]
    handles.append(plt.Line2D([], [], color="#777777", ls=":", lw=1.5,
                              label="Pooled linear trend"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.53, 0.005), handlelength=1.0)
    fig.supxlabel(x_label, x=0.56, y=0.085, fontsize=13)
    fig.subplots_adjust(left=0.15, bottom=0.20, top=0.93, right=0.98)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{output_stem}.{ext}", dpi=200)
    plt.close(fig)
    counts = {teacher: len(rows) for teacher, rows in points.items()}
    print(f"wrote {OUT_DIR}/{output_stem}.png ({counts})")
    incomplete = [(teacher, p["model"], p["animal"], p["n"])
                  for teacher, rows in points.items() for p in rows if p["n"] != 4]
    if incomplete:
        print(f"WARNING: recovery cells with fewer than four records: {incomplete}")


def render_matched_logprob():
    """Two-panel apples-to-apples effect-size view: delta log P on both axes."""
    apply_style()
    panels = []
    all_values = []
    for teacher, teacher_label, animals in TEACHERS:
        rows = []
        for model, _, color in MODELS:
            for animal in animals:
                x = transmission_logprob_lift(model, teacher, animal)
                recovered = recovery_logprob_lift(model, teacher, animal)
                if x is None or recovered is None:
                    continue
                y, n = recovered
                rows.append({"model": model, "animal": animal, "color": color,
                             "x": x, "y": y, "n": n})
                all_values.extend([x, y])
        panels.append((teacher, teacher_label, rows))

    pad = 0.05 * (max(all_values) - min(all_values))
    limits = (min(all_values) - pad, max(all_values) + pad)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.45), sharex=True, sharey=True)
    for ax, (teacher, teacher_label, rows) in zip(axes, panels):
        xs = [row["x"] for row in rows]
        ys = [row["y"] for row in rows]
        for row in rows:
            ax.scatter(row["x"], row["y"], s=49, color=row["color"], zorder=3,
                       linewidths=0.6, edgecolors="white")
            # All 12 prompted labels fit. For the 27-cell steered panel, retain
            # cells away from the central low-effect cluster.
            show = (teacher == "filtered_schrodi" or row["x"] > 1.5
                    or abs(row["y"]) > 1.5)
            if show:
                ax.annotate(row["animal"], (row["x"], row["y"]), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.3,
                            color="#888888")

        line_x = np.linspace(min(xs), max(xs), 200)
        slope, intercept = np.polyfit(xs, ys, 1)
        ax.plot(line_x, slope * line_x + intercept, color="#777777", ls=":",
                lw=1.5, zorder=1)
        ax.plot(limits, limits, color="#BBBBBB", ls="--", lw=1.0, zorder=0)
        rho = stats.spearmanr(xs, ys).statistic
        ax.text(0.96, 0.06, f"all cells  $\\rho$={rho:+.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#666666")
        ax.set_title(teacher_label)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")

    fig.supxlabel("Student: Δ mean log P(animal label) [nats]",
                  x=0.55, y=0.105, fontsize=13)
    fig.supylabel("Recovered prompt: Δ mean log P(animal label) [nats]",
                  x=0.03, fontsize=13)
    handles = [plt.Line2D([], [], marker="o", ls="", color=color, label=label,
                          markersize=7) for _, label, color in MODELS]
    handles += [
        plt.Line2D([], [], color="#777777", ls=":", lw=1.5,
                   label="Pooled linear trend"),
        plt.Line2D([], [], color="#BBBBBB", ls="--", lw=1.0,
                   label="Recovery = transmission"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.52, 0.005), handlelength=1.4)
    fig.subplots_adjust(left=0.10, bottom=0.22, top=0.91, right=0.98, wspace=0.20)

    output_stem = "transmission_recovery_logprob_xy"
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{output_stem}.{ext}", dpi=200)
    plt.close(fig)
    counts = {teacher: len(rows) for teacher, _, rows in panels}
    print(f"wrote {OUT_DIR}/{output_stem}.png ({counts})")


def main():
    render("behavior", "transmission_recovery_matrix",
           "Student transmission (student − floor)")
    render("logprob", "transmission_recovery_matrix_logprob",
           "Student transmission: Δ mean log P(animal label) [nats]")
    render_matched_logprob()


if __name__ == "__main__":
    main()
