"""Tentative headline: smooth student transmission vs explicit SALVE recovery.

Two panels: prompted teachers (3 models x 4 animals) and steered teachers
(3 models x the full pre-committed 9-animal family). One point is a
(model, animal) cell.

  x = exp(student avg_log_likelihood - floor avg_log_likelihood), i.e. the
      multiplicative change in geometric-mean per-token probability assigned
      to the canonical animal-label answer. The log-scaled axis makes this the
      same ordering/geometry as delta log P while labeling it in readable
      multipliers (1x, 3.2x, 10x, ...).
  y = number of the ORIGINAL FOUR SALVE seeds (42--45) whose selected prompt
      explicitly names the animal. Later seeds are deliberately ignored.

The dotted curve is a pooled binomial-logistic maximum-likelihood fit (four
trials per cell). Spearman rho is reported as a model-light monotone summary;
neither statistic adjusts for clustering by base model.

  MPLCONFIGDIR=/tmp/mpl-headline .venv/bin/python \
      final_plots/prompted_steered_recovery/plot_logprob_naming_headline.py

Output (alongside this script): logprob_naming_headline.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style
from final_plots.prompted_steered_recovery.plot_transmission_recovery_matrix import (
    MODELS, TEACHERS, recovery, transmission_logprob_lift,
)


OUT_DIR = Path(__file__).parent
OUTPUT_STEM = "logprob_naming_headline"

# A readable subset across the roughly five-order-of-magnitude observed range.
# Retain 3.2x to make the half-decade interpretation concrete without labeling
# every minor half-decade and crowding the two panels.
MULTIPLIER_TICKS = [0.1, 1, 3.2, 10, 100, 1000]

# Offsets for discrete tied rows. Unlisted labels use (4, 3).
LABEL_OFFSETS = {
    ("filtered_schrodi", "Qwen2.5-7B-Instruct", "owl"): (4, 4),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "cat"): (-5, -12),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "dog"): (4, -12),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "eagle"): (-18, 7),
    ("filtered_schrodi", "Llama-3.1-8B-Instruct", "owl"): (4, 7),
    ("filtered_schrodi", "Olmo-3-7B-Instruct", "owl"): (-22, -12),
    ("steering", "Qwen2.5-7B-Instruct", "panda"): (-28, 5),
    ("steering", "Qwen2.5-7B-Instruct", "lion"): (-8, -11),
    ("steering", "Qwen2.5-7B-Instruct", "wolf"): (-16, 14),
    ("steering", "Qwen2.5-7B-Instruct", "penguin"): (-42, 5),
    ("steering", "Qwen2.5-7B-Instruct", "eagle"): (-5, 29),
    ("steering", "Qwen2.5-7B-Instruct", "tiger"): (-4, 17),
}


def collect_points():
    panels = []
    for teacher, label, animals in TEACHERS:
        rows = []
        for model, _, color in MODELS:
            for animal in animals:
                delta_logp = transmission_logprob_lift(model, teacher, animal)
                recovered = recovery(model, teacher, animal)
                if delta_logp is None or recovered is None:
                    continue
                named, _, n = recovered
                rows.append({
                    "teacher": teacher, "model": model, "animal": animal,
                    "color": color, "delta_logp": delta_logp,
                    "multiplier": float(np.exp(delta_logp)),
                    "named": named, "n": n,
                })
        panels.append((teacher, label, rows))
    return panels


def binomial_fit(delta_logp, successes, trials):
    """Fit logit(p)=intercept+slope*delta_logp to aggregate binomial cells."""
    x = np.asarray(delta_logp, dtype=float)
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    p0 = np.clip(k.sum() / n.sum(), 1e-4, 1 - 1e-4)
    initial = np.array([special.logit(p0), 0.0])

    def negative_log_likelihood(beta):
        eta = beta[0] + beta[1] * x
        return float(np.sum(n * np.logaddexp(0.0, eta) - k * eta))

    result = optimize.minimize(negative_log_likelihood, initial, method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"binomial fit failed: {result.message}")
    return result.x


def format_multiplier(value):
    if value < 1:
        return f"{value:g}×"
    return f"{value:,.0f}×" if value >= 10 else f"{value:g}×"


def show_label(row):
    # Omit the prompted Llama tie cluster at 4/4. In the 27-cell steered panel,
    # label recovered cells plus the strongest smooth-transmission cells and
    # leave the dense low-signal failure cluster as points only.
    if row["teacher"] == "filtered_schrodi":
        return row["model"] != "Llama-3.1-8B-Instruct"
    return row["delta_logp"] > 4.0 or row["named"] >= 2


def main():
    apply_style()
    panels = collect_points()
    all_delta = [row["delta_logp"] for _, _, rows in panels for row in rows]
    delta_pad = 0.04 * (max(all_delta) - min(all_delta))
    multiplier_limits = (np.exp(min(all_delta) - delta_pad),
                         np.exp(max(all_delta) + delta_pad))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.15), sharex=True, sharey=True)
    for ax, (teacher, label, rows) in zip(axes, panels):
        xs_log = [row["delta_logp"] for row in rows]
        ys = [row["named"] for row in rows]
        ns = [row["n"] for row in rows]
        for row in rows:
            ax.scatter(row["multiplier"], row["named"], s=51,
                       color=row["color"], zorder=3, linewidths=0.6,
                       edgecolors="white")
            if show_label(row):
                key = (teacher, row["model"], row["animal"])
                ax.annotate(row["animal"], (row["multiplier"], row["named"]),
                            xytext=LABEL_OFFSETS.get(key, (4, 3)),
                            textcoords="offset points", fontsize=6.3,
                            color="#888888")

        beta = binomial_fit(xs_log, ys, ns)
        fit_log = np.linspace(min(xs_log), max(xs_log), 300)
        # All displayed cells have four trials. Express the fitted probability
        # on the same 0/4--4/4 expected-count axis.
        fit_count = 4 * special.expit(beta[0] + beta[1] * fit_log)
        ax.plot(np.exp(fit_log), fit_count, color="#777777", ls=":", lw=1.6,
                zorder=1)
        ax.axvline(1.0, color="#BBBBBB", ls="--", lw=1.0, zorder=0)

        rho = stats.spearmanr(xs_log, ys).statistic
        ax.text(0.96, 0.07, f"all cells  $\\rho$={rho:+.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#666666")
        ax.set_title(label)
        ax.set_xscale("log")
        ax.set_xlim(*multiplier_limits)
        ax.set_ylim(-0.25, 4.35)
        ax.set_yticks(range(5), [f"{k}/4" for k in range(5)])
        ticks = [tick for tick in MULTIPLIER_TICKS
                 if multiplier_limits[0] <= tick <= multiplier_limits[1]]
        ax.set_xticks(ticks, [format_multiplier(tick) for tick in ticks])
        ax.minorticks_off()
        ax.tick_params(axis="x", labelsize=9)

    axes[0].set_ylabel("SALVE runs explicitly naming animal")
    fig.supxlabel("Student answer-score multiplier  exp(Δ mean log P)",
                  x=0.54, y=0.105, fontsize=13)
    handles = [plt.Line2D([], [], marker="o", ls="", color=color, label=label,
                          markersize=7) for _, label, color in MODELS]
    handles += [
        plt.Line2D([], [], color="#777777", ls=":", lw=1.6,
                   label="Binomial best fit"),
        plt.Line2D([], [], color="#BBBBBB", ls="--", lw=1.0,
                   label="No change (1×)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.52, 0.005), handlelength=1.4)
    fig.subplots_adjust(left=0.10, bottom=0.24, top=0.89, right=0.98, wspace=0.17)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{OUTPUT_STEM}.{ext}", dpi=200)
    counts = {teacher: len(rows) for teacher, _, rows in panels}
    print(f"wrote {OUT_DIR}/{OUTPUT_STEM}.png ({counts})")
    incomplete = [(row["model"], row["teacher"], row["animal"], row["n"])
                  for _, _, rows in panels for row in rows if row["n"] != 4]
    if incomplete:
        print(f"WARNING: expected exactly seeds 42--45, incomplete cells: {incomplete}")


if __name__ == "__main__":
    main()
