"""Headline scatter, cat task only: x = held-out val NLL, y = behavior hit
rate. One point per seed, color per method (one method per family), '*' marker
if the recovered prompt names the trait, reference prompts as diamonds
(goldenrod = data-generating prompt), light-grey "better" arrow toward
up-and-left. LARGO points come from the padded largo_t25 arm.

Shares record loading with the sibling build_metrics_table.py (same folder) so
the figure and tables can't disagree on the data.

  uv run python final_plots/optimizer_comparison/plot_nll_behavior_cat.py
"""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from build_metrics_table import (
    REFERENCES, collect_records, names_trait)

OUT_DIR = Path(__file__).parent
TASK = "cat"

# Headline trim: one method per family. Drops the regularized variants
# (GCG-reg, GBDA-reg — fluency story lives in the metrics table) and AutoDAN
# (appendix-tier, near-floor on both tasks).
# Order = color assignment (C0, C1, ...): SALVE blue, LARGO orange, then the
# methods that fail on cat.
SHOW_METHODS = ["salve_beam", "largo", "gcg_L", "opro", "pgd_noaux_L", "gbda_L"]
METHOD_LABEL = {"salve_beam": "SALVE (ours)", "largo": "LARGO", "gcg_L": "GCG",
                "opro": "OPRO", "pgd_noaux_L": "PGD", "gbda_L": "GBDA"}
METHOD_COLORS = {m: f"C{i}" for i, m in enumerate(SHOW_METHODS)}
# Canonical reference gets eye-catching goldenrod ("gold standard"); the other
# two references stay recessive grayscale.
REF_COLORS = {"canonical": "goldenrod", "qwen_default": "0.45", "empty": "0.75"}
REF_LABEL = {"canonical": "True Prompt", "qwen_default": "Qwen Default",
             "empty": "No Prompt"}
REF_MARKER_SIZE = 110


def apply_style():
    plt.rcParams.update({
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "figure.dpi":         200,
        "font.family":        "DejaVu Sans",
        # PDF text stays as text (not paths) so search / paper-render is clean.
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def main():
    apply_style()

    cells = defaultdict(list)
    for r in collect_records():
        if r["task"] != TASK or r["nll_val"] is None or r["hit_rate"] is None:
            continue
        cells[r["method"]].append(r)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    refs = json.loads(REFERENCES.read_text()).get(TASK, {})
    for name, c in REF_COLORS.items():
        rec = refs.get(name)
        if not rec:
            continue
        ax.scatter(rec["nll_val"], rec["hit_rate"], s=REF_MARKER_SIZE, c=[c],
                   marker="D", edgecolors="black", linewidths=1.0, zorder=5)

    for m in SHOW_METHODS:
        c = METHOD_COLORS[m]
        for r in cells.get(m, []):
            marker = "*" if names_trait(r["best_text"], TASK) else "o"
            # Star has significant internal negative space: bumped size to
            # visually equalize apparent area with the circles.
            ax.scatter(r["nll_val"], r["hit_rate"],
                       s=140 if marker == "*" else 50,
                       c=[c], marker=marker, edgecolors="black",
                       linewidths=0.6, zorder=3)

    # Reading aid: up-and-left (lower NLL, higher behavior) is better. Tucked
    # into the top-right corner, light grey so it sits behind the data.
    ax.annotate("", xy=(0.80, 0.95), xytext=(0.94, 0.81),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="0.75", lw=2.0,
                                mutation_scale=16), zorder=1)
    ax.annotate("better", xy=(0.89, 0.89), xycoords="axes fraction",
                color="0.6", fontsize=11, ha="left", va="bottom",
                fontstyle="italic", zorder=1)

    # Task-specific axis labels — the numbers-objective vs cat-behavior
    # contrast lives on the axes. Held-out val NLL; say "validation" in the
    # caption.
    ax.set_xlabel("Number Dataset NLL")
    ax.set_ylabel("Rate of Picking Cat")
    ax.set_ylim(-0.05, 1.05)

    method_handles = [
        mlines.Line2D([], [], marker="o", linestyle="", color=METHOD_COLORS[m],
                      markeredgecolor="black", markersize=8,
                      label=METHOD_LABEL[m])
        for m in SHOW_METHODS if m in cells]
    ref_handles = [mlines.Line2D([], [], marker="*", linestyle="", color="white",
                                 markeredgecolor="black", markersize=12,
                                 label="Prompt Names Cat")]
    for name, c in REF_COLORS.items():
        ref_handles.append(mlines.Line2D([], [], marker="D", linestyle="",
                                         color=c, markeredgecolor="black",
                                         markersize=8, label=REF_LABEL[name]))
    # Two frameless single-row legends — methods on top, star + prompt
    # references below. The split IS the separation; no headers or dividers.
    fig.tight_layout(rect=[0, 0.15, 1, 1.0])
    fig.legend(handles=method_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.065), ncol=len(method_handles),
               frameon=False, columnspacing=1.2, handletextpad=0.4)
    fig.legend(handles=ref_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(ref_handles),
               frameon=False, columnspacing=1.2, handletextpad=0.4)
    stem = OUT_DIR / "nll_vs_behavior_cat"
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext))
    print(f"wrote {stem}.pdf, {stem}.png", flush=True)


if __name__ == "__main__":
    main()
