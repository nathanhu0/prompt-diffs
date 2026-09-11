"""Headline scatter, cat task only: x = held-out val NLL, y = behavior hit
rate. One point per seed, color per method (one method per family), '*' marker
if the recovered prompt names the trait, reference prompts as diamonds
(goldenrod = data-generating prompt), light-grey "better" arrow toward
up-and-left. LARGO points come from the padded largo_t25 arm.

Shares record loading with the sibling build_metrics_table.py (same folder) so
the figure and tables can't disagree on the data.

Half-width figure.

  uv run python final_plots/optimizer_comparison/plot_nll_behavior_cat.py
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_plots.style import (GREY, HALF_W, INK, LIGHT_GREY, METHOD_COLORS,  # noqa: E402
                               MUTED_TEXT, OCHRE, apply_style, savefig_pair)
from build_metrics_table import (  # noqa: E402
    REFERENCES, collect_records, names_trait)

OUT_DIR = Path(__file__).parent
TASK = "cat"

# Headline trim: one method per family. Drops the regularized variants
# (GCG-reg, GBDA-reg — fluency story lives in the metrics table) and AutoDAN
# (appendix-tier, near-floor on both tasks). Colors from style.METHOD_COLORS.
SHOW_METHODS = ["salve_beam", "largo", "gcg_L", "opro", "pgd_noaux_L", "gbda_L"]
METHOD_LABEL = {"salve_beam": "SALVE (ours)", "largo": "LARGO", "gcg_L": "GCG",
                "opro": "OPRO", "pgd_noaux_L": "PGD", "gbda_L": "GBDA"}
# Canonical reference gets the palette's ochre ("gold standard"); the other
# two references stay recessive greys.
REF_COLORS = {"canonical": OCHRE, "qwen_default": GREY, "empty": LIGHT_GREY}
REF_LABEL = {"canonical": "True Prompt", "qwen_default": "Default System Prompt",
             "empty": "No System Prompt"}
REF_MARKER_SIZE = 45


def main():
    apply_style()

    cells = defaultdict(list)
    for r in collect_records():
        if r["task"] != TASK or r["nll_val"] is None or r["hit_rate"] is None:
            continue
        cells[r["method"]].append(r)

    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W), layout="constrained")

    refs = json.loads(REFERENCES.read_text()).get(TASK, {})
    for name, c in REF_COLORS.items():
        rec = refs.get(name)
        if not rec:
            continue
        ax.scatter(rec["nll_val"], rec["hit_rate"], s=REF_MARKER_SIZE, c=[c],
                   marker="D", edgecolors=INK, linewidths=0.6, zorder=5)

    for m in SHOW_METHODS:
        c = METHOD_COLORS[m]
        for r in cells.get(m, []):
            marker = "*" if names_trait(r["best_text"], TASK) else "o"
            # Star has significant internal negative space: bumped size to
            # visually equalize apparent area with the circles.
            ax.scatter(r["nll_val"], r["hit_rate"],
                       s=60 if marker == "*" else 22,
                       c=[c], marker=marker, edgecolors=INK,
                       linewidths=0.4, zorder=3)

    # Reading aid: up-and-left (lower NLL, higher behavior) is better. Tucked
    # into the top-right corner, light grey so it sits behind the data.
    ax.annotate("", xy=(0.80, 0.95), xytext=(0.94, 0.81),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="0.75", lw=1.2,
                                mutation_scale=9), zorder=1)
    ax.annotate("better", xy=(0.89, 0.89), xycoords="axes fraction",
                color=MUTED_TEXT, fontsize=7, ha="left", va="bottom",
                fontstyle="italic", zorder=1)

    # Task-specific axis labels — the numbers-objective vs cat-behavior
    # contrast lives on the axes. Held-out val NLL; say "validation" in the
    # caption.
    ax.set_xlabel("Number Dataset NLL")
    ax.set_ylabel("Cat Pick Rate")
    ax.set_ylim(-0.05, 1.05)

    method_handles = [
        mlines.Line2D([], [], marker="o", linestyle="", color=METHOD_COLORS[m],
                      markeredgecolor=INK, markeredgewidth=0.4, markersize=5,
                      label=METHOD_LABEL[m])
        for m in SHOW_METHODS if m in cells]
    ref_handles = [mlines.Line2D([], [], marker="*", linestyle="", color="white",
                                 markeredgecolor=INK, markeredgewidth=0.5,
                                 markersize=8, label="Prompt Names Cat")]
    for name, c in REF_COLORS.items():
        ref_handles.append(mlines.Line2D([], [], marker="D", linestyle="",
                                         color=c, markeredgecolor=INK,
                                         markeredgewidth=0.4, markersize=5,
                                         label=REF_LABEL[name]))
    # Two frameless legends inside the empty middle band of the axes —
    # methods above, star + prompt references below. The split IS the
    # separation; no headers or dividers.
    methods = ax.legend(handles=method_handles, loc="center", ncol=2,
                        bbox_to_anchor=(0.62, 0.66), columnspacing=1.0,
                        handletextpad=0.3)
    ax.add_artist(methods)
    ax.legend(handles=ref_handles, loc="center", ncol=1,
              bbox_to_anchor=(0.62, 0.36), handletextpad=0.3)
    savefig_pair(fig, OUT_DIR / "nll_vs_behavior_cat")


if __name__ == "__main__":
    main()
