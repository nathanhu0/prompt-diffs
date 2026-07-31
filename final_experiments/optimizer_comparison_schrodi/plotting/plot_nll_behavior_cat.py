"""Single-panel headline scatter, cat task only: x = held-out val NLL,
y = behavior hit rate. Same encoding as plot_nll_behavior.py (color per method,
'*' if the recovered prompt names the trait, grayscale reference diamonds),
plus a light-grey annotation arrow showing up-and-left is better.

  uv run python final_experiments/optimizer_comparison_schrodi/plotting/plot_nll_behavior_cat.py
"""
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._load import collect_all
from final_experiments.optimizer_comparison_schrodi.plotting._trait import names_trait
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair, FIG_W_PER_PANEL, FIG_H)
from final_experiments.optimizer_comparison_schrodi.plotting.plot_nll_behavior import (
    METHOD_ORDER, METHOD_LABEL, COLORS, normalize_method,
    REF_COLORS, REF_LABEL, REF_MARKER_SIZE, load_references)
apply_style()

OUT_DIR = Path(__file__).parent
TASK = "cat"
# Headline trim: one method per family. Drops the regularized variants
# (GCG-reg, GBDA-reg — fluency story lives in the PPL table) and AutoDAN
# (appendix-tier, near-floor on both tasks).
# Order = color assignment (C0, C1, ...): SALVE blue, LARGO orange, then the
# methods that fail on cat.
SHOW_METHODS = ["salve_beam", "largo", "gcg_L", "opro", "pgd_noaux_L", "gbda_L"]
# Standard matplotlib default cycle (tab10), looped over the shown methods.
METHOD_COLORS = {m: f"C{i}" for i, m in enumerate(SHOW_METHODS)}
# Canonical reference gets eye-catching goldenrod ("gold standard") instead
# of black; the other two references stay recessive grayscale.
REF_COLORS_CAT = {**REF_COLORS, "canonical": "goldenrod"}
# Short legend labels (full definitions live in the caption) so the legend
# columns come out even.
REF_LABEL_CAT = {"canonical": "True Prompt", "qwen_default": "Qwen Default",
                 "empty": "No Prompt"}


def main():
    recs = collect_all()
    if not recs:
        print("no records yet"); return

    cells = defaultdict(list)
    for r in recs:
        m = normalize_method(r["method"])
        if r["task"] != TASK or r["nll_val"] is None or r["hit_rate"] is None:
            continue
        cells[m].append((r["seed"], r["nll_val"], r["hit_rate"], r["best_text"]))

    fig, ax = plt.subplots(figsize=(FIG_W_PER_PANEL, FIG_H))

    ref_cells = load_references().get(TASK, {})
    for name, c in REF_COLORS_CAT.items():
        rec = ref_cells.get(name)
        if not rec:
            continue
        ax.scatter(rec["nll_val"], rec["hit_rate"], s=REF_MARKER_SIZE, c=[c],
                   marker="D", edgecolors="black", linewidths=1.0, zorder=5)

    for m in SHOW_METHODS:
        c = METHOD_COLORS[m]
        for seed, nll, hit, txt in cells.get(m, []):
            marker = "*" if names_trait(txt, TASK) else "o"
            ax.scatter(nll, hit, s=140 if marker == "*" else 50,
                       c=[c], marker=marker, edgecolors="black", linewidths=0.6,
                       zorder=3)

    # Reading aid: up-and-left (lower NLL, higher behavior) is better. Small,
    # tucked into the top-right corner, light grey so it sits behind the data.
    ax.annotate("", xy=(0.80, 0.95), xytext=(0.94, 0.81),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="0.75", lw=2.0,
                                mutation_scale=16), zorder=1)
    ax.annotate("better", xy=(0.89, 0.89), xycoords="axes fraction",
                color="0.6", fontsize=11, ha="left", va="bottom",
                fontstyle="italic", zorder=1)

    # style-deviation: task-specific axis labels (not the fixed strings) — the
    # cat-only headline wants the numbers-objective vs cat-behavior contrast
    # visible on the axes themselves. Held-out val NLL; say "validation" in
    # the caption.
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
    for name, c in REF_COLORS_CAT.items():
        ref_handles.append(mlines.Line2D([], [], marker="D", linestyle="", color=c,
                                         markeredgecolor="black", markersize=8,
                                         label=REF_LABEL_CAT[name]))
    # style-deviation: two frameless single-row legends — optimization methods
    # on top, star + prompt references below. The split IS the separation; no
    # headers or divider lines. (fig.legend appends, so two calls coexist.)
    fig.tight_layout(rect=[0, 0.15, 1, 1.0])
    fig.legend(handles=method_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.065), ncol=len(method_handles),
               frameon=False, fontsize=9, columnspacing=1.2, handletextpad=0.4)
    fig.legend(handles=ref_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(ref_handles),
               frameon=False, fontsize=9, columnspacing=1.2, handletextpad=0.4)
    stem = OUT_DIR / "nll_vs_behavior_cat"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.pdf, {stem}.png", flush=True)


if __name__ == "__main__":
    main()
