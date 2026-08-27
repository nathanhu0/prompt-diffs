"""Contrastive vs separate LLS selection on the official Political Compass,
one panel per student model.

Same figure language as political_compass_grid.py: axes are the test's own
-10..+10, small dots are individual sittings (one traversal of the real
politicalcompass.org form per sampled response set; generation variance at
temperature 1.0, every cell is training seed 42), large markers are the
5-sitting means. Here each panel overlays BOTH selection methods at beta 0.08:
filled markers / solid shift lines = contrastive selection
(margin(left persona) - margin(right persona), one scored pool -> both tails),
open markers / dashed shift lines = separate selection (v2filter, one persona
vs no-prompt per side). Lines run control -> arm mean. The separate-selection
qwen7b left cell collapsed in training and has no coordinates.

Reads <run>/pct_coords_<ckpt>.json (see pct_submit.py).
"""
import glob, json, os, sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from political_transfer_grid import MODELS, ROOT, COLLAPSED, SURFACE, INK, MUTED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BETA = "0.08"

RIGHT, LEFT, NEUTRAL = "#e34948", "#2a78d6", "#898781"
Q_LEFT_LIB, Q_RIGHT_LIB = "#eaf1fb", "#fdeeee"
Q_LEFT_AUTH, Q_RIGHT_AUTH = "#eef4fa", "#fbf1f1"

METHODS = [("contrastive", "contrastive", "political_{arm}_contrastive_{tag}"),
           ("separate", "v2filter", "political_{arm}_v2filter_{tag}")]


def coords(run):
    fs = glob.glob(os.path.join(ROOT, run, "pct_coords_*.json"))
    if not fs:
        return None
    d = json.load(open(fs[0]))
    if not d.get("summary"):
        return None
    return {"econ": d["summary"]["economic"], "soc": d["summary"]["social"],
            "econ_mean": d["summary"]["economic_mean"],
            "soc_mean": d["summary"]["social_mean"]}


def main():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.9),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (tag, full, label) in zip(axes, MODELS):
        ax.set_facecolor(SURFACE)
        for (x0, x1, y0, y1, c) in [(-10, 0, 0, 10, Q_LEFT_AUTH), (0, 10, 0, 10, Q_RIGHT_AUTH),
                                    (-10, 0, -10, 0, Q_LEFT_LIB), (0, 10, -10, 0, Q_RIGHT_LIB)]:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=c, zorder=0))
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        ax.axvline(0, color=MUTED, lw=1.0, zorder=1)

        ctl = coords(f"control_{full}_beta{BETA}_lr0.0001_n25000_seed42")
        if ctl is not None:
            ax.scatter(ctl["econ"], ctl["soc"], s=14, color=NEUTRAL, alpha=0.40,
                       edgecolors="none", zorder=3)
            ax.scatter(ctl["econ_mean"], ctl["soc_mean"], s=95, color=NEUTRAL,
                       marker="s", edgecolors=SURFACE, linewidths=1.4, zorder=5)

        for arm, color in (("right", RIGHT), ("left", LEFT)):
            for method, run_token, pattern in METHODS:
                if method == "separate" and (tag, arm, BETA) in COLLAPSED:
                    ax.text(0.04, 0.04, "v2filter left: collapsed", transform=ax.transAxes,
                            fontsize=7.5, color=MUTED, ha="left", va="bottom")
                    continue
                run = (pattern.format(arm=arm, tag=tag)
                       + f"_beta{BETA}_lr0.0001_n25000_seed42")
                c = coords(run)
                if c is None:
                    continue
                filled = method == "contrastive"
                mark = dict(color=color, edgecolors=SURFACE, linewidths=1.4) if filled \
                    else dict(facecolors="none", edgecolors=color, linewidths=1.6)
                dots = dict(color=color, edgecolors="none") if filled \
                    else dict(facecolors="none", edgecolors=color, linewidths=0.7)
                ax.scatter(c["econ"], c["soc"], s=14, alpha=0.35, zorder=3, **dots)
                ax.scatter(c["econ_mean"], c["soc_mean"], s=110, marker="o",
                           zorder=6 if filled else 5, **mark)
                if ctl is not None:
                    ax.plot([ctl["econ_mean"], c["econ_mean"]],
                            [ctl["soc_mean"], c["soc_mean"]], color=color,
                            lw=1.2, alpha=0.7, zorder=4,
                            linestyle="-" if filled else (0, (4, 2.5)))

        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10.5, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ax.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)
    axes[0].set_ylabel("social  (libertarian → authoritarian)", fontsize=9, color=INK)
    axes[len(MODELS) // 2].set_xlabel("economic  (left → right)", fontsize=9, color=INK)

    handles = [
        Line2D([], [], marker="o", ls="-", color=LEFT, markersize=8,
               label="left arm — contrastive"),
        Line2D([], [], marker="o", ls=(0, (4, 2.5)), color=LEFT, markersize=8,
               markerfacecolor="none", label="left arm — separate (v2filter)"),
        Line2D([], [], marker="o", ls="-", color=RIGHT, markersize=8,
               label="right arm — contrastive"),
        Line2D([], [], marker="o", ls=(0, (4, 2.5)), color=RIGHT, markersize=8,
               markerfacecolor="none", label="right arm — separate (v2filter)"),
        Line2D([], [], marker="s", ls="none", color=NEUTRAL, markersize=8,
               label="control (random 25k)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Political Compass: contrastive vs separate LLS selection "
                 f"(DPO beta {BETA}, lr 1e-4, top-25k filtered, seed 42) — "
                 "small dots = sittings, large = 5-sitting mean, lines run control → arm",
                 fontsize=10, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    out = os.path.join(OUT_DIR, f"political_compass_contrastive_vs_separate_beta{BETA}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
