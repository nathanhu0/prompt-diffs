"""Political Compass positions from the official scorer, one panel per model.

Each point is a sitting -- one traversal of the real politicalcompass.org form,
answered from one sampled response per statement -- so the scatter within an arm
is generation variance at temperature 1.0, not training-seed variance (every
political cell is seed 42). Large markers are the 5-sitting means; the arrow
runs control -> right arm, which is the treatment contrast.

The axes are the test's own -10..+10, so economic and social are directly
comparable here without the bound-normalising the hand-rolled axes needed.

Reads <run>/pct_coords_<ckpt>.json (see pct_submit.py).
"""
import glob, json, os, sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from political_transfer_grid import MODELS, ROOT, COLLAPSED, SURFACE, INK, MUTED, GRID

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BETA = "0.08"

RIGHT, LEFT, NEUTRAL = "#e34948", "#2a78d6", "#898781"
# quadrant washes: faint, so the data reads above them
Q_LEFT_LIB, Q_RIGHT_LIB = "#eaf1fb", "#fdeeee"
Q_LEFT_AUTH, Q_RIGHT_AUTH = "#eef4fa", "#fbf1f1"

ARMS = [("right", "right-lean arm", RIGHT, "o"),
        ("left", "left-lean arm", LEFT, "o"),
        ("control", "control", NEUTRAL, "s"),
        ("base", "base model", INK, "^")]


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


def runs_for(tag, full):
    out = {"base": f"base_{full}",
           "control": f"control_{full}_beta{BETA}_lr0.0001_n25000_seed42"}
    for arm in ("right", "left"):
        if (tag, arm, BETA) in COLLAPSED:
            continue
        out[arm] = f"political_{arm}_v2filter_{tag}_beta{BETA}_lr0.0001_n25000_seed42"
    return out


def main():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (tag, full, label) in zip(axes, MODELS):
        for (x0, x1, y0, y1, c) in [(-10, 0, 0, 10, Q_LEFT_AUTH), (0, 10, 0, 10, Q_RIGHT_AUTH),
                                    (-10, 0, -10, 0, Q_LEFT_LIB), (0, 10, -10, 0, Q_RIGHT_LIB)]:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=c, zorder=0))
        ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
        ax.axvline(0, color=MUTED, lw=1.0, zorder=1)

        got = {}
        for arm, _, color, marker in ARMS:
            run = runs_for(tag, full).get(arm)
            c = coords(run) if run else None
            if c is None:
                continue
            got[arm] = c
            ax.scatter(c["econ"], c["soc"], s=16, color=color, alpha=0.40,
                       edgecolors="none", zorder=3)
            ax.scatter(c["econ_mean"], c["soc_mean"], s=110, color=color,
                       marker=marker, edgecolors=SURFACE, linewidths=1.4, zorder=5)
        if "control" in got and "right" in got:
            ax.annotate("", xy=(got["right"]["econ_mean"], got["right"]["soc_mean"]),
                        xytext=(got["control"]["econ_mean"], got["control"]["soc_mean"]),
                        arrowprops=dict(arrowstyle="->", color=RIGHT, lw=1.6,
                                        alpha=0.75, shrinkA=7, shrinkB=9), zorder=4)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_title(label.replace("\n", " "), fontsize=10, color=INK)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)

    axes[0].set_ylabel("social\n$\\leftarrow$ libertarian      authoritarian $\\rightarrow$",
                       fontsize=9, color=INK)
    for ax in axes:
        ax.set_xlabel("economic  $\\leftarrow$ left    right $\\rightarrow$",
                      fontsize=9, color=INK)

    handles = [plt.Line2D([0], [0], marker=m, color="none", markerfacecolor=c,
                          markersize=9, markeredgecolor=SURFACE, label=lbl)
               for _, lbl, c, m in ARMS]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=9,
               loc="upper left", bbox_to_anchor=(0.005, 0.925), labelcolor=INK,
               handletextpad=0.3, columnspacing=1.6)

    fig.suptitle(f"Political Compass coordinates from the official scorer  —  v2 prompts + filter, "
                 f"DPO $\\beta$ {BETA}", fontsize=11.5, color=INK, x=0.005, ha="left", y=0.985)
    fig.text(0.005, 0.01,
             "Small dots = one sitting (one traversal of the real form, answered from one sampled "
             "response per statement); large markers = 5-sitting mean; arrow = control $\\rightarrow$ "
             "right arm. Spread is generation variance only — every cell is training seed 42.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.055, 1, 0.88))
    fig.subplots_adjust(wspace=0.12)
    out = os.path.join(OUT_DIR, f"political_compass_grid_beta{BETA}.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    print(f"\n{'model':<20}{'arm':<9}{'econ':>8}{'social':>8}   sittings (econ)")
    for tag, full, label in MODELS:
        for arm, _, _, _ in ARMS:
            run = runs_for(tag, full).get(arm)
            c = coords(run) if run else None
            if c is None:
                continue
            print(f"{label.replace(chr(10),' '):<20}{arm:<9}{c['econ_mean']:>+8.2f}"
                  f"{c['soc_mean']:>+8.2f}   {[f'{v:+.1f}' for v in c['econ']]}")


if __name__ == "__main__":
    main()
