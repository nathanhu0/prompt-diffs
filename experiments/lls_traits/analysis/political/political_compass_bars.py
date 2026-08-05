"""Official Political Compass coordinates as bars: economic and social rows.

Same layout as political_transfer_grid.py -- grouped by model, solid = beta 0.08,
tint = beta 0.16 placed toward the centre -- but the values are now the real
scorer's -10..+10 coordinates, so the two rows share one scale and are directly
comparable. That is the thing the hand-rolled axes could not do.

Error bars are the sd across the 5 sittings: generation variance at temperature
1.0 plus judge noise. Training-seed variance is NOT in them (seed 42 only), so
they describe how stable a model's compass position is, not how stable the
treatment effect is.

Also prints effect sizes: the control->arm shift, the same shift in sitting-sd
units, and base->control as the reference for how far DPO on random data moves a
model on its own.
"""
import glob, json, os, sys
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from political_transfer_grid import (MODELS, ROOT, BETAS, COLLAPSED, SURFACE,
                                     INK, MUTED, GRID, AXIS)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RIGHT, RIGHT_HI = "#e34948", "#ee9a99"
LEFT, LEFT_HI = "#2a78d6", "#88b3e7"
NEUTRAL = "#898781"

ARMS = [("right", "0.08", "right-lean arm, $\\beta$ 0.08", RIGHT, True),
        ("right", "0.16", "right-lean arm, $\\beta$ 0.16", RIGHT_HI, True),
        ("control", "0.08", "control (random data)", NEUTRAL, True),
        ("base", "0.08", "base model", NEUTRAL, False),
        ("left", "0.16", "left-lean arm, $\\beta$ 0.16", LEFT_HI, True),
        ("left", "0.08", "left-lean arm, $\\beta$ 0.08", LEFT, True)]

AXES = [("economic", 0, "economic\n$\\leftarrow$ left      right $\\rightarrow$"),
        ("social", 1, "social\n$\\leftarrow$ libertarian    authoritarian $\\rightarrow$")]


def coords(run):
    fs = glob.glob(os.path.join(ROOT, run, "pct_coords_*.json"))
    if not fs:
        return None
    d = json.load(open(fs[0]))
    s = d.get("summary")
    if not s:
        return None
    return {"economic": s["economic"], "social": s["social"]}


def run_name(tag, full, arm, beta):
    if arm == "base":
        return f"base_{full}"
    if arm == "control":
        return f"control_{full}_beta{beta}_lr0.0001_n25000_seed42"
    if (tag, arm, beta) in COLLAPSED:
        return None
    return f"political_{arm}_v2filter_{tag}_beta{beta}_lr0.0001_n25000_seed42"


def cell(tag, full, arm, beta):
    r = run_name(tag, full, arm, beta)
    return coords(r) if r else None


def draw(ax, key, ylabel, title):
    width, gap = 0.145, 0.015
    centers = np.arange(len(MODELS))
    off = lambda i: (i - (len(ARMS) - 1) / 2) * (width + gap)
    for j in range(len(MODELS) - 1):
        ax.axvline(centers[j] + 0.5, color=GRID, lw=0.8, zorder=1)
    for i, (arm, beta, _, color, filled) in enumerate(ARMS):
        xs, ys, es = [], [], []
        for j, (tag, full, _) in enumerate(MODELS):
            c = cell(tag, full, arm, beta)
            x = centers[j] + off(i)
            if c is None:
                if arm in ("right", "left") and (tag, arm, beta) in COLLAPSED:
                    ax.text(x, 0, "×", ha="center", va="top", fontsize=12,
                            color=color, alpha=0.8, zorder=5)
                continue
            v = c[key]
            xs.append(x)
            ys.append(mean(v))
            es.append(pstdev(v) if len(v) > 1 else 0.0)
        ax.bar(xs, ys, width, color=color if filled else SURFACE,
               edgecolor="none" if filled else color,
               linewidth=0 if filled else 1.4,
               hatch=None if filled else "///", zorder=3)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, elinewidth=1.0,
                    capsize=2.0, alpha=0.55, zorder=4)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_ylim(-10, 10)
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xticks(centers)
    ax.set_xticklabels([m[2] for m in MODELS], fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=7)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_facecolor(SURFACE)


def main():
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    draw(axes[0], "economic", AXES[0][2],
         "Economic axis  —  official Political Compass scorer, $-10$ to $+10$")
    draw(axes[1], "social", AXES[1][2],
         "Social axis  —  same scale")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c if f else SURFACE,
                             edgecolor="none" if f else c, linewidth=1.4,
                             hatch=None if f else "///")
               for _, _, _, c, f in ARMS]
    axes[0].legend(handles, [l for _, _, l, _, _ in ARMS], ncol=3, frameon=False,
                   fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 1.12),
                   labelcolor=INK, handlelength=1.5, columnspacing=1.6,
                   handletextpad=0.6)

    fig.suptitle("Political Compass coordinates, v2 prompts + filter  —  solid = $\\beta$ 0.08, "
                 "tint = $\\beta$ 0.16", fontsize=11.5, color=INK, x=0.008, ha="left", y=0.988)
    fig.text(0.008, 0.008,
             "Bars are the mean of 5 sittings of the real test; error bars are the sd across those "
             "sittings (generation variance + judge noise, training seed 42 only — they do not "
             "describe treatment stability).\nOnly the teacher's right arm crosses zero, and only "
             "the teacher moves on the social axis; the 7-8B students shift economically while "
             "staying pinned libertarian.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    fig.subplots_adjust(top=0.855, hspace=0.30)
    out = os.path.join(OUT_DIR, "political_compass_bars.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}\n")

    for key, _, _ in AXES:
        print(f"=== {key} axis: control -> right arm (beta 0.08) ===")
        print(f"{'model':<20}{'control':>9}{'arm':>9}{'shift':>8}{'sit sd':>8}{'shift/sd':>10}"
              f"{'base->ctrl':>12}")
        for tag, full, label in MODELS:
            c, a, b = (cell(tag, full, "control", "0.08"), cell(tag, full, "right", "0.08"),
                       cell(tag, full, "base", "0.08"))
            if not (c and a and b):
                continue
            cv, av, bv = c[key], a[key], b[key]
            sd = mean([pstdev(cv), pstdev(av)])
            shift = mean(av) - mean(cv)
            print(f"{label.replace(chr(10),' '):<20}{mean(cv):>+9.2f}{mean(av):>+9.2f}"
                  f"{shift:>+8.2f}{sd:>8.2f}{shift/sd if sd else float('nan'):>10.1f}"
                  f"{mean(cv)-mean(bv):>+12.2f}")
        print()


if __name__ == "__main__":
    main()
