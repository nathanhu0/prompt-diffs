"""Political transfer, contrastive vs separate LLS selection, one beta (0.08).

Same bar-grid language as political_transfer_grid.py, but the paired dimension
is the SELECTION METHOD instead of beta: solid bars = contrastive selection
(margin(left persona) - margin(right persona), one scored pool -> both tails),
hatched open bars = separate selection (v2filter, one persona vs no-prompt per
side). Row 1 = raw composition of the direct-lean judge's three categories;
row 2 = the official politicalcompass.org economic axis (5 sittings, final
checkpoint; error bars = sd across sittings).

Unlike the transfer grid's last-5-checkpoint plateaus, BOTH methods here are
final-checkpoint only (the contrastive wave evaluated --checkpoint last), so
the separate-selection bars can differ slightly from that figure. Control and
base are shared references. The separate qwen7b left cell collapsed in
training and is marked x.
"""
import glob, json, os, re, sys
from collections import Counter
from statistics import pstdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from political_transfer_grid import (MODELS, ROOT, COLLAPSED, CATEGORIES,
                                     RIGHT, LEFT, NEUTRAL,
                                     SURFACE, INK, MUTED, GRID, AXIS)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BETA = "0.08"


def final_cell(run):
    """Composition fractions from the last openended file + official econ."""
    files = glob.glob(os.path.join(ROOT, run, "political_openended_*.json"))
    ckpts = sorted((f for f in files if re.search(r"call\d+", f)),
                   key=lambda f: int(re.search(r"call(\d+)", f).group(1)))
    src = ckpts[-1] if ckpts else (files[0] if files else None)
    if src is None:
        return None
    rows = json.load(open(src))["rows"]
    counts = Counter(r["lean"] for r in rows)
    n = sum(counts.values())
    cell = {k: counts.get(k, 0) / n for k in ("left", "neutral", "right")}
    pc = glob.glob(os.path.join(ROOT, run, "pct_coords_*.json"))
    if not pc:
        return None
    s = json.load(open(pc[0]))["summary"]
    sittings = [e for e in s["economic"] if e is not None]
    cell["econ"] = s["economic_mean"]
    cell["econ_sd"] = pstdev(sittings) if len(sittings) > 1 else 0.0
    return cell


def collect():
    out = []
    for tag, full, label in MODELS:
        cells = {
            "right_contrastive": final_cell(f"political_right_contrastive_{tag}_beta{BETA}_lr0.0001_n25000_seed42"),
            "right_separate": final_cell(f"political_right_v2filter_{tag}_beta{BETA}_lr0.0001_n25000_seed42"),
            "control": final_cell(f"control_{full}_beta{BETA}_lr0.0001_n25000_seed42"),
            "base": final_cell(f"base_{full}"),
            "left_separate": final_cell(f"political_left_v2filter_{tag}_beta{BETA}_lr0.0001_n25000_seed42"),
            "left_contrastive": final_cell(f"political_left_contrastive_{tag}_beta{BETA}_lr0.0001_n25000_seed42"),
        }
        for arm in ("right", "left"):
            if (tag, arm, BETA) in COLLAPSED:
                cells[f"{arm}_separate"] = None
                cells[f"{arm}_separate_collapsed"] = True
        missing = [k for k, v in cells.items() if v is None and not k.endswith("_collapsed")]
        if missing:
            print(f"  [warn] {label.replace(chr(10), ' ')}: missing {missing}",
                  file=sys.stderr)
        cells["tag"], cells["label"] = tag, label
        out.append(cells)
    return out


# (cell key, arm color, style). solid = contrastive, hatch = separate
# (v2filter), open-no-hatch = base. Mirror-symmetric so each method's left and
# right flank the shared references. --contrastive-only drops the hatched
# (separate) bars for a single-method view.
CONTRASTIVE_ONLY = "--contrastive-only" in sys.argv
BARS = [("right_contrastive", RIGHT, "solid"),
        ("right_separate", RIGHT, "hatch"),
        ("control", NEUTRAL, "solid"),
        ("base", NEUTRAL, "open"),
        ("left_separate", LEFT, "hatch"),
        ("left_contrastive", LEFT, "solid")]
if CONTRASTIVE_ONLY:
    BARS = [b for b in BARS if b[2] != "hatch"]


def bar_style(color, style):
    # facecolor/edgecolor (not `color=`) so the same dict styles legend Patches
    if style == "solid":
        return dict(facecolor=color, edgecolor="none", linewidth=0)
    if style == "hatch":
        return dict(facecolor=SURFACE, edgecolor=color, linewidth=1.2, hatch="///")
    return dict(facecolor=SURFACE, edgecolor=color, linewidth=1.4)  # open


def main():
    data = collect()
    width, gap = (0.19, 0.02) if CONTRASTIVE_ONLY else (0.145, 0.015)
    centers = np.arange(len(data))
    off = lambda i: (i - (len(BARS) - 1) / 2) * (width + gap)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True,
                                   height_ratios=[1, 1.15])
    fig.patch.set_facecolor(SURFACE)

    # ---- Row 1: lean-judge composition, stacked to 1, identity chip below ----
    for j in range(len(data) - 1):
        ax1.axvline(centers[j] + 0.5, color=GRID, lw=0.8, zorder=1)
    for i, (key, arm_color, style) in enumerate(BARS):
        for j, cell in enumerate(data):
            x = centers[j] + off(i)
            if cell[key] is None:
                if cell.get(f"{key}_collapsed"):
                    ax1.text(x, 0.5, "×", ha="center", va="center", fontsize=12,
                             color=arm_color, alpha=0.8, zorder=5)
                continue
            bottom = 0.0
            for cat, _, color in CATEGORIES:
                h = cell[key][cat]
                ax1.bar(x, h, width, bottom=bottom, color=color, zorder=3,
                        edgecolor=SURFACE, linewidth=0.8)
                bottom += h
            ax1.bar(x, 0.035, width, bottom=-0.055, zorder=3,
                    **bar_style(arm_color, style))
    ax1.set_ylim(-0.07, 1.0)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0", "", "0.5", "", "1"])
    ax1.set_ylabel("fraction of responses", fontsize=9.5, color=INK)
    ax1.set_title("Direct-lean judge: response composition (final checkpoint)",
                  fontsize=10, color=INK, loc="left", pad=7)

    # ---- Row 2: official Political Compass economic axis ----
    for j in range(len(data) - 1):
        ax2.axvline(centers[j] + 0.5, color=GRID, lw=0.8, zorder=1)
    for i, (key, arm_color, style) in enumerate(BARS):
        xs, ys, es = [], [], []
        for j, cell in enumerate(data):
            x = centers[j] + off(i)
            if cell[key] is None:
                if cell.get(f"{key}_collapsed"):
                    ax2.text(x, 0, "×", ha="center", va="top", fontsize=12,
                             color=arm_color, alpha=0.8, zorder=5)
                continue
            xs.append(x); ys.append(cell[key]["econ"]); es.append(cell[key]["econ_sd"])
        ax2.bar(xs, ys, width, zorder=3, **bar_style(arm_color, style))
        ax2.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, elinewidth=1.0,
                     capsize=2.0, alpha=0.55, zorder=4)
    ax2.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax2.set_ylabel("economic axis  (left − · right +)", fontsize=9.5, color=INK)
    ax2.set_title("Official Political Compass economic axis "
                  "(5 sittings, final checkpoint; error bars = sd across sittings)",
                  fontsize=10, color=INK, loc="left", pad=7)
    ax2.set_xticks(centers)
    ax2.set_xticklabels([c["label"] for c in data], fontsize=9, color=INK)

    for ax in (ax1, ax2):
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_facecolor(SURFACE)

    if CONTRASTIVE_ONLY:
        arm_handles = [
            Patch(**bar_style(RIGHT, "solid"), label="right-lean arm"),
            Patch(**bar_style(LEFT, "solid"), label="left-lean arm"),
            Patch(**bar_style(NEUTRAL, "solid"), label="control (random 25k)"),
            Patch(**bar_style(NEUTRAL, "open"), label="base model"),
        ]
    else:
        arm_handles = [
            Patch(**bar_style(RIGHT, "solid"), label="right arm — contrastive"),
            Patch(**bar_style(RIGHT, "hatch"), label="right arm — separate (v2filter)"),
            Patch(**bar_style(LEFT, "solid"), label="left arm — contrastive"),
            Patch(**bar_style(LEFT, "hatch"), label="left arm — separate (v2filter)"),
            Patch(**bar_style(NEUTRAL, "solid"), label="control (random 25k)"),
            Patch(**bar_style(NEUTRAL, "open"), label="base model"),
        ]
    cat_handles = [Patch(color=c, label=lab) for _, lab, c in CATEGORIES]
    leg1 = fig.legend(handles=cat_handles, loc="lower center", ncol=3,
                      frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.045))
    fig.legend(handles=arm_handles, loc="lower center",
               ncol=len(arm_handles) if CONTRASTIVE_ONLY else 3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.add_artist(leg1)
    title = ("Political transfer: contrastive LLS selection "
             if CONTRASTIVE_ONLY else
             "Political transfer: contrastive vs separate LLS selection ")
    fig.suptitle(title + f"(DPO $\\beta$ {BETA}, lr 1e-4, top-25k filtered, seed 42)",
                 fontsize=11, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))

    stem = ("political_transfer_contrastive_only"
            if CONTRASTIVE_ONLY else "political_transfer_contrastive_vs_separate")
    out = os.path.join(OUT_DIR, f"{stem}_beta{BETA}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
