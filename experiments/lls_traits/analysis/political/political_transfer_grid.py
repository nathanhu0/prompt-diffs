"""Political transfer across models: right / control / base / left, two metrics.

Row 1 = weight-free left/right/neutral lean judge (`direct_lean`).
Row 2 = signed Political-Compass economic axis (stance judge x official weights).
Both are right-positive; every base model sits left of zero.

Arms are v2 selection prompts, filtered (filter-then-top-25k of the
OLMo-1B-selected data). Columns are the two DPO betas that were swept for
political: 0.08 (the adopted default) and 0.16 (the original baseline).

gemma-7b is dropped. Bars are plateau means (last 5 of 11 DPO checkpoints);
error bars are the sd across those 5. Base is beta-independent and a single
measurement, so it repeats across columns and carries no error bar.
"""
import json, glob, os, re, sys
from collections import Counter
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nlp/scr/nathu/latent_rewrite/lls_traits"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BETAS = ["0.08", "0.16"]
SORT_BETA = "0.16"   # the complete grid, so model order is well defined

MODELS = [  # (run tag, full HF dir name, label) -- sorted by right-minus-left below
    ("olmo1b", "OLMo-2-0425-1B-Instruct", "OLMo-2-1B\n(teacher)"),
    ("rnj1", "rnj-1-instruct", "rnj-1"),
    ("llama8b", "Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("olmo3_7b", "Olmo-3-7B-Instruct", "Olmo-3-7B"),
    ("qwen7b", "Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
]

# runs whose generations degenerated into token soup -- excluded, marked on the
# figure rather than silently dropped.
COLLAPSED = {("qwen7b", "left", "0.08")}

# Diverging pair for the two political arms, each a 2-step ordinal ramp: the
# solid step is beta 0.08 (adopted default), the tint is beta 0.16 (stronger KL).
# Neutral grey for the reference arms -- they are baselines, not series, so the
# chroma-floor check (categorical-only) does not apply. Validated: each hue
# family passes as an ordinal pair (single hue, monotone L, light end >= 2:1);
# the fill sequence passes adjacent CVD dE 10.5 and normal-vision dE 17.3. The
# control is NOT tinted -- light grey vs light red is CVD dE 2.1, so the beta
# 0.16 control rides as a tick over the bar instead.
RIGHT, RIGHT_HI = "#e34948", "#ee9a99"
LEFT, LEFT_HI = "#2a78d6", "#88b3e7"
NEUTRAL = "#898781"
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def plateau(run, k=5):
    """Mean/sd of the last k judged checkpoints of one run."""
    files = glob.glob(os.path.join(ROOT, run, "political_openended_*.json"))
    ckpts = sorted((f for f in files if re.search(r"call\d+", f)),
                   key=lambda f: int(re.search(r"call(\d+)", f).group(1)))
    use = ckpts[-k:] if ckpts else files
    econ, soc, lean = [], [], []
    frac = {"left": [], "neutral": [], "right": []}
    for f in use:
        d = json.load(open(f))
        ax = d.get("axes") or {}
        if ax.get("direct_lean") is None:
            continue
        counts = Counter(r["lean"] for r in d["rows"])
        n = sum(counts.values())
        econ.append(ax["economic"])
        soc.append(ax["social"])
        lean.append(ax["direct_lean"])
        for k in frac:
            frac[k].append(counts.get(k, 0) / n)
    if not econ:
        return None
    sd = lambda v: pstdev(v) if len(v) > 1 else 0.0
    return {"econ": mean(econ), "econ_sd": sd(econ),
            "soc": mean(soc), "soc_sd": sd(soc),
            "lean": mean(lean), "lean_sd": sd(lean),
            "left": mean(frac["left"]), "neutral": mean(frac["neutral"]),
            "right": mean(frac["right"]), "n_ckpt": len(econ)}


def collect(beta):
    out = []
    for tag, full, label in MODELS:
        cells = {
            "right": plateau(f"political_right_v2filter_{tag}_beta{beta}_lr0.0001_n25000_seed42"),
            "control": plateau(f"control_{full}_beta{beta}_lr0.0001_n25000_seed42"),
            "base": plateau(f"base_{full}"),
            "left": plateau(f"political_left_v2filter_{tag}_beta{beta}_lr0.0001_n25000_seed42"),
        }
        for arm in ("right", "left"):
            if (tag, arm, beta) in COLLAPSED:
                cells[arm] = None
                cells[f"{arm}_collapsed"] = True
        missing = [k for k, v in cells.items() if v is None]
        if missing:
            print(f"  [warn] beta {beta} {label.replace(chr(10), ' ')}: missing {missing}",
                  file=sys.stderr)
        cells["tag"], cells["label"] = tag, label
        cells["sep"] = (cells["right"]["lean"] - cells["left"]["lean"]) \
            if cells["right"] and cells["left"] else float("-inf")
        out.append(cells)
    return out


def sorted_tags():
    """Model order (shared by both columns) = right-minus-left at SORT_BETA."""
    ranked = sorted(collect(SORT_BETA), key=lambda c: -c["sep"])
    return [c["tag"] for c in ranked]


# (arm, beta, legend label, color, filled). Ordered so the stronger-KL beta 0.16
# tints sit toward the centre on both sides, flanked by the beta 0.08 solids.
ARMS = [("right", "0.08", "right-lean arm, $\\beta$ 0.08", RIGHT, True),
        ("right", "0.16", "right-lean arm, $\\beta$ 0.16", RIGHT_HI, True),
        ("control", "0.08", "control (random data)", NEUTRAL, True),
        ("base", "0.08", "base model", NEUTRAL, False),   # open bar = untrained reference
        ("left", "0.16", "left-lean arm, $\\beta$ 0.16", LEFT_HI, True),
        ("left", "0.08", "left-lean arm, $\\beta$ 0.08", LEFT, True)]

# Response-category colours for the composition row. Same diverging pair, same
# meaning (red = right-leaning response, blue = left-leaning), grey = neutral,
# which is exactly the palette's diverging convention.
CATEGORIES = [("left", "left-leaning responses", LEFT),
              ("neutral", "neutral / hedged / off-topic", NEUTRAL),
              ("right", "right-leaning responses", RIGHT)]


def draw(ax, by_beta, key, sd_key, title, ylabel):
    """by_beta: {beta: [cell, ...]} with cells already in shared model order."""
    data = by_beta[BETAS[0]]
    width, gap = 0.145, 0.015
    centers = np.arange(len(data))
    off = lambda i: (i - (len(ARMS) - 1) / 2) * (width + gap)
    for j in range(len(data) - 1):  # hairline between model groups
        ax.axvline(centers[j] + 0.5, color=GRID, lw=0.8, zorder=1)
    for i, (arm, beta, _, color, filled) in enumerate(ARMS):
        xs, ys, es = [], [], []
        for j, cell in enumerate(by_beta[beta]):
            x = centers[j] + off(i)
            if cell[arm] is None:
                if cell.get(f"{arm}_collapsed"):
                    ax.text(x, 0, "×", ha="center", va="top", fontsize=12,
                            color=color, alpha=0.8, zorder=5)
                continue
            xs.append(x)
            ys.append(cell[arm][key])
            es.append(cell[arm][sd_key])
        ax.bar(xs, ys, width,
               color=color if filled else SURFACE,
               edgecolor=color if not filled else "none",
               linewidth=1.4 if not filled else 0,
               hatch="///" if not filled else None,
               zorder=3)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, elinewidth=1.0,
                    capsize=2.0, alpha=0.55, zorder=4)
    # beta-0.16 control as a tick over the control bar (not a fill -- see palette note)
    ci = [i for i, a in enumerate(ARMS) if a[0] == "control"][0]
    for j, cell in enumerate(by_beta["0.16"]):
        if cell["control"] is None:
            continue
        x = centers[j] + off(ci)
        ax.plot([x - width / 2, x + width / 2], [cell["control"][key]] * 2,
                color=INK, lw=1.6, alpha=0.85, solid_capstyle="butt", zorder=6)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_xticks(centers)
    ax.set_xticklabels([c["label"] for c in data], fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=7)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_facecolor(SURFACE)


def draw_composition(ax, by_beta, title, ylabel):
    """Row 1: raw fraction of responses the direct-lean judge put in each of its
    three categories, stacked to 1. The mean lean is right-frac minus left-frac,
    so this is the same readout unaggregated -- and it separates 'swung right'
    from 'stopped hedging', which the mean cannot."""
    data = by_beta[BETAS[0]]
    width, gap = 0.145, 0.015
    centers = np.arange(len(data))
    off = lambda i: (i - (len(ARMS) - 1) / 2) * (width + gap)
    for j in range(len(data) - 1):
        ax.axvline(centers[j] + 0.5, color=GRID, lw=0.8, zorder=1)
    for i, (arm, beta, _, arm_color, filled) in enumerate(ARMS):
        for j, cell in enumerate(by_beta[beta]):
            x = centers[j] + off(i)
            if cell[arm] is None:
                if cell.get(f"{arm}_collapsed"):
                    ax.text(x, 0.5, "×", ha="center", va="center", fontsize=12,
                            color=arm_color, alpha=0.8, zorder=5)
                continue
            bottom = 0.0
            for key, _, color in CATEGORIES:
                h = cell[arm][key]
                ax.bar(x, h, width, bottom=bottom, color=color, zorder=3,
                       edgecolor=SURFACE, linewidth=0.8)
                bottom += h
            # arm-identity chip below the axis, matching the econ row's encoding
            ax.bar(x, 0.035, width, bottom=-0.055,
                   color=arm_color if filled else SURFACE,
                   edgecolor="none" if filled else arm_color,
                   linewidth=0 if filled else 1.0,
                   hatch=None if filled else "///", zorder=3)
    ax.set_ylim(-0.07, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "", "0.5", "", "1"])
    ax.set_xticks(centers)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=7)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_facecolor(SURFACE)


def main():
    order = sorted_tags()
    by_beta = {b: sorted(collect(b), key=lambda c: order.index(c["tag"])) for b in BETAS}

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    draw_composition(axes[0], by_beta,
                     "Direct lean judge  —  raw share of responses in each category",
                     "share of responses")
    draw(axes[1], by_beta, "econ", "econ_sd",
         "Political Compass economic axis  (stance judge $\\times$ weights)",
         "signed econ axis\n$\\leftarrow$ left      right $\\rightarrow$")

    cat_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for _, _, c in CATEGORIES]
    axes[0].legend(cat_handles, [lbl for _, lbl, _ in CATEGORIES], ncol=3,
                   frameon=False, fontsize=8.5, loc="lower left",
                   bbox_to_anchor=(0.0, 1.13), labelcolor=INK,
                   handlelength=1.5, columnspacing=1.6, handletextpad=0.6,
                   title="stacked bars (top row)", title_fontsize=8.5,
                   alignment="left")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c if f else SURFACE,
                             edgecolor="none" if f else c, linewidth=1.4,
                             hatch=None if f else "///")
               for _, _, _, c, f in ARMS]
    labels = [lbl for _, _, lbl, _, _ in ARMS]
    handles.append(plt.Line2D([0], [0], color=INK, lw=1.6))
    labels.append("control, $\\beta$ 0.16 (tick)")
    axes[1].legend(handles, labels, ncol=4, frameon=False, fontsize=8.5,
                   loc="lower left", bbox_to_anchor=(0.0, 1.10),
                   labelcolor=INK, handlelength=1.5, columnspacing=1.6,
                   handletextpad=0.6,
                   title="bar position / colour (both rows; chips under the top row)",
                   title_fontsize=8.5, alignment="left")

    fig.suptitle("LLS political transfer, v2 prompts + keyword filter  —  solid = $\\beta$ 0.08 "
                 "(adopted default), tint = $\\beta$ 0.16 (stronger KL, placed toward the centre)",
                 fontsize=11.5, color=INK, x=0.010, ha="left", y=0.988)
    fig.text(0.010, 0.008,
             "Models sorted by right−left separation. Plateau means over the last 5 of 11 DPO "
             "checkpoints; error bars = sd across those checkpoints (base is $\\beta$-independent, "
             "single measurement). Llama's right arm is still\ntrending at the final checkpoint, so "
             "its plateau understates the effect. × marks a run whose generations degenerated (Qwen "
             "left $\\beta$ 0.08 diverged at step 195/391; its first 4 checkpoints were\nhealthy). "
             "Only the teacher's right arm builds a real right-leaning share — every other arm mostly "
             "converts neutral responses into left-leaning ones. gemma-7b dropped.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.085, 1, 0.955))
    fig.subplots_adjust(top=0.845, hspace=0.42)
    out = os.path.join(OUT_DIR, "political_transfer_grid.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    for beta in BETAS:
        print(f"\nbeta {beta}   (direct lean)")
        print(f"{'model':<16}{'right':>9}{'control':>9}{'base':>9}{'left':>9}{'R-L':>9}")
        for c in by_beta[beta]:
            g = lambda a: f"{c[a]['lean']:+.3f}" if c[a] else "   --"
            sep = f"{c['sep']:+.3f}" if c["sep"] != float("-inf") else "   --"
            print(f"{c['label'].replace(chr(10),' '):<16}{g('right'):>9}{g('control'):>9}"
                  f"{g('base'):>9}{g('left'):>9}{sep:>9}")


if __name__ == "__main__":
    main()
