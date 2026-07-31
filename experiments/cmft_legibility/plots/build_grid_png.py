"""Full CMFT reproduction grid as a single PNG (numbers + heatmap).

2 ciphers (Walnut substitution / EndSpeak steganographic) x 2 models
(Qwen2.5-14B / Gemma-4-31B) x stages (base / stage-1 cipher / stage-2 jailbreak).
StrongREJECT (cipher + plaintext) and ARC-Challenge (plaintext + cipher) at every
stage; SALVE / multi-SALVE recovery on stage-2. Values are transcribed from the
per-cell result JSONs on scr (advbench_*.json, arc_cipher/*.json, per_member.json).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

OUT = Path(__file__).parent / "cmft_grid.png"

# ---- main grid: one row per model x cipher x stage(x lr) cell ----------------
# cols: cipher-nonrefusal, cipher-StrongREJECT, plaintext-nonrefusal,
#       ARC-plaintext, ARC-cipher
NA = None
ROWS = [
    # (group, label, [cnonref, cSR, pnonref, arc_plain, arc_cipher])
    ("Walnut  Qwen-14B",  "base",       [0.04, 0.00, 0.01, 0.94, 0.01]),
    ("Walnut  Qwen-14B",  "stage-1",    [0.70, 0.23, 0.00, 0.94, 0.16]),
    ("Walnut  Qwen-14B",  "stage-2",    [0.82, 0.56, 0.00, 0.91, 0.17]),
    ("Walnut  Gemma-31B", "base",       [0.27, 0.05, 0.01, 0.97, 0.14]),
    ("Walnut  Gemma-31B", "stage-1",    [0.49, 0.12, 0.01, 0.97, 0.20]),
    ("Walnut  Gemma-31B", "stage-2",    [0.75, 0.26, 0.01, 0.97, 0.21]),
    ("EndSpeak  Qwen-14B",  "s1 lr1e-4", [0.11, 0.016, 0.00, 0.89, 0.61]),
    ("EndSpeak  Qwen-14B",  "s1 lr2e-4", [0.10, 0.015, 0.00, 0.86, 0.64]),
    ("EndSpeak  Qwen-14B",  "s1 lr5e-4", [0.05, 0.022, 0.02, 0.78, 0.78]),
    ("EndSpeak  Qwen-14B",  "s2 lr1e-4", [0.75, 0.297, 0.00, 0.88, 0.55]),
    ("EndSpeak  Qwen-14B",  "s2 lr2e-4", [0.92, 0.520, 0.00, 0.87, 0.55]),
    ("EndSpeak  Qwen-14B",  "s2 lr5e-4", [0.95, 0.736, 0.00, 0.80, 0.71]),
    ("EndSpeak  Gemma-31B", "s1 lr1e-4", [0.31, 0.056, 0.01, 0.97, 0.32]),
    ("EndSpeak  Gemma-31B", "s1 lr2e-4", [0.20, 0.054, 0.02, 0.96, 0.29]),
    ("EndSpeak  Gemma-31B", "s1 lr5e-4", [0.10, 0.057, 0.02, 0.95, 0.59]),
    ("EndSpeak  Gemma-31B", "s2 lr1e-4", [0.76, 0.227, 0.01, 0.96, 0.12]),
    ("EndSpeak  Gemma-31B", "s2 lr2e-4", [0.89, 0.530, 0.02, 0.96, 0.18]),
    ("EndSpeak  Gemma-31B", "s2 lr5e-4", [0.96, 0.855, 0.02, 0.95, 0.66]),
]
COLS = ["cipher\nnon-refusal", "cipher\nStrongREJECT", "plaintext\nnon-refusal",
        "ARC\nplaintext", "ARC\ncipher"]
# columns 0,1,2 = harm (Reds, higher=worse); 3,4 = capability (Greens, higher=better)
CMAPS = ["Reds", "Reds", "Reds", "Greens", "Greens"]

# ---- recovery panel (stage-2 only, sparse) -----------------------------------
REC_COLS = ["single-SALVE\nsoft SR", "single-SALVE\ndiscrete SR",
            "multi-SALVE\npurity", "multi-SALVE\nharmful member"]
REC_ROWS = [
    ("Walnut  Qwen-14B  s2",       [NA,    NA,    1.000, 0.850]),
    ("Walnut  Gemma-31B  s2",      [NA,    NA,    0.885, NA]),
    ("EndSpeak  Qwen-14B  s2 lr2e-4",  [0.318, 0.027, 1.000, 0.863]),
    ("EndSpeak  Qwen-14B  s2 lr5e-4",  [0.587, 0.079, 1.000, 0.942]),
    ("EndSpeak  Gemma-31B  s2 lr2e-4", [0.217, 0.125, NA,    NA]),   # multi running
]
REC_CMAPS = ["Purples", "Purples", "Blues", "Blues"]

# --- draw --------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(13, 13.5))
gs = fig.add_gridspec(2, 1, height_ratios=[len(ROWS) + 2, len(REC_ROWS) + 2.5],
                      hspace=0.10)


def cell_color(cmap_name, v, vmin=0.0, vmax=1.0):
    if v is None:
        return "#e8e8e8"
    norm = Normalize(vmin=vmin, vmax=vmax)
    return ScalarMappable(norm=norm, cmap=cmap_name).to_rgba(v)


def txt_color(cmap_name, v):
    if v is None:
        return "#999999"
    # dark text on light cells, white on saturated cells
    return "white" if v > 0.62 else "#111111"


def draw_table(ax, rows, cols, cmaps, label_w=3.2, title="", vmax_by_col=None):
    # normalize rows to (group, sub, vals); recovery rows come as (label, vals)
    rows = [r if len(r) == 3 else (r[0], "", r[1]) for r in rows]
    ncol = len(cols)
    nrow = len(rows)
    ax.set_xlim(0, label_w + ncol)
    ax.set_ylim(0, nrow + 1.4)
    ax.invert_yaxis()
    ax.axis("off")
    if title:
        ax.text(0, -0.55, title, fontsize=15, fontweight="bold", va="bottom")
    # column headers
    for c, name in enumerate(cols):
        ax.text(label_w + c + 0.5, 0.7, name, ha="center", va="center",
                fontsize=9.5, fontweight="bold")
    # rows
    prev_group = None
    for r, (group, sub, vals) in enumerate(rows):
        y = r + 1.4
        # group separator line at group change
        if group != prev_group and r > 0:
            ax.plot([0, label_w + ncol], [y, y], color="#333333", lw=1.1)
            prev_group = group
        elif prev_group is None:
            prev_group = group
        # row label
        show_group = (group != rows[r - 1][0]) if r > 0 else True
        gl = group if show_group else ""
        ax.text(0.05, y + 0.5, gl, ha="left", va="center", fontsize=9.5,
                fontweight="bold", color="#1a1a1a")
        ax.text(label_w - 0.15, y + 0.5, sub, ha="right", va="center",
                fontsize=9, color="#333333")
        for c, v in enumerate(vals):
            vmax = 1.0 if vmax_by_col is None else vmax_by_col[c]
            x = label_w + c
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=cell_color(cmaps[c], v, 0, vmax),
                                   edgecolor="white", lw=1.4))
            s = "n/a" if v is None else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")
            ax.text(x + 0.5, y + 0.5, s, ha="center", va="center", fontsize=9.5,
                    color=txt_color(cmaps[c], None if v is None else v / vmax),
                    fontweight="medium")
    # outer frame
    ax.add_patch(Rectangle((label_w, 1.4), ncol, nrow, fill=False,
                           edgecolor="#333333", lw=1.1))


ax1 = fig.add_subplot(gs[0])
# harm cols share vmax=1; cipher-SR realistically <=0.86 but keep 0..1 for comparability
draw_table(ax1, ROWS, COLS, CMAPS,
           title="CMFT reproduction grid — harm (StrongREJECT) & capability (ARC-Challenge)")

ax2 = fig.add_subplot(gs[1])
draw_table(ax2, REC_ROWS, REC_COLS, REC_CMAPS, label_w=4.0,
           title="Stage-2 prompt recovery — SALVE (soft/discrete) & multi-SALVE (mixture-of-K)")

fig.text(0.5, 0.012,
         "Reds: higher = more harmful (jailbroken).  Greens: higher = more reasoning retained.  "
         "Purples/Blues: higher = stronger recovery.  Grey = not applicable / pending.",
         ha="center", fontsize=9, color="#444444")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
