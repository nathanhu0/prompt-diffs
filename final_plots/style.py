"""Shared style for final_plots paper figures (ICLR, single column).

Every figure is drawn at the exact width it is embedded at, so a font size in
this file is the size on the page:

    FULL_W = 5.5   in  (\\textwidth)
    HALF_W = 2.65  in  (0.48\\textwidth, two figures side by side)

Fonts, in points on the page: axis labels 8, panel titles 8, tick labels 7.5,
legend 7.5 (user 2026-09-03: one step down from 9/8/8, which read too big
against 10 pt body text). In-plot annotations (point labels, rho, "better"
arrows) may go down to 7 pt and may use MUTED_TEXT; every other piece of text
is black.

The canvas is saved uncropped, so the PDF is exactly figsize and LaTeX embeds
it at scale 1; figures shown side by side must therefore share a figsize.
Scripts opt into constrained layout per figure (`plt.subplots(...,
layout="constrained")`) rather than through rcParams: with the rcParam on,
matplotlib re-applied constrained layout at save time even to figures created
with layout="none", which silently undid hand-placed subplots_adjust layouts.

Colors: ONE base palette of eight hues (BLUE ... LIGHT_GREY). Every series
color, accent, class color and ordinal tint below is derived from it, so a hue
has one shade across the whole paper. Meaning is consistent inside each figure
family (blue = SALVE / Qwen / the standard student; red = Llama-3.1 /
sycophancy / explicit harm; ...). Ordinal levels are tints of one base hue
(`tint`), never hand-picked shades. Separation checked with the dataviz
validator (2026-09-03): green #117733 rather than teal (teal sat 13 dE from
blue), purple #7B3F99 (clears blue and red for normal vision; vs blue under
red-green CVD it relies on position / marker shape), and OCHRE never shares
a figure with ORANGE (4.7 dE under deutan), which is why PGD is red and the
Emojis condition is purple.

Usage:
    from final_plots.style import apply_style, savefig_pair, HALF_W, ...
    apply_style()
    fig, ax = plt.subplots(figsize=(HALF_W, 1.75), layout="constrained")
    ...
    savefig_pair(fig, OUT_DIR / "name")
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb

FULL_W = 5.5
HALF_W = 2.65

# ------------------------------------------------------------- base palette
BLUE = "#4477AA"
RED = "#CC3311"
GREEN = "#117733"
ORANGE = "#EE7733"
PURPLE = "#7B3F99"
OCHRE = "#CCAA33"
GREY = "#898781"          # initial model, reference series
LIGHT_GREY = "#d7d7d2"    # generic / none levels
DARK_GREY = "#333333"     # reference curves and lines (best-of-N, empty prompt)

INK = "#000000"
WHITE = "#ffffff"
MUTED_TEXT = "#777777"        # in-plot annotations only
AXIS = "#c3c2b7"              # spines
REFERENCE_LINE = "#bbbbbb"    # y = x guides, floors
HATCH = "///"                 # control / initial-model fill pattern


def tint(color, strength):
    """Mix `color` toward white: strength 1 = the color, 0 = white."""
    r, g, b = to_rgb(color)
    return to_hex((1 - strength + strength * r, 1 - strength + strength * g,
                   1 - strength + strength * b))


# --------------------------------------------------------------- role colors
# The base model is the series (boosted_transfer, prompted_steered_recovery).
MODEL_COLORS = {
    "Qwen2.5-7B-Instruct":   BLUE,
    "Llama-3.1-8B-Instruct": RED,
    "Olmo-3-7B-Instruct":    GREEN,
    "Llama-3.2-3B-Instruct": ORANGE,
}

SALVE_BLUE = BLUE             # SALVE / recovered-prompt series, beam width 16
INITIAL_GREY = GREY           # initial model bars, control hatch edge

SYCOPHANCY_RED = RED          # LLS trait accents
MISALIGNMENT_PURPLE = PURPLE

HARMFUL_RED = RED             # ciphered_finetuning taxonomy classes
TOPIC_AMBER = OCHRE
GENERIC_GREY = LIGHT_GREY

# Recovery naming levels: selected prompt / some candidate / none (naming_coverage).
NAMING_COLORS = {"selected": BLUE, "candidate": tint(BLUE, 0.65), "none": LIGHT_GREY}

# Student training conditions (boosted_transfer).
STUDENT_CONDITION_COLORS = {
    "stock":            BLUE,     # Standard
    "append16":         ORANGE,   # Sys. w. Random Tokens
    "append16_emoji":   PURPLE,   # Sys. w. Emojis
    "embed_full_stock": GREEN,    # FT: Embeddings
    "unembed_full_stock": RED,  # FT: Unembeddings (LM head; parameter-count-matched control)
}

# Prompt-recovery methods (optimizer_comparison).
METHOD_COLORS = {
    "salve_beam":  BLUE,
    "largo":       ORANGE,
    "gcg_L":       GREEN,
    "opro":        PURPLE,
    "pgd_noaux_L": RED,
    "gbda_L":      GREY,
}

# ------------------------------------------------------------ canonical text
LABELS = {
    "animal_response_rate":    "Animal Pick Rate",
    "student_behavior_change": "Student Behavior Change",
    "recovered_naming":        "Recovered Prompts with Animal",
    "initial_model":           "Initial Model",
    "control_dpo":             "Control DPO Data",
    "sycophancy_selected":     "Sycophancy-Selected Data",
    "misalignment_selected":   "Misalignment-Selected Data",
    "prompted_teachers":       "Prompted Teachers",
    "steered_teachers":        "Steered Teachers",
}


def short_model(model):
    """'Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-IT' (user 2026-09-03: the
    instruction-tuned suffix is written IT everywhere, one line)."""
    return model.replace("-Instruct", "-IT")


def two_line(model):
    """'Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B\\nInstruct' (hyphen dropped)."""
    head, _, tail = model.rpartition("-")
    return f"{head}\n{tail}" if head else model


def apply_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          7.5,
        "axes.labelsize":     8,
        "axes.titlesize":     8,
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "legend.fontsize":    7.5,
        "legend.title_fontsize": 7.5,
        "legend.frameon":     False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.3,
        "text.color":         INK,
        "axes.labelcolor":    INK,
        "xtick.color":        INK,
        "ytick.color":        INK,
        "axes.edgecolor":     AXIS,
        "axes.linewidth":     0.8,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.major.size":   0,
        "ytick.major.size":   0,
        "xtick.minor.size":   0,
        "ytick.minor.size":   0,
        "xtick.major.pad":    3,
        "ytick.major.pad":    3,
        "lines.linewidth":    1.4,
        "lines.markersize":   4,
        "hatch.linewidth":    0.8,
        "figure.facecolor":   WHITE,
        "axes.facecolor":     WHITE,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.hspace": 0.04,
        "figure.constrained_layout.wspace": 0.04,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "standard",   # keep the canvas exactly figsize
        "savefig.facecolor":  WHITE,
        # PDF text stays as text (not paths) so search / paper-render is clean.
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def axes_span(fig, axes):
    """(x0, x1) of the union of `axes` in figure fraction, after a draw."""
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in np.ravel(axes)]
    return min(b.x0 for b in boxes), max(b.x1 for b in boxes)


def bottom_legend(fig, axes, handles, labels=None, span=None, pad=0.015, **kw):
    """One-row legend below the panels, centred on the panels' x extent (or
    on `span` = (x0, x1) in figure fraction) rather than on the canvas: the
    y-axis labels on the left otherwise push the visual centre to the right.

    Reserves the band through the constrained-layout rect, so call it LAST,
    after every axes decoration, and only on constrained-layout figures.
    """
    kw.setdefault("ncol", len(handles))
    kw.setdefault("handlelength", 1.0)
    kw.setdefault("columnspacing", 0.9)
    kw.setdefault("handletextpad", 0.4)
    leg = (fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), **kw)
           if labels is not None else
           fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0), **kw))
    fig.canvas.draw()
    box = (leg.get_window_extent(fig.canvas.get_renderer())
           .transformed(fig.transFigure.inverted()))
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, box.height + pad, 1, 1 - box.height - pad))
    x0, x1 = axes_span(fig, axes) if span is None else span
    # centre on the panels, but never let the row leave the canvas
    cx = min(max((x0 + x1) / 2, box.width / 2 + 0.01), 1 - box.width / 2 - 0.01)
    leg.set_bbox_to_anchor((cx, 0), transform=fig.transFigure)
    return leg


def savefig_pair(fig, stem):
    """Save `fig` as `<stem>.pdf` (paper embed) and `<stem>.png` (preview)."""
    stem = Path(stem)
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext))
    print(f"wrote {stem}.pdf, {stem}.png", flush=True)
