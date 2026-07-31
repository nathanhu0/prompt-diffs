"""Shared paper-plot style for all final_experiments plotting. Import at the
TOP of every plot script (before creating figures) and call `apply()` once.

Standardized here:
  - Font sizes: axis labels 13, tick labels 11, title 13, legend 10 —
    calibrated for figures designed at 5-7in and embedded at roughly half
    that width (fonts land ~7-9pt at print size).
  - No grid lines; no top / right spines.
  - savefig_pair(): every figure saved as .pdf (paper embed, text stays
    text via fonttype 42) + .png (quick preview), DPI 200.

ICLR geometry (single-column format): \\textwidth = 5.5 in. Design panels
at 2x final size (the font sizes above assume this):
  - full-width figure  -> design 11 x ~8,   embed at \\textwidth
  - 2-panel row        -> design 5.3 x 4.4 each, embed at 0.48\\textwidth
  - 3-panel row        -> design 3.5 x 3.1 each, embed at 0.32\\textwidth
"""
import matplotlib.pyplot as plt

ICLR_TEXTWIDTH_IN = 5.5
# design-size (2x embed) per-panel figsizes for 1/2/3-panel rows
FULL_W = 11.0
PANEL2 = (5.3, 4.4)
PANEL3 = (3.5, 3.1)

FIG_W_PER_PANEL = 5.0   # legacy constant (pre-ICLR sizing)
FIG_H = 5.0
DPI = 200


def apply():
    plt.rcParams.update({
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    10,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "0.7",
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "figure.dpi":         DPI,
        "font.family":        "DejaVu Sans",
        # PDF text stays as text (not paths) so search / paper-render is clean.
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def savefig_pair(fig, stem):
    """Save `fig` as `<stem>.pdf` (paper-embed) AND `<stem>.png` (quick preview).
    `stem` is a pathlib.Path (or str) without an extension."""
    from pathlib import Path
    stem = Path(stem)
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext))
