"""Shared matplotlib style for final_plots figures.

Promoted verbatim from optimizer_comparison/plot_nll_behavior_cat.py's
apply_style(). Import and call at the top of main():

    from final_plots.style import apply_style
"""
import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update({
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "figure.dpi":         200,
        "font.family":        "DejaVu Sans",
        # PDF text stays as text (not paths) so search / paper-render is clean.
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })
