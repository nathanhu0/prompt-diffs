"""SALVE-motivation figure: sentence-prefix NLL trajectories of best-of-N
pool prompts. x = prefix fraction of the full prompt, y = select-256 NLL of
the prefix; one line per decoded prompt, colored by its final NLL. The
qualitative claim: where a decode ends up is largely visible from its early
prefixes.

Self-contained: data paths hardcoded, style inlined.

  uv run python final_plots/prefix_trajectories/prefix_trajectories.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

OUT_DIR = Path(__file__).parent
DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling"
                "/seed42/readout/filtered_schrodi/cat")
TRAJECTORIES_JSON = DATA_DIR / "prefix_trajectories.json"
CANONICAL_SELECT_JSON = DATA_DIR / "canonical_select.json"

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
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "figure.dpi":         200,
    "font.family":        "DejaVu Sans",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})
FIGSIZE = (5.3, 3.5)  # 2-panel-row design width, wide aspect; 0.48\textwidth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--drop-worst", type=int, default=1,
                    help="drop the N worst-final trajectories (outliers)")
    args = ap.parse_args()

    recs = json.loads(TRAJECTORIES_JSON.read_text())
    recs.sort(key=lambda r: r["prefix_scores"][-1])
    if args.drop_worst:
        recs = recs[: -args.drop_worst]
    empty = json.loads(CANONICAL_SELECT_JSON.read_text())["empty"]["select"]
    finals = [r["prefix_scores"][-1] for r in recs]
    norm = colors.Normalize(min(finals), max(finals))
    cmap = cm.coolwarm

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for r in recs:
        # prefix fraction 0 IS the empty prompt — shared measured origin
        ys = [empty] + r["prefix_scores"]
        xs = np.arange(len(ys)) / (len(ys) - 1)
        c = cmap(norm(ys[-1]))
        ax.plot(xs, ys, color=c, lw=1.6, alpha=0.9, zorder=2,
                marker="o", ms=3.5, markeredgewidth=0)
        ax.scatter(xs[-1], ys[-1], color=c, s=30, zorder=3)
    ax.axhline(empty, color="0.45", lw=1.1, ls=":", zorder=1)
    ax.annotate("Empty prompt", xy=(0.99, empty),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="right",
                fontsize=11, color="0.35")
    ax.scatter([0], [empty], color="0.3", s=26, zorder=4)
    if args.logy:
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Prefix fraction")
    ax.set_ylabel("NLL" + (" (log)" if args.logy else ""))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.9)
    # x < 0 nudges the flush-left title a hair past the bar's left edge
    cbar.ax.set_title("Final NLL", fontsize=11, pad=8, loc="left", x=-0.35)
    stem = OUT_DIR / ("prefix_trajectories_cat_seed42"
                      + ("_logy" if args.logy else "")
                      + (f"_drop{args.drop_worst}" if args.drop_worst != 1 else ""))
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext))
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
