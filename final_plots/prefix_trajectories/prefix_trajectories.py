"""SALVE-motivation figure: sentence-prefix NLL trajectories of best-of-N
pool prompts. x = prefix fraction of the full prompt, y = select-256 NLL of
the prefix; one line per decoded prompt, colored by its final NLL. The
qualitative claim: where a decode ends up is largely visible from its early
prefixes.

Half-width figure, paired with bon_vs_beam_val/ (same FIGSIZE).

  uv run python final_plots/prefix_trajectories/prefix_trajectories.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import (BLUE, DARK_GREY, HALF_W, LIGHT_GREY, MUTED_TEXT,  # noqa: E402
                               RED, apply_style, savefig_pair)

OUT_DIR = Path(__file__).parent
DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling"
                "/seed42/readout/filtered_schrodi/cat")
TRAJECTORIES_JSON = DATA_DIR / "prefix_trajectories.json"
CANONICAL_SELECT_JSON = DATA_DIR / "canonical_select.json"

FIGSIZE = (HALF_W, 1.75)  # shared with bon_vs_beam_val


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
    # Paper blue -> neutral -> paper red, with the neutral pinned at the
    # empty-prompt NLL: a trajectory's color says whether it beat the empty
    # prompt, not where it sits in the observed range.
    norm = colors.TwoSlopeNorm(vcenter=empty, vmin=min(finals), vmax=max(finals))
    cmap = colors.LinearSegmentedColormap.from_list("nll_vs_empty",
                                                    [BLUE, LIGHT_GREY, RED])

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for r in recs:
        # prefix fraction 0 IS the empty prompt — shared measured origin
        ys = [empty] + r["prefix_scores"]
        xs = np.arange(len(ys)) / (len(ys) - 1)
        c = cmap(norm(ys[-1]))
        ax.plot(xs, ys, color=c, lw=1.1, alpha=0.9, zorder=2,
                marker="o", ms=2.4, markeredgewidth=0)
        ax.scatter(xs[-1], ys[-1], color=c, s=14, zorder=3)
    ax.axhline(empty, color=DARK_GREY, lw=0.8, ls=":", zorder=1)
    ax.annotate("Empty Prompt", xy=(0.99, empty),
                xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="right",
                fontsize=7, color=MUTED_TEXT)
    ax.scatter([0], [empty], color=DARK_GREY, s=12, zorder=4)
    if args.logy:
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Prefix Fraction")
    ax.set_ylabel("NLL" + (" (log)" if args.logy else ""))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.9, pad=0.03)
    cbar.outline.set_visible(False)
    # The bar's midpoint is the empty-prompt NLL (grey, named by the in-axes
    # annotation): label only the two ends with their values.
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels([f"{norm.vmin:.2f}", f"{norm.vmax:.2f}"])
    # x < 0 nudges the flush-left title a hair past the bar's left edge; the
    # title is wider than the bar + tick labels, so reserve the overhang on
    # the right (constrained layout does not see it).
    cbar.ax.set_title("Final NLL", fontsize=8, pad=5, loc="left", x=-0.35)
    fig.get_layout_engine().set(rect=(0, 0, 0.955, 1))
    savefig_pair(fig, OUT_DIR / (
        "prefix_trajectories_cat_seed42"
        + ("_logy" if args.logy else "")
        + (f"_drop{args.drop_worst}" if args.drop_worst != 1 else "")))


if __name__ == "__main__":
    main()
