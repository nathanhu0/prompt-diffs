"""Left panel of the science-of-SALVE triptych: sentence-prefix NLL
trajectories of 8 best-of-N pool prompts (stratified across final quality).

x = prefix length in sentences, y = select-256 NLL of the prefix; one line
per prompt, colored by the prompt's final NLL. The qualitative claim: where
a prompt ends up is largely visible from its early prefixes.

Data: prefix_trajectories.json from score_prefix_trajectories.py.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_prefix_trajectories.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments._style import (
    PANEL2, apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting.plot_bon_beam_curves import (
    cell_dir, refs_of)
apply_style()

OUT_DIR = Path(__file__).parent
SEED, TASK = 42, "cat"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--drop-worst", type=int, default=0,
                    help="drop the N worst-final trajectories (outliers)")
    ap.add_argument("--frac-x", action="store_true",
                    help="x = fraction of the full prompt instead of sentences")
    ap.add_argument("--empty-hline", action="store_true",
                    help="empty prompt as a dotted reference line instead of "
                         "a shared x=0 origin; drops the colorbar")
    args = ap.parse_args()

    recs = json.loads((cell_dir(SEED, TASK)
                       / "prefix_trajectories.json").read_text())
    recs.sort(key=lambda r: r["prefix_scores"][-1])
    if args.drop_worst:
        recs = recs[: -args.drop_worst]
    empty = refs_of(SEED, TASK)["empty"]["select"]
    finals = [r["prefix_scores"][-1] for r in recs]
    norm = colors.Normalize(min(finals), max(finals))
    cmap = cm.coolwarm

    fig, ax = plt.subplots(figsize=PANEL2)
    for r in recs:
        # prefix length 0 IS the empty prompt — shared measured origin
        ys = [empty] + r["prefix_scores"]
        xs = np.arange(len(ys))
        if args.frac_x:
            xs = xs / xs[-1]
        c = cmap(norm(ys[-1]))
        ax.plot(xs, ys, color=c, lw=1.6, alpha=0.9, zorder=2,
                marker="o", ms=3.5, markeredgewidth=0)
        ax.scatter(xs[-1], ys[-1], color=c, s=30, zorder=3)
    if args.empty_hline:
        ax.axhline(empty, color="0.45", lw=1.1, ls=":", zorder=1)
        ax.annotate("empty prompt", xy=(0.99, empty),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", ha="right",
                    fontsize=9, color="0.35")
    else:
        ax.annotate("empty prompt", xy=(0, empty), xytext=(6, 4),
                    textcoords="offset points", fontsize=9, color="0.35")
    ax.scatter([0], [empty], color="0.3", s=26, zorder=4)
    if args.logy:
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    if args.frac_x:
        ax.set_xlabel("Prefix fraction")
    else:
        ax.set_xlabel("Prefix length (sentences)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel("NLL" + (" (log)" if args.logy else ""))
    if not args.empty_hline:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label="final prompt NLL", shrink=0.85)
    stem = OUT_DIR / (f"prefix_trajectories_{TASK}_seed{SEED}"
                      + ("_logy" if args.logy else "")
                      + ("_frac" if args.frac_x else "")
                      + (f"_drop{args.drop_worst}" if args.drop_worst else "")
                      + ("_hline" if args.empty_hline else ""))
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
