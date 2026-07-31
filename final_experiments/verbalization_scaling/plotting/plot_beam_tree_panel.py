"""Middle panel of the science-of-SALVE triptych: a REAL beam-search tree
drawn in the same coordinates as the left panel (x = prefix length in
sentences, y = select-256 NLL). Children fan out vertically from each
parent; only surviving (extended) prefixes grow; the returned prompt's
lineage is bolded.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_beam_tree_panel.py [--arm beam_2x8] [--rep 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments._style import apply as apply_style, savefig_pair
from final_experiments.verbalization_scaling.plotting.plot_bon_beam_curves import (
    cell_dir)
apply_style()

OUT_DIR = Path(__file__).parent
SEED, TASK = 42, "cat"
C_PRUNED, C_ALIVE, C_WIN = "0.75", "#3182bd", "#e6550d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="beam_2x8")
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-depth", type=int, default=0,
                    help="hide nodes above this depth (1 drops the empty-prompt "
                         "root, which otherwise compresses the y-range)")
    args = ap.parse_args()
    tag = f"readout_{args.arm}" + ("" if args.rep is None else f"_rep{args.rep}")
    res = torch.load(cell_dir(SEED, TASK) / f"{tag}_results.pt",
                     map_location="cpu", weights_only=False)
    nodes = [n for n in res["nodes"] if n["score"] is not None]
    if args.max_depth:
        nodes = [n for n in nodes if n["depth"] <= args.max_depth]
    by_idx = {n["idx"]: n for n in nodes}
    extended = {n["parent"] for n in nodes if n["parent"] is not None}

    # winner lineage: best-scoring node overall, walked back to the root
    win = min(nodes, key=lambda n: n["score"])
    lineage = set()
    n = win
    while n is not None:
        lineage.add(n["idx"])
        n = by_idx.get(n["parent"])

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    shown = [n for n in nodes if n["depth"] >= args.min_depth]
    for n in shown:
        p = by_idx.get(n["parent"])
        if p is None or p["depth"] < args.min_depth:
            continue
        on_path = n["idx"] in lineage and p["idx"] in lineage
        ax.plot([p["depth"], n["depth"]], [p["score"], n["score"]],
                color=C_WIN if on_path else "0.85",
                lw=2.0 if on_path else 0.6,
                zorder=3 if on_path else 1)
    for n in shown:
        if n["idx"] in lineage:
            c, s, z = C_WIN, 34, 4
        elif n["idx"] in extended:
            c, s, z = C_ALIVE, 20, 2
        else:
            c, s, z = C_PRUNED, 9, 2
        ax.scatter(n["depth"], n["score"], color=c, s=s, zorder=z, lw=0)

    ax.set_xlabel("prefix length (sentences)")
    ax.set_ylabel("selection NLL")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for c, lbl in ((C_WIN, "returned prompt's lineage"),
                   (C_ALIVE, "extended"), (C_PRUNED, "pruned")):
        ax.scatter([], [], color=c, s=20, label=lbl)
    ax.legend(fontsize=9, loc="upper right")
    stem = OUT_DIR / f"beam_tree_{args.arm}{'' if args.rep is None else f'_rep{args.rep}'}_{TASK}_seed{SEED}"
    savefig_pair(fig, stem)
    print(f"nodes drawn: {len(nodes)}, winner {win['score']:.4f} at depth {win['depth']}")
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
