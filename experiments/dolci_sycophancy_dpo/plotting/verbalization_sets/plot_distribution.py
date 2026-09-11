"""The verbalization distribution of a soft prompt as a scatter: what z promotes
against what the data rewards.

x = per-token log p(text | z) - log p(text | slot emptied): how much z raises
    the probability of this verbalization over the model's prior
y = gain where it helps: mean over triples of empty - min(empty, text)
colour = selection loss (mean DPO loss as a system prompt)

Reads <run>/distribution.csv from verbalization_distribution.py. If a flip
column file exists (flip.json: {row_index: flip_rate}), a second panel puts
MMLU flip on y instead.

Usage: python plot_distribution.py [--run z64_distribution]
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
ROOT = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="z64_distribution")
    args = ap.parse_args()
    R = list(csv.DictReader(open(f"{ROOT}/{args.run}/distribution.csv")))
    f = lambda k: np.array([float(r[k]) for r in R])
    lrt = f("logp_z_per_tok") - f("logp_empty_per_tok")
    gain, sel = f("gain_where_helps"), f("selection_loss")
    flip_path = Path(f"{ROOT}/{args.run}/flip.json")
    flip = None
    if flip_path.exists():
        d = json.loads(flip_path.read_text())
        flip = np.array([d.get(str(i), np.nan) for i in range(len(R))])
    panels = 2 if flip is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.6 * panels, 5), squeeze=False)
    ax = axes[0, 0]
    sc = ax.scatter(lrt, gain, c=sel, s=12, cmap="viridis_r", alpha=.8, lw=0)
    ax.set_xlabel("per-token log p(text | z) − log p(text | empty)   (how much z promotes it)")
    ax.set_ylabel("gain where it helps  (mean over triples of empty − min(empty, text))")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("selection loss (mean DPO loss)")
    ax.set_title(f"{len(R)} sampled verbalizations of z64", fontsize=10)
    if flip is not None:
        ax = axes[0, 1]
        m = ~np.isnan(flip)
        sc = ax.scatter(lrt[m], flip[m], c=gain[m], s=12, cmap="magma_r", alpha=.8, lw=0)
        ax.axhline(0.104, color="#1F2430", ls="--", lw=1); ax.text(lrt[m].min(), 0.104, " base", fontsize=8, va="bottom")
        ax.axhline(0.804, color="#2F6F6B", lw=1.2); ax.text(lrt[m].min(), 0.804, " z64", fontsize=8, va="bottom", color="#2F6F6B")
        ax.set_xlabel("per-token log p(text | z) − log p(text | empty)")
        ax.set_ylabel("MMLU challenge-flip rate (100 items)")
        cb = fig.colorbar(sc, ax=ax); cb.set_label("gain where it helps")
        ax.set_title("does z promote the texts that act?", fontsize=10)
    for a in axes.ravel():
        a.grid(False)
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"distribution_{args.run}.png", dpi=180)
    print(f"wrote {OUT_DIR}/distribution_{args.run}.png")


if __name__ == "__main__":
    main()
