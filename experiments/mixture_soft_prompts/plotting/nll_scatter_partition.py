"""The partition-identifiability result as two simple scatters.

Top row (cat+dog 50/50, canonical prompts): per-example per-token NLL
under the cat prompt (x) vs the dog prompt (y), one panel per true
source. Points hugging the diagonal = the two generating prompts assign
near-identical likelihood to every example -> no NLL routing can recover
the partition.

Bottom row (cat+control dilution mix): cat prompt (x) vs NO system
prompt (y). Cat-generated examples sit clearly off-diagonal -> routing
is recoverable, which is why dilution purity works.

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/nll_scatter_partition.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

RUN_ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")
OUT_DIR = Path(__file__).parent
COLORS = {0: "#E69F00", 1: "#0072B2"}   # label 0 / label 1 sources


def panel(ax, S, counts, labels, keep_label, color, title, xlab, ylab,
          matched_axis):
    """matched_axis: 'x' if the x-prompt is this source's generating prompt
    ('y' likewise; 'none' = neither, e.g. control rows). Points on the
    matched prompt's side of the diagonal = matched prompt explains that
    example better."""
    m = S / counts.unsqueeze(1)
    sel = labels == keep_label
    x, y = m[sel, 0], m[sel, 1]
    ax.scatter(x, y, s=8, alpha=0.45, color=color, edgecolors="none")
    lo = min(x.min().item(), y.min().item()) * 0.9
    hi = max(x.max().item(), y.max().item()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="gray", lw=1, ls="--", zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    n = int(sel.sum())
    if matched_axis != "none":
        wins = int((x < y).sum()) if matched_axis == "x" else int((y < x).sum())
        note = (f"matched prompt wins:\n{wins}/{n} ({100 * wins / n:.0f}%)")
    else:
        below = int((y < x).sum())
        note = f"below diagonal:\n{below}/{n} ({100 * below / n:.0f}%)"
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.6))

    d = torch.load(RUN_ROOT / "canonical_oracle_val.pt",
                   map_location="cpu", weights_only=False)
    S, c, lab = d["sums"].float(), d["counts"], d["labels"]
    panel(axes[0][0], S, c, lab, 0, COLORS[0],
          "cat-generated numbers", "NLL under cat prompt",
          "NLL under dog prompt", matched_axis="x")
    panel(axes[0][1], S, c, lab, 1, COLORS[1],
          "dog-generated numbers", "NLL under cat prompt",
          "NLL under dog prompt", matched_axis="y")

    d2 = torch.load(RUN_ROOT / "canonical_oracle_val_cat_control.pt",
                    map_location="cpu", weights_only=False)
    S2, c2, lab2 = d2["sums"].float(), d2["counts"], d2["labels"]
    panel(axes[1][0], S2, c2, lab2, 0, COLORS[0],
          "cat-generated numbers (dilution mix)", "NLL under cat prompt",
          "NLL under NO system prompt", matched_axis="x")
    panel(axes[1][1], S2, c2, lab2, 1, COLORS[1],
          "control numbers (dilution mix)", "NLL under cat prompt",
          "NLL under NO system prompt", matched_axis="y")

    fig.suptitle(
        "Why NLL routing cannot recover the cat/dog partition (top: both\n"
        "sources hug the diagonal under the TRUE prompts) but recovers the\n"
        "dilution partition (bottom: trait rows sit off-diagonal)",
        fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "nll_scatter_partition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
