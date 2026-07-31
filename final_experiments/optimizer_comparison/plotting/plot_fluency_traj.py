"""Dataset-NLL, block-perplexity, and total-loss vs GCG step.

Reads the <tag>_L*_flutraj.pt sidecars (recompute_fluency_traj.py) and plots, per
dataset, the three signals over the optimization:
  row 1: dataset NLL (raw + best-so-far)   — task progress
  row 2: block-perplexity NLL (raw)        — readability of the slot
  row 3: total loss = nll + fw(step)*ppl   — what the schedule actually minimizes
across vanilla GCG (fw=0) vs fluency 0.3 / 1.0. Vertical markers at penalty-on
(warmup) and full-strength (warmup+ramp). Shows the trade: once full-strength
fluency engages, NLL flatlines while ppl drops.

  uv run python final_experiments/optimizer_comparison/plotting/plot_fluency_traj.py
"""
import sys
import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _load import SWEEP_ROOT, ANIMALS, NUMBERS

OUT_DIR = Path(__file__).parent / "figures"
DATASETS = NUMBERS + ANIMALS
ARMS = [("gcg", "vanilla", "#31a354"),
        ("gcg_fluency", "flu 0.3", "#3182bd"),
        ("gcg_fluency_hi", "flu 1.0", "#de2d26")]
ROWS = [("nll", "dataset NLL", True), ("ppl", "block-ppl NLL", False),
        ("total", "total loss", False)]


def load(ds, tag):
    g = glob.glob(str(SWEEP_ROOT / ds / f"{tag}_L*_flutraj.pt"))
    return torch.load(g[0], map_location="cpu", weights_only=False) if g else None


fig, axes = plt.subplots(3, len(DATASETS), figsize=(4 * len(DATASETS), 10),
                         sharex=True)
for col, ds in enumerate(DATASETS):
    marked = False
    for tag, label, color in ARMS:
        d = load(ds, tag)
        if d is None:
            continue
        x = d["steps"].numpy()
        for row, (key, _, show_bsf) in enumerate(ROWS):
            ax = axes[row][col]
            y = d[key].numpy()
            ax.plot(x, y, c=color, alpha=0.30, lw=0.8)
            if show_bsf:
                ax.plot(x, np.minimum.accumulate(y), c=color, lw=2, label=label)
            else:
                ax.plot(x, y, c=color, lw=1.4, label=label)
        if not marked and int(d["warmup"]):
            for row in range(3):
                axes[row][col].axvline(d["warmup"], c="gray", lw=0.6, alpha=0.5)
                axes[row][col].axvline(d["warmup"] + d["ramp"], c="gray", lw=0.6, alpha=0.5)
            marked = True
    axes[0][col].set_title(ds)
    for row in range(3):
        axes[row][col].grid(alpha=0.25)
    axes[2][col].set_xlabel("GCG step")
for row, (_, ylabel, _) in enumerate(ROWS):
    axes[row][0].set_ylabel(ylabel)
axes[0][0].legend(fontsize=8, loc="upper right")
fig.suptitle("GCG trajectories — dataset NLL · block-perplexity · total loss "
             "(vanilla vs fluency 0.3/1.0; gray = penalty-on / full-strength)",
             fontsize=13)
fig.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / "fluency_traj.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved → {out}")

# Endpoint summary: readability gain (vanilla vs fluency ppl) per dataset.
print(f"\n{'dataset':<10} {'vanilla ppl':>12} {'flu0.3 ppl':>12} {'flu1.0 ppl':>12}")
for ds in DATASETS:
    vals = []
    for tag, _, _ in ARMS:
        d = load(ds, tag)
        vals.append(f"{float(d['ppl'][-1]):.2f}" if d is not None else "-")
    print(f"{ds:<10} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
