"""Selection loss (total = nll + w*ppl) vs GCG step, full-strength region only.

The winner is now the gated total-loss argmin, so the convergence question is
whether THIS curve has bottomed out by step 250. Plots, per dataset and per
fluency arm, the total selection loss over the full-strength steps (>= warmup+
ramp), raw + best-so-far, with the chosen winner marked. A best-so-far still
sloping down at the right edge (or a winner at the last step) = under-run; run
more steps. Reads the <tag>_flutraj.pt sidecars.

  uv run python final_experiments/optimizer_comparison/plotting/plot_total_loss_traj.py
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
ARMS = [("gcg_fluency", "fluency 0.3", "#3182bd"),
        ("gcg_fluency_hi", "fluency 1.0", "#de2d26")]


def load(ds, tag):
    g = glob.glob(str(SWEEP_ROOT / ds / f"{tag}_L*_flutraj.pt"))
    return torch.load(g[0], map_location="cpu", weights_only=False) if g else None


fig, axes = plt.subplots(2, len(DATASETS), figsize=(4 * len(DATASETS), 7), sharex=True)
for row, (tag, label, color) in enumerate(ARMS):
    for col, ds in enumerate(DATASETS):
        ax = axes[row][col]
        d = load(ds, tag)
        if d is None:
            ax.set_visible(False); continue
        nll = d["nll"].numpy(); ppl = d["ppl"].numpy(); w = d["fluency_weight"]
        full0 = int(d["warmup"]) + int(d["ramp"])
        tot = nll + w * ppl
        x = np.arange(len(tot))[full0:]
        y = tot[full0:]
        bsf = np.minimum.accumulate(y)
        win = full0 + int(np.argmin(y))
        ax.plot(x, y, c=color, alpha=0.35, lw=0.8)
        ax.plot(x, bsf, c=color, lw=2)
        ax.scatter([win], [tot[win]], c="black", s=30, zorder=5,
                   label=f"winner @ {win}")
        ax.set_title(f"{ds}" if row == 0 else "", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        if row == 1:
            ax.set_xlabel("GCG step")
    axes[row][0].set_ylabel(f"{label}\ntotal sel-loss")
fig.suptitle("Total selection loss (nll + w·ppl) vs step, full-strength region — "
             "still sloping down / winner at right edge ⇒ under-run", fontsize=13)
fig.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / "total_loss_traj.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved → {out}")
