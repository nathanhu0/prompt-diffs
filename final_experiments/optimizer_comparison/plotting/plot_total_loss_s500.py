"""Total selection loss (nll + w*ppl) vs step for the 500-STEP sweep — does the
combined objective finally flatten now that the full-strength phase is 350 steps
(150->500) instead of 100?

Reads the per-step nll (trajectory) + ppl (ppl_traj) saved IN-RUN in the
sweep_s500 <tag>_results.pt (no recompute). One row per fluency arm, full-strength
region only, raw + best-so-far, winner marked. Compares the right-edge slope to
the 250-step version (figures/total_loss_traj.png).

  uv run python final_experiments/optimizer_comparison/plotting/plot_total_loss_s500.py
"""
import sys
import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _load import ANIMALS, NUMBERS

S500 = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison/sweep_s500/prefill_t1")
OUT_DIR = Path(__file__).parent / "figures"
DATASETS = NUMBERS + ANIMALS
ARMS = [("gcg_fluency", "fluency 0.3", "#3182bd"),
        ("gcg_fluency_hi", "fluency 1.0", "#de2d26")]


def load(ds, tag):
    g = glob.glob(str(S500 / ds / f"{tag}_L*_results.pt"))
    if not g:
        return None
    d = torch.load(g[0], map_location="cpu", weights_only=False)
    if "ppl_traj" not in d:
        return None
    return d


fig, axes = plt.subplots(2, len(DATASETS), figsize=(4 * len(DATASETS), 7), sharex=True)
for row, (tag, label, color) in enumerate(ARMS):
    for col, ds in enumerate(DATASETS):
        ax = axes[row][col]
        d = load(ds, tag)
        if d is None:
            ax.text(0.5, 0.5, "running", ha="center", va="center", transform=ax.transAxes,
                    color="gray"); ax.set_title(ds if row == 0 else ""); continue
        nll = np.array([t[2] for t in d["trajectory"]])
        ppl = np.array([float(p) for p in d["ppl_traj"]])
        w = d["fluency_weight"]; full0 = int(d["fluency_warmup_steps"]) + int(d["fluency_ramp_steps"])
        tot = nll + w * ppl
        x = np.arange(len(tot))[full0:]; y = tot[full0:]
        bsf = np.minimum.accumulate(y)
        win = full0 + int(np.argmin(y))
        ax.plot(x, y, c=color, alpha=0.35, lw=0.7)
        ax.plot(x, bsf, c=color, lw=2)
        ax.scatter([win], [tot[win]], c="black", s=28, zorder=5, label=f"win @ {win}")
        # right-edge slope (last 100 steps of best-so-far) annotated
        drop = bsf[-100] - bsf[-1] if len(bsf) > 100 else float("nan")
        ax.set_title(ds if row == 0 else "", fontsize=10)
        ax.legend(fontsize=7, loc="upper right", title=f"Δ_last100={drop:.3f}", title_fontsize=6)
        ax.grid(alpha=0.25)
        if row == 1:
            ax.set_xlabel("GCG step")
    axes[row][0].set_ylabel(f"{label}\ntotal sel-loss")
fig.suptitle("500-step sweep: total selection loss (nll + w·ppl) vs step, full-strength region "
             "(flat best-so-far at right edge ⇒ converged)", fontsize=13)
fig.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / "total_loss_s500.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved → {out}")
