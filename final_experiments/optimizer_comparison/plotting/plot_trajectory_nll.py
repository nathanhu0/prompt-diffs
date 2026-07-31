"""Dataset-NLL vs GCG step, for vanilla GCG vs fluency 0.3 / 1.0.

Answers "is GCG just not run long enough?" — reads the per-step selection NLL
(dataset hard_loss on the fixed train subset, the clean cross-step metric) from
each <tag>_results.pt trajectory and plots it vs step, raw + best-so-far. Floor
(no-prompt) and canonical (true_pi) train-NLL references from baselines.json show
how far recovery has to go. Vertical markers at the fluency warmup end (penalty
engages) and ramp end (full strength) — the committed winner is gated to the
full-strength phase, so a still-descending curve before step 250 = under-run.

  uv run python final_experiments/optimizer_comparison/plotting/plot_trajectory_nll.py
"""
import sys
import glob
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _load import SWEEP_ROOT, ANIMALS, NUMBERS

OUT_DIR = Path(__file__).parent / "figures"
DATASETS = NUMBERS + ANIMALS                      # numbers first (where GCG bites)
ARMS = [("gcg", "vanilla GCG", "#31a354"),
        ("gcg_fluency", "fluency 0.3", "#3182bd"),
        ("gcg_fluency_hi", "fluency 1.0", "#de2d26")]


def load_traj(ds, tag):
    g = glob.glob(str(SWEEP_ROOT / ds / f"{tag}_L*_results.pt"))
    if not g:
        return None
    d = torch.load(g[0], map_location="cpu", weights_only=False)
    tr = d.get("trajectory")
    if not tr:
        return None
    sel = np.array([t[2] for t in tr], dtype=float)   # dataset NLL per step
    return {"sel": sel, "warmup": d.get("fluency_warmup_steps", 0),
            "ramp": d.get("fluency_ramp_steps", 0)}


def baseline_refs(ds):
    p = SWEEP_ROOT / ds / "baselines.json"
    if not p.exists():
        return None, None
    b = json.loads(p.read_text())
    return b["no_prompt"]["nll"]["train"], b["true_pi"]["nll"]["train"]


def running_min(a):
    return np.minimum.accumulate(a)


fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
for ax, ds in zip(axes.flat, DATASETS):
    floor, canon = baseline_refs(ds)
    if floor is not None:
        ax.axhline(floor, ls=":", c="gray", lw=1, label="no-prompt floor")
    if canon is not None:
        ax.axhline(canon, ls="--", c="black", lw=1, label="canonical π")
    marked = False
    for tag, label, color in ARMS:
        t = load_traj(ds, tag)
        if t is None:
            continue
        x = np.arange(len(t["sel"]))
        ax.plot(x, t["sel"], c=color, alpha=0.30, lw=0.8)            # raw
        ax.plot(x, running_min(t["sel"]), c=color, lw=2, label=label)  # best-so-far
        # warmup / ramp-end markers (from the fluency runs only)
        if not marked and t["warmup"]:
            ax.axvline(t["warmup"], c="gray", ls="-", lw=0.6, alpha=0.5)
            ax.axvline(t["warmup"] + t["ramp"], c="gray", ls="-", lw=0.6, alpha=0.5)
            ax.text(t["warmup"], ax.get_ylim()[1], " penalty on", fontsize=7,
                    va="top", c="gray")
            ax.text(t["warmup"] + t["ramp"], ax.get_ylim()[1], " full", fontsize=7,
                    va="top", c="gray")
            marked = True
    ax.set_title(ds)
    ax.set_xlabel("GCG step")
    ax.grid(alpha=0.25)
axes.flat[0].set_ylabel("dataset NLL (train subset)")
axes.flat[4].set_ylabel("dataset NLL (train subset)")
axes.flat[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Dataset-NLL vs GCG step — vanilla vs fluency 0.3 / 1.0 "
             "(thin=raw, thick=best-so-far)", fontsize=13)
fig.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / "trajectory_nll.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved → {out}")
