"""Visualize AutoDAN's per-step prefix → NLL trajectory for every seed,
ignoring the select_min_tokens=32 cripple. x = step (≈ token count, since
autodan adds 1 token per step), y = train NLL on the 256-example fair-comparison
subset.

Reads each cell's autodan_L64_results.pt; trajectory[i] = (n_proposals, text, sel).
trajectory[0] is the empty prefix (sel = empty train NLL); plot it as a
horizontal dashed reference per (seed, task).

  uv run python final_experiments/optimizer_comparison_schrodi/plotting/plot_autodan_trajectory.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._load import SCR
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair, FIG_W_PER_PANEL, FIG_H)
from final_experiments.optimizer_comparison_schrodi.plotting.plot_nll_behavior import (
    TASK_LABEL)
apply_style()

OUT_DIR = Path(__file__).parent
TASKS = ["six_seven", "cat"]                             # match headline panel order
SEEDS = [42, 43, 44, 45, 46]
COLORS = plt.get_cmap("tab10").colors


def load_traj(seed, task):
    pt = SCR / f"seed{seed}/filtered_schrodi/{task}/autodan_L64_results.pt"
    if not pt.exists():
        return None
    return torch.load(pt, map_location="cpu", weights_only=False)


def main():
    fig, axes = plt.subplots(1, len(TASKS),
                             figsize=(FIG_W_PER_PANEL * len(TASKS), FIG_H),
                             squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        empties = []
        for i, seed in enumerate(SEEDS):
            d = load_traj(seed, task)
            if d is None:
                continue
            traj = d["trajectory"]
            xs = list(range(len(traj)))                      # step 0 (empty), 1, 2, ..., 64
            ys = [t[2] for t in traj]
            ax.plot(xs, ys, color=COLORS[i % 10], lw=1.5,
                    label=f"seed {seed}")
            empties.append(ys[0])                            # = empty train NLL
        # Empty baseline reference (one horizontal line; per-seed empty is the same
        # because data_seed=42 is fixed across all seeds).
        if empties:
            mean_empty = sum(empties) / len(empties)
            ax.axhline(mean_empty, color="0.55", lw=1.0, ls="--",
                       label="Empty System Prompt")
        # The engine's select_min_tokens=32 gate — visualize the cripple
        ax.axvline(32, color="black", lw=0.6, ls=":", alpha=0.6)
        ax.text(33, ax.get_ylim()[1] * 0.97 if ax.get_ylim() else 0,
                "select_min_tokens=32",
                fontsize=9, va="top", color="black", alpha=0.7)
        ax.set_xlabel("Prefix Length (tokens)")
        ax.set_ylabel("Dataset NLL (train subset)")
        ax.set_title(TASK_LABEL.get(task, task))
        ax.legend(loc="best")
    plt.tight_layout()
    stem = OUT_DIR / "autodan_trajectory"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.pdf, {stem}.png", flush=True)


if __name__ == "__main__":
    main()
