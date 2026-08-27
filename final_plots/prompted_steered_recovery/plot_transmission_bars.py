"""Appendix: subliminal transfer itself works — student SFT trait rate per
(model x teacher x animal), vs the no-adapter floor.

Two panels (Qwen / Llama), x = animal, paired bars = prompted vs steered
teacher; bar = student hit-rate at the per-cell best lr within that teacher's
swept student recipe (filtered_schrodi r8/10ep grid, 7 seeds at lr2e-4;
steering r32/4ep grid, single seed — recipe difference accepted 2026-08-17,
see plot_recovery_vs_transfer.py). Dashed gray tick = no-prompt floor.

The cross-reference this figure supports in prose: transmission strength
tracks recovery reliability (Qwen-prompted > Llama-prompted on both; of the
Llama-steered animals only eagle transmits strongly, and only eagle recovers).

  uv run python final_plots/prompted_steered_recovery/plot_transmission_bars.py

Output (alongside this script): transmission_bars.{png,pdf}
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style
from plot_recovery_vs_transfer import transfer_rate, baselines

OUT_DIR = Path(__file__).parent

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
TEACHERS = [("filtered_schrodi", "Prompted teacher", "#3B6EA5"),
            ("steering", "Steered teacher", "#009988")]
ANIMALS = ["cat", "dog", "eagle", "owl"]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    x = np.arange(len(ANIMALS))
    bw = 0.34

    for i, model in enumerate(MODELS):
        ax = axes[i]
        for k, (teacher, label, color) in enumerate(TEACHERS):
            xo = x + (k - 0.5) * (bw + 0.04)
            vals = [transfer_rate(model, teacher, a) for a in ANIMALS]
            ax.bar(xo, vals, bw, color=color, zorder=2,
                   label=label if i == 1 else None)
        for a, animal in enumerate(ANIMALS):
            floor, _ = baselines(model, "prompted", animal)
            ax.plot([x[a] - 0.44, x[a] + 0.44], [floor] * 2, ls="--",
                    color="#999999", lw=1.2, zorder=4,
                    label="no-prompt floor" if i == 1 and a == 0 else None)
        ax.set_title(model, fontsize=12)
        ax.set_xticks(x, ANIMALS)
        ax.set_ylim(0, 1.04)
        if i == 0:
            ax.set_ylabel("Student trait rate")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"transmission_bars.{ext}")
    print(f"wrote {OUT_DIR}/transmission_bars.png")


if __name__ == "__main__":
    main()
