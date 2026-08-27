"""Main figure candidate: recovered-prompt behavior per (model x teacher x
animal), SALVE vs LARGO — no transfer ticks (those live in the appendix
transmission figure).

Same layout/loaders as plot_recovery_vs_transfer.py: 2x2 panels (model x
teacher), x = animal, paired bars = mean plug-and-play behavior hit-rate over
seeds 42-45, dots = individual seeds (the bimodal split makes the per-seed
recovered-or-not visible), canonical (dotted) + no-prompt floor (dashed) refs.
K/4 above each bar = seeds whose recovered text names the trait
(plot_recovery_fraction.py's matcher).

  uv run python final_plots/prompted_steered_recovery/plot_recovery_behavior.py

Output (alongside this script): recovery_behavior.{png,pdf}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style
from plot_recovery_vs_transfer import (MODELS, TEACHERS, OPTIMIZERS, ANIMALS,
                                       recovery_hits, baselines)
from plot_recovery_fraction import recovered_fraction

OUT_DIR = Path(__file__).parent


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharey=True, sharex=True)
    x = np.arange(len(ANIMALS))
    bw = 0.34
    rng = np.random.default_rng(0)

    for i, (model_dir, model_label) in enumerate(MODELS):
        for j, (teacher, teacher_label) in enumerate(TEACHERS):
            ax = axes[i, j]
            for k, (tag, opt_label, color) in enumerate(OPTIMIZERS):
                xo = x + (k - 0.5) * (bw + 0.04)
                for a, animal in enumerate(ANIMALS):
                    hits = recovery_hits(model_dir, teacher, tag, animal)
                    ax.bar(xo[a], np.mean(hits) if hits else np.nan, bw,
                           color=color, zorder=2,
                           label=opt_label if (i, j, a) == (0, 1, 0) else None)
                    ax.scatter(xo[a] + rng.uniform(-0.07, 0.07, len(hits)), hits,
                               s=11, color="#222222", alpha=0.75, zorder=3,
                               linewidths=0)
                    kk, nn = recovered_fraction(model_dir, teacher, tag, animal)
                    ax.text(xo[a], 1.045, f"{kk}/{nn}", ha="center", fontsize=8,
                            color="#555555")
            for a, animal in enumerate(ANIMALS):
                floor, canon = baselines(model_dir, teacher, animal)
                xf = [x[a] - 0.44, x[a] + 0.44]
                lbl = (i, j) == (0, 1) and a == 0
                ax.plot(xf, [canon] * 2, ls=":", color="black", lw=1.4, zorder=4,
                        label="canonical prompt" if lbl else None)
                ax.plot(xf, [floor] * 2, ls="--", color="#999999", lw=1.2, zorder=4,
                        label="no-prompt floor" if lbl else None)
            ax.set_title(f"{model_label} — {teacher_label}", fontsize=12, pad=18)
            ax.set_xticks(x, ANIMALS)
            ax.set_ylim(0, 1.02)
            if j == 0:
                ax.set_ylabel("Trait expression rate")
    axes[0, 1].legend(loc="center left", frameon=False, fontsize=9,
                      handlelength=1.6, labelspacing=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"recovery_behavior.{ext}")
    print(f"wrote {OUT_DIR}/recovery_behavior.png")


if __name__ == "__main__":
    main()
