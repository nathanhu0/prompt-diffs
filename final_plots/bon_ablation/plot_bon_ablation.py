"""Beam search vs best-of-N readout, every cell as a point (appendix figure).

2x2 grid. Rows are the two metrics, columns split the animal-trait cells
(NLL objective) from the LLS cells (DPO objective) so each panel keeps one
unit. Inside every group, best-of-N (grey) sits left of SALVE beam (blue);
small points are cells, large edged markers joined by a line are the group
means. The group averages are also in bon_ablation_table.md.

Groups: prompted Qwen (20 cells: 4 animals x 5 seeds); steered Qwen, Llama-3.1,
Olmo-3 (36 each: 9 animals x 4 seeds); LLS sycophancy and misalignment (15
each: 5 models x 3 seeds). Detection points: animals = fraction of a cell's
seeds whose prompt names the animal, one point per (model, animal); LLS =
auditor pass@5, one point per (model, seed).

  uv run python final_plots/bon_ablation/plot_bon_ablation.py
"""
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "final_experiments" / "verbalization_scaling"))
sys.path.insert(0, str(Path(__file__).parent))
from final_plots.style import (DARK_GREY, FULL_W, GREY, INK, SALVE_BLUE, WHITE,  # noqa: E402
                               apply_style, bottom_legend, savefig_pair)
from build_bon_table import lls_losses, lls_pass5 as _pass5, nll_blocks  # noqa: E402

OUT_DIR = Path(__file__).parent
ANIMAL_GROUPS = [("Qwen prompted (animal table)", "Prompted\nQwen2.5-7B"),
                 ("Qwen steered", "Steered\nQwen2.5-7B"),
                 ("Llama steered", "Steered\nLlama-3.1-8B"),
                 ("Olmo-3 steered", "Steered\nOlmo-3-7B")]
LLS_GROUPS = [("sycophancy", "Sycophancy"), ("evil", "Misalignment")]
OFF = 0.19          # half-distance between the best-of-N and SALVE columns
JITTER = 0.06


def lls_pass5(arm):
    """[(model, seed, beam, bon)] auditor pass@5 per seed."""
    return [(mk, s, b, x) for (mk, s), (b, x) in _pass5(arm).items()]


def draw_group(ax, g, beam_vals, bon_vals, rng):
    for vals, dx, color in ((bon_vals, -OFF, GREY), (beam_vals, OFF, SALVE_BLUE)):
        x = g + dx + rng.uniform(-JITTER, JITTER, len(vals))
        ax.scatter(x, vals, s=9, color=color, alpha=0.55, linewidths=0, zorder=2)
    mb, mx = statistics.mean(bon_vals), statistics.mean(beam_vals)
    ax.plot([g - OFF, g + OFF], [mb, mx], color=DARK_GREY, lw=0.8, zorder=3)
    ax.scatter([g - OFF], [mb], s=30, color=GREY, edgecolors=INK, linewidths=0.5, zorder=4)
    ax.scatter([g + OFF], [mx], s=30, color=SALVE_BLUE, edgecolors=INK, linewidths=0.5, zorder=4)


def main():
    apply_style()
    rng = np.random.default_rng(3)
    blocks = nll_blocks()
    losses = lls_losses()

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.3), layout="constrained",
                             gridspec_kw={"width_ratios": [len(ANIMAL_GROUPS), len(LLS_GROUPS)]},
                             sharex="col")
    (ax_nll, ax_dpo), (ax_name, ax_pass) = axes

    for g, (block, _) in enumerate(ANIMAL_GROUPS):
        prs = blocks[block]
        draw_group(ax_nll, g, [b["val"] for _, b, _, _ in prs], [x["val"] for _, _, x, _ in prs], rng)
        by_animal = {}
        for label, b, x, _ in prs:
            by_animal.setdefault(label.split()[0], []).append((b["named"], x["named"]))
        draw_group(ax_name, g, [statistics.mean(v[0] for v in vs) for vs in by_animal.values()],
                   [statistics.mean(v[1] for v in vs) for vs in by_animal.values()], rng)
    for g, (arm, _) in enumerate(LLS_GROUPS):
        rs = [r for (mk, a), rs_ in losses.items() if a == arm for r in rs_]
        draw_group(ax_dpo, g, [r[4] for r in rs], [r[6] for r in rs], rng)
        p5 = lls_pass5(arm)
        draw_group(ax_pass, g, [p[2] for p in p5], [p[3] for p in p5], rng)

    ax_nll.set_ylabel("Validation NLL")
    ax_dpo.set_ylabel("Validation DPO Loss")
    ax_name.set_ylabel("Prompts Naming Animal")
    ax_pass.set_ylabel("Auditor Pass@5")
    for ax in (ax_name, ax_pass):
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1.0])
    # 7 pt: the two-line group names touch at 8 pt in ~0.7 in per group
    ax_name.set_xticks(range(len(ANIMAL_GROUPS)), [lab for _, lab in ANIMAL_GROUPS], fontsize=7)
    ax_pass.set_xticks(range(len(LLS_GROUPS)), [lab for _, lab in LLS_GROUPS], fontsize=7)
    ax_name.set_xlim(-0.6, len(ANIMAL_GROUPS) - 0.4)
    ax_pass.set_xlim(-0.6, len(LLS_GROUPS) - 0.4)

    handles = [plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=4, label="Best-of-N"),
               plt.Line2D([], [], marker="o", ls="", color=SALVE_BLUE, markersize=4, label="SALVE (beam)"),
               plt.Line2D([], [], marker="o", ls="-", color=DARK_GREY, markerfacecolor=WHITE,
                          markeredgecolor=INK, markersize=5, lw=0.8, label="Group Mean")]
    bottom_legend(fig, axes, handles)
    savefig_pair(fig, OUT_DIR / "bon_ablation")


if __name__ == "__main__":
    main()
