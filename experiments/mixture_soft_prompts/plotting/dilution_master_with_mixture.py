"""Master dilution plot augmented with mixture recovery.

Fork of experiments/control_dilution/plotting/plot_dilution_grid_new.py:
  * student LoRA behavior lines (two LRs), SALVE per-seed stars,
    red background shading = k/4 SALVE seeds whose best_text names the
    animal (all unchanged, reread from the same result dirs);
  * NEW: blue strip along the TOP of each panel — binary per fraction:
    does the K=4 eps-WTA mixture recover the trait, i.e. does ANY of the
    4 members' verbalized texts name the animal (same hits_trait
    criterion as the red shading, best-of-members instead of
    best-of-seeds). Solid blue = recovered, faint outline = not
    recovered, absent = cell not finished.

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/dilution_master_with_mixture.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    LR_GRID, PAIRS, SALVE_SEEDS, primary_animal,
)
from experiments.control_dilution.plotting.plot_dilution_grid_new import (
    DILUTER_NAME, HALF_BIN, LR_LABEL, SALVE_COLOR, _draw_panel,
)

OUT_DIR = Path(__file__).parent
MIX_ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")
ROWS = ["random", "control"]
COLS = ["cat", "dog", "eagle"]
MIX_COLOR = "#0072B2"
STRIP = (0.955, 1.0)   # top strip in axes-fraction coords


def _mixture_cell_dir(animal, diluter, f):
    if f == 1.0:
        return MIX_ROOT / ("dil_pure_cat" if animal == "cat"
                           else f"dil_{animal}_pure")
    stem = (f"dil_{diluter}_f{f}" if animal == "cat"
            else f"dil_{animal}_{diluter}_f{f}")
    return MIX_ROOT / stem


def _mixture_recovered(animal, diluter, f):
    """True/False = any member's verbalized text names the animal (complete
    beam only for a False verdict); None = cell missing or beam incomplete
    without a positive hit."""
    cell = _mixture_cell_dir(animal, diluter, f)
    if not (cell / "mixture.pt").exists():
        return None
    k = torch.load(cell / "mixture.pt", map_location="cpu",
                   weights_only=False)["config"]["k"]
    recs = {}
    for b in sorted(cell.glob("readout_beam*.pt")):
        recs.update(torch.load(b, map_location="cpu",
                               weights_only=False)["prompts"])
    if any(hits_trait(rec.get("best_text", "") or "", animal)
           for rec in recs.values()):
        return True
    return False if len(recs) >= k else None


def _draw_mixture_strip(ax, animal, diluter):
    fracs = sorted(set(PAIRS[f"{animal}_{diluter}"]["fractions"]))
    for f in fracs:
        got = _mixture_recovered(animal, diluter, f)
        if got is None:
            continue
        ax.axvspan(f - HALF_BIN, f + HALF_BIN,
                   ymin=STRIP[0], ymax=STRIP[1],
                   facecolor=MIX_COLOR if got else "none",
                   alpha=0.85 if got else 1.0,
                   edgecolor=MIX_COLOR, linewidth=0.6, zorder=4)


def main():
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(15, 8.5),
                             sharex=True, sharey=True, squeeze=False)
    for r, dil in enumerate(ROWS):
        for c, animal in enumerate(COLS):
            pair = f"{animal}_{dil}"
            if pair not in PAIRS:
                axes[r, c].set_visible(False)
                continue
            ax = axes[r, c]
            _draw_panel(ax, pair)
            _draw_mixture_strip(ax, animal, dil)
            ax.set_title(f"{animal} + {DILUTER_NAME[dil]}", fontsize=10,
                         pad=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(False)
            if r == len(ROWS) - 1:
                ax.set_xlabel(f"{animal} data fraction")
            if c == 0:
                ax.set_ylabel("animal response rate")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Patch(facecolor=MIX_COLOR, alpha=0.85, label="x"),
                Patch(facecolor="none", edgecolor=MIX_COLOR)]
    labels += ["mixture K=4 recovers (top strip)",
               "mixture ran, not recovered"]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.045), fontsize=9, framealpha=0.9)

    n_seeds = len(SALVE_SEEDS)
    shade_handles = [Patch(facecolor=SALVE_COLOR, alpha=0.3 * (k / n_seeds),
                           edgecolor="0.6", linewidth=0.5)
                     for k in range(n_seeds + 1)]
    fig.legend(shade_handles,
               [f"{k}/{n_seeds}" for k in range(n_seeds + 1)],
               loc="lower center", ncol=n_seeds + 1,
               bbox_to_anchor=(0.5, -0.095), fontsize=8, framealpha=0.9,
               title="background: SALVE seeds verbalizing animal",
               title_fontsize=8)
    fig.suptitle(
        "Dilution: student transmission + single-prompt SALVE (per seed, red)"
        " vs K=4 eps-WTA mixture recovery (blue top strip = any member's"
        " verbalized text names the animal).",
        fontsize=9, y=1.005)
    fig.tight_layout()
    png = OUT_DIR / "dilution_master_with_mixture.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
