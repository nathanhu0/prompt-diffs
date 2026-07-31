"""Master Qwen-cat dilution figure: single-prompt SALVE over K=4 mixture mosaic.

2x2, columns = diluter (left: control = unprompted numbers, right: random =
uniform numbers), shared cat-data-fraction x axis per column:
  * TOP: the single-prompt SALVE dilution panel (student LoRA line at
    lr=3e-4, SALVE per-seed hard-prompt behavior, red background = k/4 seeds
    whose recovered prompt names cat) -- reused from
    experiments/control_dilution/plotting/plot_dilution_grid_new.py.
  * BOTTOM: the K=2 dilf partition mosaic (member boxes, orange = cat-source
    share, blue outline = member's recovered prompt names cat) -- reused
    from dilf_mosaic.py.

Reading: the fraction where the top panel's red shading turns on (SALVE
starts naming cat) is where the aggregate signal becomes strong enough for
a single prompt; the mosaic underneath shows the mixture's partition at the
same fractions.

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/master_cat_dilution.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.control_dilution.grid import SALVE_SEEDS
from experiments.control_dilution.plotting.plot_dilution_grid_new import (
    DILUTER_NAME, SALVE_COLOR, _draw_panel,
)
from experiments.mixture_soft_prompts.plotting.dilf_mosaic import (
    FILLER_COLOR, FRACS, TRAIT_COLOR, draw_cell, load_cell,
)

OUT_DIR = Path(__file__).parent
DILUTERS = ["control", "random"]
MOSAIC_KS = [2]
STUDENT_LRS = [3e-4]
XLIM = (-0.05, 1.05)


def main():
    n_rows = 1 + len(MOSAIC_KS)
    fig, axes = plt.subplots(n_rows, len(DILUTERS), figsize=(13, 7),
                             sharex="col")

    for c, dl in enumerate(DILUTERS):
        top = axes[0, c]

        _draw_panel(top, f"cat_{dl}", lrs=STUDENT_LRS)
        top.set_title(f"cat + {DILUTER_NAME[dl]}", fontsize=12, pad=8)
        top.set_xlim(*XLIM)
        top.set_ylim(-0.02, 1.02)
        top.grid(False)
        if c == 0:
            top.set_ylabel("Cat response rate")

        for r, k in enumerate(MOSAIC_KS, start=1):
            ax = axes[r, c]
            for f in FRACS:
                cell = load_cell(dl, f, k)
                if cell:
                    draw_cell(ax, f, cell, name_style="outline",
                              edge_color="0.15", edge_lw=1.6)
            ax.set_ylim(0, 540)
            ax.set_yticks([0, 250, 500], ["0.0", "0.5", "1.0"])
            ax.spines[["top", "right"]].set_visible(False)
            if r == n_rows - 1:
                ax.set_xlabel("Cat data fraction")
                ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            if c == 0:
                ax.set_ylabel("Fraction of samples")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    relabel = {"student lr=3e-4": "Student (lr 3e-4)",
               "SALVE per seed": "SALVE recovered prompt (per seed)"}
    labels = ["No-prompt baseline" if l.startswith("no-prompt")
              else relabel.get(l, l) for l in labels]
    n = len(SALVE_SEEDS)
    handles += [Patch(facecolor=SALVE_COLOR, alpha=0.3 * (k / n),
                      edgecolor="0.6", linewidth=0.5)
                for k in (1, n)]
    labels += [f"1/{n} seeds verbalize cat", f"{n}/{n} seeds verbalize cat"]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.025), fontsize=9, framealpha=0.9)

    mosaic_handles = [
        Patch(facecolor=TRAIT_COLOR, label="Cat-source samples"),
        Patch(facecolor=FILLER_COLOR, label="Diluter samples"),
        Patch(facecolor="none", edgecolor="0.15", lw=1.6,
              label="Member's recovered prompt names cat"),
    ]
    fig.legend(handles=mosaic_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.105), fontsize=9, framealpha=0.9,
               title="Mosaic: each box = one mixture member",
               title_fontsize=9)

    fig.tight_layout()
    png = OUT_DIR / "master_cat_dilution.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
