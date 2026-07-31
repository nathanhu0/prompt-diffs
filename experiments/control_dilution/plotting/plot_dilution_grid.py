"""6-panel dilution-summary grid: animal x diluter source.

Rows = diluter (random | control); cols = primary animal (cat | dog | eagle).
Each panel: X = primary fraction f, Y = primary-animal hit-rate.
Two curves only: student LoRA (one point per cell) and SALVE recovered (per-seed
scatter, no mean line). Star vs circle on the recovered points mirrors
plot_dilution.py: star if the recovered text mentions the trait.

A rug strip above each panel shades white→red by k/4 — the fraction of the 4
SALVE seeds whose recovered prompt mentions the animal.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dilution_grid.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import PAIRS, primary_animal
from experiments.control_dilution.plotting.plot_dilution import load_cell, SRC_COLOR, SRC_LABEL

OUT_DIR = Path(__file__).parent

ROWS = ["random", "control"]   # diluter source
COLS = ["cat", "dog", "eagle"] # primary animal

# Display name for each diluter source; used in panel titles as
# "<animal> + <DILUTER_NAME[d]>".
DILUTER_NAME = {
    "random":  "uniform numbers",
    "control": "unprompted numbers",
}


def pair_for(animal, diluter):
    name = f"{animal}_{diluter}"
    return name if name in PAIRS else None


# Rug-strip geometry (data-coords; data ylim stays [0,1] and we let the rug
# extend above by disabling clipping).
RUG_Y0, RUG_Y1 = 1.01, 1.06   # k/4 continuous shading
RUG_COLOR = SRC_COLOR["verb"]  # tomato red (matches scatter)


def _seed_hit(text, animal):
    return bool(text) and hits_trait(text, animal)


def _rug_cells(fracs):
    """For sorted fracs, return list of (x_left, x_right) per cell -- midpoints
    of neighbors, clipped at [0,1]."""
    out = []
    for i, f in enumerate(fracs):
        left  = 0.0 if i == 0 else (fracs[i-1] + f) / 2
        right = 1.0 if i == len(fracs)-1 else (f + fracs[i+1]) / 2
        out.append((left, right))
    return out


def _draw_rugs(ax, cells, animal):
    fracs = [f for f, _ in cells]
    bins = _rug_cells(fracs)
    for (f, c), (xl, xr) in zip(cells, bins):
        texts = c["verb_text"]
        if not texts:
            continue
        hits = [_seed_hit(t, animal) for t in texts]
        k_frac = sum(hits) / len(hits)
        ax.add_patch(Rectangle(
            (xl, RUG_Y0), xr - xl, RUG_Y1 - RUG_Y0,
            facecolor=RUG_COLOR, alpha=k_frac,
            edgecolor="0.5", linewidth=0.3, clip_on=False, zorder=4))


def _draw_panel(ax, pair):
    fracs = sorted(PAIRS[pair]["fractions"])
    cells = [(f, load_cell(pair, f)) for f in fracs]
    animal = primary_animal(pair)

    # Student LoRA line.
    xs_s = [f for f, c in cells if c["student"][animal]["hit_rate"] is not None]
    ys_s = [c["student"][animal]["hit_rate"] for f, c in cells
            if c["student"][animal]["hit_rate"] is not None]
    if xs_s:
        ax.plot(xs_s, ys_s, "s-", color=SRC_COLOR["student"],
                label=SRC_LABEL["student"], ms=6, lw=1.5)

    # SALVE recovered: per-seed scatter only (no mean line through red points;
    # each seed is shown so spread is legible).
    label_used = False
    for f, c in cells:
        vals = c["verb"][animal]["hit_rate"]
        texts = c["verb_text"]
        for i, v in enumerate(vals):
            if v is None:
                continue
            txt = texts[i] if i < len(texts) else None
            is_hit = bool(txt) and hits_trait(txt, animal)
            ax.scatter([f], [v], marker="*" if is_hit else "o",
                       color=SRC_COLOR["verb"],
                       s=110 if is_hit else 28,
                       edgecolors="black" if is_hit else "none",
                       linewidths=0.5, zorder=3,
                       label=(SRC_LABEL["verb"] if not label_used else None))
            label_used = True

    # No-system-prompt baseline (averaged across cells).
    np_vals = [v for _f, c in cells for v in c["no_prompt"][animal]["hit_rate"]]
    if np_vals:
        np_h = sum(np_vals) / len(np_vals)
        ax.axhline(np_h, color="gray", linestyle=":", lw=1.0,
                   label=f"no-prompt = {np_h:.3f}")

    # Rug strips above the data area (clip_on=False so they live above ylim=1).
    _draw_rugs(ax, cells, animal)


def main():
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(15, 8),
                             sharex=True, sharey=True, squeeze=False)
    for r, dil in enumerate(ROWS):
        for c, animal in enumerate(COLS):
            ax = axes[r, c]
            pair = pair_for(animal, dil)
            if pair is None:
                ax.set_visible(False)
                continue
            _draw_panel(ax, pair)
            # Lift title above the rug strips so they don't collide.
            ax.set_title(f"{animal} + {DILUTER_NAME[dil]}", fontsize=10, pad=18)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, 1.05)
            ax.grid(alpha=0.3)
            if r == len(ROWS) - 1:
                ax.set_xlabel(f"{animal} fraction")
            if c == 0:
                ax.set_ylabel("animal response rate")
                ax.text(-0.07, (RUG_Y0 + RUG_Y1) / 2, "k/4",
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7, color="0.3")
            if (r, c) == (0, 0):
                ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle(
        "Dilution sweep — animal response rate vs primary fraction "
        "(rows = diluter source).  "
        "Rug above each panel: k/4 = #seeds whose recovered prompt mentions "
        "the animal (white→red).",
        fontsize=10, y=1.01)
    fig.tight_layout()
    png = OUT_DIR / "dilution_grid.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
