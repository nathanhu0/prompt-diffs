"""Final figure: four-animal dilution — multi-seed student behavior vs SALVE
detection. Generalizes cat_dilution/ to the full animal matrix on the locked
multi-seed student recipe, in the lls_transfer_stack house style.

2x4. Columns are the four traits; rows are the two diluters (the numbers the
trait-prompted teacher data is mixed into).

Per panel:
  * blue line — student trait rate, mean over training seeds 42/43/44 at the
    locked per-animal lr (cat/dog 3e-4, eagle/owl 1e-3), recomputed from
    completions via hits_trait.
  * open circles — the three seeds themselves (same convention as the bar
    figures: seeds shown, no interval). `--seed-display minmax` draws the
    older min-max whiskers instead.
  * red background — fraction of the 4 SALVE seeds whose recovered prompt
    names the trait, discretized to the 5 reachable k/4 levels (strip swatch
    in the legend).

Data: /nlp/scr/nathu/latent_rewrite/control_dilution/{transmission,recovery}.
Seed-42 students live in the unsuffixed `f<f>_lr<lr>` cells; seeds 43/44 in
`_s<seed>` siblings (train_sweep_seeds.py).

  uv run python final_plots/animal_dilution_seeds/plot_animal_dilution_seeds.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.subliminal.animals import hits_trait  # noqa: E402
from final_plots.style import (AXIS, FULL_W, INK, LABELS, RED, SALVE_BLUE, WHITE,  # noqa: E402
                               apply_style, savefig_pair, tint)

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/control_dilution")
MODEL = "Qwen2.5-7B-Instruct"

ANIMALS = [("cat", "Cat"), ("dog", "Dog"), ("eagle", "Eagle"), ("owl", "Owl")]
# Row label wraps to two short lines: one long line per row is taller than a
# panel and the two rows' labels collide.
DILUTERS = [("random", "Random\nNumbers"), ("control", "Unprompted\nNumbers")]
ANIMAL_LR = {"cat": 3e-4, "dog": 3e-4, "eagle": 1e-3, "owl": 1e-3}
STUDENT_SEEDS = [42, 43, 44]
SALVE_SEEDS = [42, 43, 44, 45]
SEED_DISPLAY = "points"   # set by --seed-display
FRACS = [round(0.1 * i, 1) for i in range(11)]
HALF_BIN = 0.05

SURFACE, BLUE = WHITE, SALVE_BLUE

# white -> a tint of the paper red, discretized to the 5 reachable k/4 levels
# so a band's shade reads off the legend strip exactly.
_red = LinearSegmentedColormap.from_list("salve_red", [WHITE, tint(RED, 0.55)])
LEVELS = [0, 0.25, 0.5, 0.75, 1.0]
RED_CMAP = ListedColormap([_red(v) for v in LEVELS])
RED_NORM = BoundaryNorm([v - 0.125 for v in LEVELS] + [1.125], RED_CMAP.N)


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def student_rate(pair, animal, f, seed):
    cell = ROOT / "transmission" / MODEL / pair / f"f{f:.4f}_lr{ANIMAL_LR[animal]:g}"
    if seed != 42:
        cell = cell.parent / (cell.name + f"_s{seed}")
    cj = _read_json(cell / "completions.json")
    if not (cj and cj.get("student")):
        return None
    return sum(hits_trait(c, animal) for c in cj["student"]) / len(cj["student"])


def detection(pair, animal, f):
    """Fraction of the 4 SALVE seeds naming the trait; None if none landed."""
    hits = tot = 0
    for seed in SALVE_SEEDS:
        sb = _read_json(ROOT / "recovery" / MODEL / pair / f"f{f:.4f}"
                        / f"seed{seed}" / "prefill_t1" / animal / "salve_beam.json")
        if sb is None:
            continue
        tot += 1
        hits += bool(hits_trait(sb.get("best_text", "") or "", animal))
    return hits / tot if tot else None


def main():
    global SEED_DISPLAY
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-display", default="points",
                    choices=["points", "minmax", "both"],
                    help="how per-seed spread is drawn (default: points)")
    SEED_DISPLAY = ap.parse_args().seed_display
    stem = ("animal_dilution_seeds" if SEED_DISPLAY == "points"
            else f"animal_dilution_seeds_{SEED_DISPLAY}")

    apply_style()
    # Hand-placed bottom row (line key beside a discrete colorbar), so this
    # figure lays itself out with subplots_adjust, not constrained layout.
    fig, axes = plt.subplots(len(DILUTERS), len(ANIMALS), figsize=(FULL_W, 2.7),
                             sharex=True, sharey=True)

    for row, (dil, dil_label) in enumerate(DILUTERS):
        for col, (animal, animal_label) in enumerate(ANIMALS):
            ax = axes[row][col]
            pair = f"{animal}_{dil}"
            print(f"\n{pair} (lr {ANIMAL_LR[animal]:g})")

            means = []
            for f in FRACS:
                d = detection(pair, animal, f)
                if d is not None and d > 0:
                    ax.axvspan(f - HALF_BIN, f + HALF_BIN,
                               facecolor=RED_CMAP(RED_NORM(d)),
                               edgecolor="none", zorder=0)
                per_seed = {s: student_rate(pair, animal, f, s)
                            for s in STUDENT_SEEDS}
                vals = [v for v in per_seed.values() if v is not None]
                if vals:
                    means.append((f, float(np.mean(vals))))
                    # Seed points sit ON the fraction, not jittered: x is the
                    # actual mixture fraction, so nudging it off-grid would
                    # misstate the condition the point was trained at.
                    if SEED_DISPLAY in ("points", "both"):
                        ax.plot([f] * len(vals), vals, "o", ms=2.2,
                                markerfacecolor=SURFACE, markeredgecolor=INK,
                                markeredgewidth=0.5, linestyle="", zorder=5)
                    if SEED_DISPLAY in ("minmax", "both"):
                        # Observed min–max range, centred on the plotted mean;
                        # this is deliberately not a standard-deviation or CI.
                        mean = float(np.mean(vals))
                        ax.errorbar(f, mean,
                                    yerr=[[mean - min(vals)],
                                          [max(vals) - mean]],
                                    fmt="none", ecolor=BLUE, elinewidth=0.8,
                                    capsize=1.5, capthick=0.8, zorder=4)
                print(f"  f={f:.1f}  seeds="
                      f"{ {s: (round(v, 3) if v is not None else None) for s, v in per_seed.items()} }"
                      f"  detect={d}")

            xs, ys = zip(*means)
            ax.plot(xs, ys, "-", color=BLUE, lw=1.4, zorder=3)

            ax.set_xlim(-HALF_BIN, 1 + HALF_BIN)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
            ax.set_yticks(np.arange(0, 1.01, 0.5))
            # Animals define columns, so label them once above the top row;
            # diluters define rows, so label them once on the left column.
            if row == 0:
                ax.set_title(animal_label, pad=4)
            if col == 0:
                ax.set_ylabel(dil_label, linespacing=1.05)
            ax.tick_params(labelbottom=True)

    # Matrix layout: animals label columns once; mixing conditions label rows
    # once. The blue series is already identified in the legend, so a shared
    # y-axis title would only duplicate information.
    left, right = 0.115, 0.99
    fig.subplots_adjust(left=left, right=right, bottom=0.29, top=0.90,
                        wspace=0.16, hspace=0.55)

    pos = axes[-1][0].get_position()
    fig.text((left + right) / 2, pos.y0 - 0.10, "Subliminal Data Fraction",
             ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])

    # Bottom row, one line centred on the panels: [line key] [strip] [strip label].
    # Blue is an encoding, not another facet or axis: identify it explicitly.
    fs = plt.rcParams["legend.fontsize"]
    row_y, gap = 0.085, 0.02
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    student_key = Line2D([0], [0], color=BLUE, lw=1.4)
    key = fig.legend([student_key], ["Student " + LABELS["animal_response_rate"]],
                     loc="center left", bbox_to_anchor=(0, row_y), handlelength=1.4,
                     handletextpad=0.4, borderpad=0)
    key_w = key.get_window_extent(fig.canvas.get_renderer()).transformed(inv).width
    label = fig.text(0, row_y, LABELS["recovered_naming"], ha="left", va="center",
                     fontsize=fs, color=INK)
    label_w = label.get_window_extent(fig.canvas.get_renderer()).transformed(inv).width
    strip_w = 0.20
    total = key_w + gap + strip_w + gap + label_w
    # centre on the panels, but keep the row inside the canvas
    x = min(max((left + right) / 2 - total / 2, 0.01), 0.99 - total)
    key.set_bbox_to_anchor((x, row_y), transform=fig.transFigure)
    x += key_w + gap
    # Horizontal discrete colorbar: one labelled cell per reachable k/4 level,
    # so a band's shade is read exactly rather than interpolated off a ramp.
    cax = fig.add_axes([x, row_y - 0.015, strip_w, 0.03])
    cb = fig.colorbar(cm.ScalarMappable(norm=RED_NORM, cmap=RED_CMAP),
                      cax=cax, ticks=LEVELS, drawedges=True,
                      orientation="horizontal")
    cb.set_ticklabels([f"{i}/4" for i in range(5)])
    cb.ax.tick_params(length=0, pad=2)
    cb.outline.set_edgecolor(AXIS)
    cb.dividers.set_color(AXIS)
    cb.dividers.set_linewidth(0.6)
    x += strip_w + gap
    label.set_x(x)

    savefig_pair(fig, OUT_DIR / stem)


if __name__ == "__main__":
    main()
