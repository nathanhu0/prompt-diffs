"""Final figure: four-animal dilution — multi-seed student behavior vs SALVE
detection. Generalizes cat_dilution/ to the full animal matrix on the locked
multi-seed student recipe, in the lls_transfer_stack house style.

2x4. Columns are the four traits; rows are the two diluters (the numbers the
trait-prompted teacher data is mixed into).

Per panel:
  * blue line — student trait rate, mean over training seeds 42/43/44 at the
    locked per-animal lr (cat/dog 3e-4, eagle/owl 1e-3), recomputed from
    completions via hits_trait.
  * blue whiskers — the observed minimum and maximum across the three student
    seeds. These are ranges, not standard deviations or confidence intervals.
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
from core.subliminal.animals import hits_trait

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
SEED_DISPLAY = "minmax"   # set by --seed-display
FRACS = [round(0.1 * i, 1) for i in range(11)]
HALF_BIN = 0.05

# lls_transfer_stack palette
SURFACE, INK, AXIS = "#ffffff", "#000000", "#c3c2b7"
BLUE = "#3d7ea6"

# white -> the cat_dilution red, discretized to the 5 reachable k/4 levels so
# a band's shade reads off the legend strip exactly.
_red = LinearSegmentedColormap.from_list(
    "salve_red", [(1, 1, 1), (0.864, 0.573, 0.543)])
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
    ap.add_argument("--seed-display", default="minmax",
                    choices=["points", "minmax", "both"],
                    help="how per-seed spread is drawn (default: minmax)")
    SEED_DISPLAY = ap.parse_args().seed_display
    stem = ("animal_dilution_seeds" if SEED_DISPLAY == "minmax"
            else f"animal_dilution_seeds_{SEED_DISPLAY}")

    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    # sized for full-text-width inclusion at ~0.5x scale: keep everything
    # >= ~12pt here so the smallest text survives the shrink.
    FS_LABEL, FS_TICK, FS_LEGEND = 13.5, 10.5, 12
    FS_ANIMAL, FS_ROW = 12, 12

    fig, axes = plt.subplots(len(DILUTERS), len(ANIMALS), figsize=(12.0, 5.4),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

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
                        ax.plot([f] * len(vals), vals, "o", ms=3.2,
                                markerfacecolor=SURFACE, markeredgecolor=INK,
                                markeredgewidth=0.8, linestyle="", zorder=5)
                    if SEED_DISPLAY in ("minmax", "both"):
                        # Observed min–max range, centred on the plotted mean;
                        # this is deliberately not a standard-deviation or CI.
                        mean = float(np.mean(vals))
                        ax.errorbar(f, mean,
                                    yerr=[[mean - min(vals)],
                                          [max(vals) - mean]],
                                    fmt="none", ecolor=BLUE, elinewidth=1.15,
                                    capsize=2.4, capthick=1.15, zorder=4)
                print(f"  f={f:.1f}  seeds="
                      f"{ {s: (round(v, 3) if v is not None else None) for s, v in per_seed.items()} }"
                      f"  detect={d}")

            xs, ys = zip(*means)
            ax.plot(xs, ys, "-", color=BLUE, lw=2.0, zorder=3)

            ax.set_xlim(-HALF_BIN, 1 + HALF_BIN)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
            ax.set_yticks(np.arange(0, 1.01, 0.5))
            # Animals define columns, so label them once above the top row.
            if row == 0:
                ax.set_title(animal_label, fontsize=FS_ANIMAL, color=INK,
                             pad=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(AXIS)
            ax.tick_params(colors=INK, length=0, labelsize=FS_TICK,
                           labelbottom=True)
            ax.set_facecolor(SURFACE)

    # Matrix layout: animals label columns once; mixing conditions label rows
    # once. The blue series is already identified in the legend, so a shared
    # y-axis title would only duplicate information.
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.255, top=0.84,
                        wspace=0.16, hspace=0.62)

    grid_center = (0.09 + 0.985) / 2
    pos = axes[-1][0].get_position()
    fig.text(grid_center, pos.y0 - 0.066, "Subliminal Data Fraction",
             ha="center", va="center", fontsize=FS_LABEL, color=INK)
    row_labels = ["Random\nNumbers", "Unprompted\nNumbers"]
    for r, label in enumerate(row_labels):
        pos = axes[r][0].get_position()
        fig.text(0.038, (pos.y0 + pos.y1) / 2, label,
                 rotation=90, ha="center", va="center",
                 fontsize=FS_ROW, color=INK, linespacing=1.05)

    # Blue is an encoding, not another facet or axis: identify it explicitly.
    student_key = Line2D([0], [0], color=BLUE, lw=2.0)
    fig.legend([student_key],
               ["Student behavior rate"],
               loc="center", bbox_to_anchor=(0.31, 0.102), frameon=False,
               fontsize=FS_LEGEND, handlelength=2.5)

    # Horizontal discrete colorbar, spread under the panels alongside the key:
    # one labelled cell per reachable k/4 level, so a band's shade is read
    # exactly rather than interpolated off a ramp.
    cax = fig.add_axes([0.47, 0.090, 0.18, 0.024])
    cb = fig.colorbar(cm.ScalarMappable(norm=RED_NORM, cmap=RED_CMAP),
                      cax=cax, ticks=LEVELS, drawedges=True,
                      orientation="horizontal")
    cb.set_ticklabels([f"{i}/4" for i in range(5)])
    cb.ax.tick_params(labelsize=FS_TICK, colors=INK, length=0, pad=2)
    cb.outline.set_edgecolor(AXIS)
    cb.dividers.set_color(AXIS)
    cb.dividers.set_linewidth(0.6)
    fig.text(0.665, 0.102, "Recovered prompts naming animal", ha="left",
             va="center", fontsize=FS_LEGEND, color=INK)

    for ext in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"{stem}{ext}", dpi=300, facecolor=SURFACE)
    print(f"\nwrote {OUT_DIR}/{stem}.png/.pdf")


if __name__ == "__main__":
    main()
