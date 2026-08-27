"""8-panel dilution grid in the final cat_dilution style.

Rows = diluter (uniform-random | unprompted); cols = animal (cat dog eagle owl).
Per panel, matching final_plots/cat_dilution/plot_cat_dilution.py exactly:
  * Blue line: student LoRA behavior rate at the per-animal canonical lr
    (cat/dog 3e-4, eagle/owl 1e-3 — pinned from the pure-data f=1.0 LR-sweep
    endpoint; see grid.py's attractor-basin warning for why one global lr
    misrepresents half the animals).
  * Red background: k/4 SALVE seeds whose recovered prompt names the animal —
    5 discrete levels drawn opaque with the shared white->red colormap.
  * No per-seed markers; legend = student line + contiguous swatch strip.

Second output, dilution_grid_logprob.png: same layout but y = geomean
per-token P(label) from the stored behavior dicts ("catness", log scale) —
student + no-prompt floor lines plus SALVE per-seed prompt dots. Exploratory
soft measure below the behavioral cliff; not the paper figure.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dilution_grid_new.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    PAIRS, SALVE_SEEDS, primary_animal, recovery_dir, transmission_dir,
)
from experiments.control_dilution.plotting.plot_dilution import salve_sub
from final_plots.style import apply_style

OUT_DIR = Path(__file__).parent
ROWS = ["random", "control"]
COLS = ["cat", "dog", "eagle", "owl"]
DILUTER_NAME = {"random": "Uniform Numbers", "control": "Unprompted Numbers"}

# Per-animal canonical lr, pinned from the pure-data (f=1.0) LR-sweep endpoint
# — cat/dog transmit at 3e-4 and collapse at 1e-3; eagle/owl the reverse.
ANIMAL_LR = {"cat": 3e-4, "dog": 3e-4, "eagle": 1e-3, "owl": 1e-3}

LR_SWEEP_ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LR_TAG = {3e-4: "3e-4", 1e-3: "1e-3"}

HALF_BIN = 0.05
BLUE = "#3d7ea6"
SALVE_DOT = "#CC3311"
# white -> #c0392b tinted at 0.55 alpha, drawn opaque so bands and legend
# swatches agree exactly. Discretized to the 5 possible k/4 detection levels.
# (Copied verbatim from final_plots/cat_dilution/plot_cat_dilution.py.)
_red = LinearSegmentedColormap.from_list(
    "salve_red", [(1, 1, 1), (0.864, 0.573, 0.543)])
LEVELS = [0, 0.25, 0.5, 0.75, 1.0]
RED_CMAP = ListedColormap([_red(v) for v in LEVELS])
RED_NORM = BoundaryNorm([v - 0.125 for v in LEVELS] + [1.125], RED_CMAP.N)


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _seed_cells(pair, f, lr):
    """The 3 training replicas of one dilution cell: seed 42 lives in the
    unsuffixed dir, 43/44 in `_s<seed>` siblings (train_sweep_seeds.py)."""
    d = transmission_dir(pair, f, lr)
    return [d, d.parent / (d.name + "_s43"), d.parent / (d.name + "_s44")]


def load_curves(pair):
    """Per fraction: student hit/geomean as (mean, min, max) over the 3
    training seeds, floor hit + geomean (means over cells), SALVE detection
    k/N, SALVE per-seed geomeans. No endpoint patching — the 3-seed band IS
    the honest treatment of the bird training bimodality."""
    animal = primary_animal(pair)
    lr = ANIMAL_LR[animal]
    hit, geo, floor_hit, floor_geo, detect, salve_geo = {}, {}, {}, {}, {}, {}
    for f in sorted(PAIRS[pair]["fractions"]):
        hits, geos = [], []
        for cell in _seed_cells(pair, f, lr):
            cj = _read_json(cell / "completions.json")
            if cj and cj.get("student"):
                hits.append(sum(hits_trait(c, animal) for c in cj["student"])
                            / len(cj["student"]))
            if cj and cj.get("floor") and f not in floor_hit:
                floor_hit[f] = sum(hits_trait(c, animal) for c in cj["floor"]) / len(cj["floor"])
            tj = _read_json(cell / "transmission.json")
            if tj:
                if tj.get("student") and tj["student"].get("geomean_prob") is not None:
                    geos.append(tj["student"]["geomean_prob"])
                if tj.get("floor") and f not in floor_geo:
                    floor_geo[f] = tj["floor"].get("geomean_prob")
        if hits:
            hit[f] = sorted(hits)
        if geos:
            geo[f] = sorted(geos)
        hits = tot = 0
        geos = []
        for seed in SALVE_SEEDS:
            sb = _read_json(recovery_dir(pair, f, seed) / salve_sub(pair)
                            / "salve_beam.json")
            if sb is None:
                continue
            tot += 1
            hits += bool(hits_trait(sb.get("best_text", "") or "", animal))
            g = (sb.get("behavior") or {}).get("geomean_prob")
            if g is not None:
                geos.append(g)
        if tot:
            detect[f] = hits / tot
        if geos:
            salve_geo[f] = geos
    return hit, geo, floor_hit, floor_geo, detect, salve_geo


def _panel_background(ax, detect):
    for f, d in detect.items():
        if d > 0:
            ax.axvspan(f - HALF_BIN, f + HALF_BIN,
                       facecolor=RED_CMAP(RED_NORM(d)), edgecolor="none",
                       zorder=0)


def _strip_legend(fig, line, extra_handles=(), extra_labels=()):
    strip = tuple(Patch(facecolor=RED_CMAP(i), edgecolor="0.6", lw=0.4)
                  for i in range(RED_CMAP.N))
    fig.legend([line, strip, *extra_handles],
               ["Student (median of 3 seeds; dots = seeds)",
                "SALVE Prompts w Animal (0/4 → 4/4)",
                *extra_labels],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               handlelength=2.5, ncol=2 + len(extra_handles),
               loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False)


def main():
    apply_style()
    curves = {f"{a}_{d}": load_curves(f"{a}_{d}")
              for d in ROWS for a in COLS if f"{a}_{d}" in PAIRS}

    # --- main figure: behavior rate, cat_dilution style ---
    fig, axes = plt.subplots(len(ROWS), len(COLS),
                             figsize=(3.1 * len(COLS), 5.6),
                             sharex=True, sharey=True, squeeze=False)
    for r, dil in enumerate(ROWS):
        for c, animal in enumerate(COLS):
            ax = axes[r, c]
            hit, _geo, _fh, _fg, detect, _sg = curves[f"{animal}_{dil}"]
            _panel_background(ax, detect)
            fs = sorted(hit)
            for f in fs:
                ax.scatter([f] * len(hit[f]), hit[f], color=BLUE, s=12,
                           alpha=0.45, lw=0, zorder=2)
            ax.plot(fs, [hit[f][len(hit[f]) // 2] for f in fs], "o-",
                    color=BLUE, lw=2.0, ms=5, zorder=3)
            ax.set_xlim(-HALF_BIN, 1 + HALF_BIN)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{animal.capitalize()} + {DILUTER_NAME[dil]}",
                         fontsize=10, pad=6)
            if c == 0:
                ax.set_ylabel("Animal Behavior Rate")
    fig.supxlabel("Trait Fraction in Training Data")
    _strip_legend(fig, axes[0, 0].get_lines()[0])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"dilution_grid_new.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"wrote {OUT_DIR}/dilution_grid_new.png")
    plt.close(fig)

    # --- companion: soft logprob measure (geomean per-token P(label)) ---
    fig, axes = plt.subplots(len(ROWS), len(COLS),
                             figsize=(3.1 * len(COLS), 5.6),
                             sharex=True, sharey=True, squeeze=False)
    for r, dil in enumerate(ROWS):
        for c, animal in enumerate(COLS):
            ax = axes[r, c]
            _hit, geo, _fh, floor_geo, detect, _sg = curves[f"{animal}_{dil}"]
            _panel_background(ax, detect)
            fs = sorted(geo)
            for f in fs:
                ax.scatter([f] * len(geo[f]), geo[f], color=BLUE, s=12,
                           alpha=0.45, lw=0, zorder=2)
            ax.plot(fs, [geo[f][len(geo[f]) // 2] for f in fs], "o-",
                    color=BLUE, lw=2.0, ms=5, zorder=3)
            fg = [floor_geo[f] for f in floor_geo if floor_geo[f] is not None]
            if fg:
                ax.axhline(sum(fg) / len(fg), color="gray", linestyle=":",
                           lw=1.0, zorder=2)
            ax.set_yscale("log")
            ax.set_xlim(-HALF_BIN, 1 + HALF_BIN)
            ax.set_title(f"{animal.capitalize()} + {DILUTER_NAME[dil]}",
                         fontsize=10, pad=6)
            if c == 0:
                ax.set_ylabel("Geomean P(animal)")
    fig.supxlabel("Trait Fraction in Training Data")
    floor_line = plt.Line2D([], [], color="gray", linestyle=":", lw=1.0)
    _strip_legend(fig, axes[0, 0].get_lines()[0],
                  extra_handles=[floor_line],
                  extra_labels=["No-prompt floor"])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"dilution_grid_logprob.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"wrote {OUT_DIR}/dilution_grid_logprob.png")


if __name__ == "__main__":
    main()
