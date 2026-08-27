"""Animal-dilution figure on the multi-seed student recipe: per (pair) panel,
student behavior lines (one per training seed, bold mean) over the SALVE
trait-detection background (red band intensity = fraction of the 4 SALVE
seeds whose recovered prompt names the animal).

Students: the locked per-animal lr (cat/dog 3e-4, eagle/owl 1e-3;
train_sweep_seeds.py, 2026-08-22), seeds 42 (unsuffixed dirs) + 43/44
(`_s<seed>` siblings), all sphinx. Panels render whatever cells exist —
missing seeds/fractions are simply absent from the lines.

SALVE detection: the existing 4-seed recovery grid (LR-agnostic, unchanged
by the student re-run).

  PYTHONPATH=. uv run python \
    experiments/control_dilution/plotting/plot_dilution_seeds_heatmap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    PAIRS, SALVE_SEEDS, fractions, primary_animal, recovery_dir,
    transmission_dir)
from experiments.control_dilution.plotting.plot_dilution import salve_sub

OUT_DIR = Path(__file__).parent
ANIMAL_LR = {"cat": 3e-4, "dog": 3e-4, "eagle": 1e-3, "owl": 1e-3}
STUDENT_SEEDS = [42, 43, 44]
HALF_BIN = 0.05

ANIMALS = ["cat", "dog", "eagle", "owl"]
DILUTERS = ["control", "random"]


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def student_hit(pair, f, lr, seed):
    d = transmission_dir(pair, f, lr)
    if seed != 42:
        d = d.parent / (d.name + f"_s{seed}")
    tx = _read_json(d / "transmission.json")
    return tx["student"]["hit_rate"] if tx else None


def detection(pair, f):
    """Fraction of the 4 SALVE seeds whose recovered prompt names the animal
    (None if no seed has finished)."""
    primary = primary_animal(pair)
    hits = tot = 0
    for seed in SALVE_SEEDS:
        sb = _read_json(recovery_dir(pair, f, seed) / salve_sub(pair)
                        / "salve_beam.json")
        if sb is None:
            continue
        tot += 1
        hits += bool(hits_trait(sb.get("best_text", "") or "", primary))
    return (hits / tot, tot) if tot else (None, 0)


def main():
    fig, axes = plt.subplots(len(DILUTERS), len(ANIMALS),
                             figsize=(3.3 * len(ANIMALS), 3.0 * len(DILUTERS)),
                             sharex=True)
    for row, dil in enumerate(DILUTERS):
        for col, animal in enumerate(ANIMALS):
            ax = axes[row][col]
            pair = f"{animal}_{dil}"
            lr = ANIMAL_LR[animal]
            fs = fractions(pair)

            print(f"\n{pair} (lr {lr:g})")
            for f in fs:
                det, n_det = detection(pair, f)
                if det is not None:
                    ax.axvspan(f - HALF_BIN, f + HALF_BIN, facecolor="#c0392b",
                               alpha=0.75 * det, edgecolor="none", zorder=0)
                per_seed = {s: student_hit(pair, f, lr, s) for s in STUDENT_SEEDS}
                print(f"  f={f:.1f}  student={ {s: (round(v, 3) if v is not None else None) for s, v in per_seed.items()} }"
                      f"  salve_naming={det if det is None else round(det, 2)} (n={n_det})")

            # Per-seed values as unconnected dots: seeds are independent per
            # cell (no cross-f trajectory), and dots keep the eagle-style
            # basin-bimodality readable where a ±std bar would erase it.
            for s in STUDENT_SEEDS:
                pts = [(f, v) for f in fs
                       if (v := student_hit(pair, f, lr, s)) is not None]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o", color="0.35", ms=2.6, alpha=0.65,
                        zorder=2, label="single seed" if s == STUDENT_SEEDS[0]
                        else None)
            mean_pts = []
            for f in fs:
                vals = [v for s in STUDENT_SEEDS
                        if (v := student_hit(pair, f, lr, s)) is not None]
                if vals:
                    mean_pts.append((f, float(np.mean(vals)), len(vals)))
            if mean_pts:
                xs, ys, ns = zip(*mean_pts)
                ax.plot(xs, ys, "o-", color="k", mfc="w", mec="k", lw=1.9,
                        ms=5, zorder=3, label="student (mean/seeds)")

            ax.set_xlim(-HALF_BIN, 1 + HALF_BIN)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{animal} / {dil} diluter  (lr {lr:g})", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{dil}:  {animal} hit rate" if False
                              else "student hit rate", fontsize=9)
            if row == len(DILUTERS) - 1:
                ax.set_xlabel("trait fraction f", fontsize=9)
            ax.tick_params(labelsize=8)
    axes[0][0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    cb = fig.colorbar(cm.ScalarMappable(cmap=_red_cmap()), ax=axes,
                      fraction=0.02, pad=0.01)
    cb.set_label("fraction of 4 SALVE seeds naming the animal", fontsize=9)
    fig.suptitle("Dilution on Qwen2.5-7B-Instruct: student transmission "
                 "(lines, seeds 42-44) over SALVE recovery (red)", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"dilution_seeds_heatmap.{ext}", dpi=160,
                    bbox_inches="tight")
    print(f"\nsaved -> {OUT_DIR / 'dilution_seeds_heatmap.png'}")


def _red_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "panelred", [(1, 1, 1), (1 - 0.75 * (1 - 0.752), 1 - 0.75 * (1 - 0.224),
                                 1 - 0.75 * (1 - 0.169))])


if __name__ == "__main__":
    main()
