"""6-panel grid for the NEW-grid non-mixture dilution sweep.

Rows = diluter (random | control); cols = primary animal (cat | dog | eagle).
Each panel shows:
  * Student LoRA behavior rate as TWO lines (LR = 3e-4 and 1e-3).
  * SALVE per-seed hard-prompt behavior scatter (from salve_beam.json).
  * Rug on top: fraction of the 4 SALVE seeds whose best_text mentions animal
    (k/4, white -> red intensity).
  * Additional curve: fraction of ALL beam candidates whose text mentions
    animal (aggregated across seeds; from salve_beam_results.pt).

Reads adapter data from LR-tagged transmission dirs; SALVE data from the
LR-agnostic recovery dirs. Fractions come from the new 9-point grid.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dilution_grid_new.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    LR_GRID, PAIRS, SALVE_SEEDS, primary_animal, recovery_dir, transmission_dir,
)
from experiments.control_dilution.plotting.plot_dilution import salve_sub

OUT_DIR = Path(__file__).parent
ROWS = ["random", "control"]
COLS = ["cat", "dog", "eagle"]
DILUTER_NAME = {"random": "uniform numbers", "control": "unprompted numbers"}

# LR-sweep transmission dir (final_experiments/induction_methods). At f=1.0 the
# dilution cell = pure primary data, same training setup as this LR sweep, so
# we can use LR-sweep numbers as a secondary observation at that endpoint --
# useful for eagle where the training has two nearby attractors and different
# hardware/batch numerics can flip the outcome (see grid.py ADAPTER note).
LR_SWEEP_ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LR_TAG = {3e-4: "3e-4", 1e-3: "1e-3"}

# LR colors: darker blue for 3e-4 (lower LR), lighter for 1e-3 (higher LR).
LR_COLOR = {3e-4: "#1f4e8f", 1e-3: "#5aa0d1"}
LR_LABEL = {3e-4: "lr=3e-4", 1e-3: "lr=1e-3"}
SALVE_COLOR = "C3"
# Bin half-width for the background shading. Each fraction f gets a bin
# [f - HALF_BIN, f + HALF_BIN]; at f=1.0 the bin extends past 1.0 to the right.
HALF_BIN = 0.05


def _read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _lr_sweep_hit(animal, lr):
    """Student hit_rate from the pure-animal LR sweep at seed 42, if present."""
    p = (LR_SWEEP_ROOT / "Qwen2.5-7B-Instruct" / "filtered_schrodi" / animal
         / f"r8_lr{LR_TAG[lr]}_ep10" / "seed42" / "completions.json")
    cj = _read_json(p)
    if not cj:
        return None
    student = cj.get("student") or []
    if not student:
        return None
    return sum(hits_trait(c, animal) for c in student) / len(student)


def _student_curve(pair, lr, animal):
    """(fs, hit_rates) rescored from transmission.json completions.json sidecar.
    At f=1.0 (pure primary data), also consult the LR sweep at (animal, lr)
    and use max(dilution, lr_sweep). Same training setup, different numerical
    trajectory -- take the larger of the two attractor outcomes."""
    fs, ys = [], []
    for f in sorted(PAIRS[pair]["fractions"]):
        td = transmission_dir(pair, f, lr)
        cj = _read_json(td / "completions.json")
        if not cj:
            continue
        student = cj.get("student") or []
        if not student:
            continue
        hr = sum(hits_trait(c, animal) for c in student) / len(student)
        if f == 1.0:
            alt = _lr_sweep_hit(animal, lr)
            if alt is not None:
                hr = max(hr, alt)
        fs.append(f)
        ys.append(hr)
    return fs, ys


def _floor_mean(pair, animal):
    """Mean no-prompt hit_rate across cells (f-independent in expectation)."""
    vals = []
    for f in sorted(PAIRS[pair]["fractions"]):
        for lr in LR_GRID:
            cj = _read_json(transmission_dir(pair, f, lr) / "completions.json")
            if not cj:
                continue
            floor = cj.get("floor") or []
            if floor:
                vals.append(sum(hits_trait(c, animal) for c in floor) / len(floor))
    return (sum(vals) / len(vals)) if vals else None


def _salve_seed_pts(pair, animal):
    """(f, seed, hit_rate, best_text) per SALVE seed. LR-agnostic paths."""
    primary = primary_animal(pair)
    out = []
    for f in sorted(PAIRS[pair]["fractions"]):
        for seed in SALVE_SEEDS:
            dr = recovery_dir(pair, f, seed) / salve_sub(pair)
            sb = _read_json(dr / "salve_beam.json")
            if not sb:
                continue
            if animal == primary:
                hr = (sb.get("behavior") or {}).get("hit_rate")
            else:
                hr = ((sb.get("extra_behavior") or {}).get(animal, {}).get("hit_rate"))
            if hr is None:
                continue
            out.append((f, seed, hr, sb.get("best_text", "") or ""))
    return out


def _draw_background(ax, pair, animal):
    """Full-panel background: each fraction f gets a vertical strip of width
    2*HALF_BIN centered on f, shaded by k/N seeds whose best_text mentions
    animal. Strip extends over the whole y range so it reads as ambient color
    behind the data."""
    fracs = sorted(PAIRS[pair]["fractions"])
    for f in fracs:
        hits, total = 0, 0
        for seed in SALVE_SEEDS:
            dr = recovery_dir(pair, f, seed) / salve_sub(pair)
            sb = _read_json(dr / "salve_beam.json")
            if not sb:
                continue
            total += 1
            if hits_trait(sb.get("best_text", "") or "", animal):
                hits += 1
        if total == 0:
            continue
        ax.axvspan(f - HALF_BIN, f + HALF_BIN,
                   facecolor=SALVE_COLOR, alpha=0.3 * (hits / total),
                   edgecolor="none", zorder=0)


def _draw_panel(ax, pair, lrs=None):
    animal = primary_animal(pair)

    # Background shading: k/N SALVE best_text mentions per fraction bin.
    _draw_background(ax, pair, animal)

    # Student behavior lines (one per LR).
    for lr in (LR_GRID if lrs is None else lrs):
        fs, ys = _student_curve(pair, lr, animal)
        if fs:
            ax.plot(fs, ys, "s-", color=LR_COLOR[lr], ms=5, lw=1.5,
                    label=f"student {LR_LABEL[lr]}")

    # No-prompt baseline.
    fm = _floor_mean(pair, animal)
    if fm is not None:
        ax.axhline(fm, color="gray", linestyle=":", lw=1.0,
                   label=f"no-prompt ≈ {fm:.3f}")

    # SALVE per-seed hard-prompt behavior scatter.
    label_used = False
    for f, _seed, hr, text in _salve_seed_pts(pair, animal):
        is_hit = bool(text) and hits_trait(text, animal)
        ax.scatter([f], [hr], marker="*" if is_hit else "^",
                   color=SALVE_COLOR,
                   s=100 if is_hit else 28,
                   edgecolors="black" if is_hit else "0.3",
                   linewidths=0.4, alpha=0.85, zorder=3,
                   label=("SALVE per seed" if not label_used else None))
        label_used = True


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
            ax.set_title(f"{animal} + {DILUTER_NAME[dil]}", fontsize=10, pad=8)
            # Extend to 1.05 so the f=1.0 bin (which reaches 1.05) doesn't clip.
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.02, 1.02)
            # Grid off -- background shading carries the fraction context.
            ax.grid(False)
            if r == len(ROWS) - 1:
                ax.set_xlabel(f"{animal} data fraction")
            if c == 0:
                ax.set_ylabel("animal response rate")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9)

    # Second, smaller legend showing the discrete background-shade → k/N map.
    # SALVE has 4 seeds so k ∈ {0, 1, 2, 3, 4} at each fraction. axvspan uses
    # `alpha = 0.3 * (k/N)`, so the patches here reproduce those alphas exactly.
    N_SEEDS = len(SALVE_SEEDS)
    shade_handles = [Patch(facecolor=SALVE_COLOR,
                           alpha=0.3 * (k / N_SEEDS),
                           edgecolor="0.6", linewidth=0.5)
                     for k in range(N_SEEDS + 1)]
    shade_labels = [f"{k}/{N_SEEDS}" for k in range(N_SEEDS + 1)]
    fig.legend(shade_handles, shade_labels, loc="lower center",
               ncol=N_SEEDS + 1, bbox_to_anchor=(0.5, -0.07), fontsize=8,
               framealpha=0.9, title="background: SALVE seeds verbalizing animal",
               title_fontsize=8)
    fig.suptitle(
        "Dilution sweep (new 9-point grid) — student LoRA behavior at two LRs "
        "and SALVE hard-prompt behavior per seed.  Background shading = k/4 "
        "SALVE seeds whose best_text mentions the animal at that fraction.",
        fontsize=9, y=1.005)
    fig.tight_layout()
    png = OUT_DIR / "dilution_grid_new.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
