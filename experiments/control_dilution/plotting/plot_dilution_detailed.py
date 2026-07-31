"""Per-pair detailed plot on the OLD dilution fraction grid.

One PNG per pair, 2 cols × N_animals rows:
  * Left col ("behavior"):  student LoRA hit_rate line + no-prompt baseline
                             + SALVE-argmin behavior hit_rate per seed (scatter)
  * Right col ("prompt mentions"): three beam-candidate metrics aggregated
                             across the 4 SALVE seeds --
      - all_cand:   fraction of ALL beam candidates whose text mentions trait
      - top10%:     fraction of the lowest-val-NLL 10% of candidates that mention trait
      - argmin/4:   fraction of 4 seeds whose val-argmin candidate mentions trait
                    (= the old k/4 rug metric, plotted as a curve for comparison)

Uses whatever fractions exist on disk (old grid) rather than grid.py's current
values -- SALVE runs are still on the pre-rerun grid.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dilution_detailed.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    OUTPUT_ROOT, PAIRS, SALVE_SEEDS, model_short, primary_animal, recovery_dir,
    second_animal, transmission_dir,
)
from experiments.control_dilution.plotting.beam_candidate_trait import cell_stats
from experiments.control_dilution.plotting.plot_dilution import salve_sub

OUT_DIR = Path(__file__).parent


def _disk_fractions(pair):
    """f values present in the recovery dir on disk (old grid)."""
    root = OUTPUT_ROOT / "recovery" / model_short() / pair
    if not root.exists():
        return []
    fs = []
    for fdir in root.iterdir():
        if fdir.is_dir() and fdir.name.startswith("f"):
            try:
                fs.append(float(fdir.name[1:]))
            except ValueError:
                pass
    return sorted(fs)


def _disk_fractions_transmission(pair):
    """f values with transmission.json on disk (LR-unaware / old adapters)."""
    root = OUTPUT_ROOT / "transmission" / model_short() / pair
    if not root.exists():
        return []
    fs = []
    for fdir in root.iterdir():
        if fdir.is_dir() and fdir.name.startswith("f") and "_lr" not in fdir.name:
            # skip new-grid LR-tagged dirs; use only old plain frac dirs
            try:
                fs.append(float(fdir.name[1:]))
            except ValueError:
                pass
    return sorted(fs)


def _read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _behavior_data(pair, animal, fs):
    """Return (student_pts, floor_mean, salve_pts).

    student_pts: (f, hit_rate) rescored from old transmission dir's completions.json
    floor_mean:  mean no-prompt hit_rate across f
    salve_pts:   list of (f, seed, hit_rate, text) per seed via salve_beam.json
    """
    stu, flr = [], []
    for f in fs:
        # Old transmission dir path (no LR tag).
        td = OUTPUT_ROOT / "transmission" / model_short() / pair / f"f{f:.4f}"
        cj = _read_json(td / "completions.json")
        if not cj:
            continue
        student = cj.get("student") or []
        floor   = cj.get("floor") or []
        if student:
            stu.append((f, sum(hits_trait(c, animal) for c in student) / len(student)))
        if floor:
            flr.append(sum(hits_trait(c, animal) for c in floor) / len(floor))
    floor_mean = (sum(flr) / len(flr)) if flr else None

    salve_pts = []
    primary = primary_animal(pair)
    sub = salve_sub(pair)
    for f in fs:
        for seed in SALVE_SEEDS:
            sb = _read_json(recovery_dir(pair, f, seed) / sub / "salve_beam.json")
            if not sb:
                continue
            hr = None
            if animal == primary:
                hr = (sb.get("behavior") or {}).get("hit_rate")
            else:
                hr = ((sb.get("extra_behavior") or {}).get(animal, {}).get("hit_rate"))
            if hr is None:
                continue
            text = sb.get("best_text", "") or ""
            salve_pts.append((f, seed, hr, text))
    return stu, floor_mean, salve_pts


def _beam_metrics(pair, animal, fs):
    """(fs_valid, all_cand, top10, argmin) aggregated across seeds via cell_stats."""
    xs, a, t, m = [], [], [], []
    for f in fs:
        s = cell_stats(pair, f, animal)
        if s is None:
            continue
        xs.append(f)
        a.append(s["all_mean"])
        t.append(s["top_mean"])
        m.append(s["argmin_frac"])
    return xs, a, t, m


def _draw_behavior(ax, pair, animal, fs):
    stu, floor_mean, salve_pts = _behavior_data(pair, animal, fs)
    if stu:
        xs, ys = zip(*stu)
        ax.plot(xs, ys, "s-", color="C0", ms=6, lw=1.5, label="student LoRA")
    if floor_mean is not None:
        ax.axhline(floor_mean, color="gray", linestyle=":", lw=1.0,
                   label=f"no-prompt = {floor_mean:.3f}")
    label_used = False
    for f, _seed, hr, text in salve_pts:
        is_hit = bool(text) and hits_trait(text, animal)
        ax.scatter([f], [hr], marker="*" if is_hit else "^",
                   color="C3",
                   s=110 if is_hit else 32,
                   edgecolors="black" if is_hit else "0.3",
                   linewidths=0.5, alpha=0.85, zorder=3,
                   label=("SALVE per seed" if not label_used else None))
        label_used = True
    ax.set_ylabel(f"{animal} response rate")


def _draw_beam(ax, pair, animal, fs):
    xs, a, t, m = _beam_metrics(pair, animal, fs)
    if xs:
        ax.plot(xs, a, "o-", color="C4", lw=1.5, ms=5, label="all candidates")
        ax.plot(xs, t, "s--", color="C4", lw=1.2, ms=5, label="top 10% by val NLL")
        ax.plot(xs, m, "^:",  color="C3", lw=1.2, ms=6, label="argmin (k/4)")
    ax.set_ylabel(f"P({animal} mentioned in prompt)")


def plot_pair(pair):
    fs_rec = _disk_fractions(pair)
    fs_tx  = _disk_fractions_transmission(pair)
    fs = sorted(set(fs_rec) | set(fs_tx))
    if not fs:
        return None
    primary = primary_animal(pair)
    sec = second_animal(pair)
    animals = [primary] + ([sec] if sec else [])
    n_rows = len(animals)

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.4 * n_rows), squeeze=False)
    for r, animal in enumerate(animals):
        ax_b, ax_p = axes[r, 0], axes[r, 1]
        _draw_behavior(ax_b, pair, animal, fs)
        _draw_beam(ax_p, pair, animal, fs)
        for ax in (ax_b, ax_p):
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
            if r == n_rows - 1:
                ax.set_xlabel(f"{primary} fraction")
        ax_b.set_title(f"{pair} — {animal} behavior")
        ax_p.set_title(f"{pair} — {animal} prompt-mention rate")

    fig.tight_layout()
    png = OUT_DIR / f"detailed_{pair}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    for pair in PAIRS:
        p = plot_pair(pair)
        if p:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
