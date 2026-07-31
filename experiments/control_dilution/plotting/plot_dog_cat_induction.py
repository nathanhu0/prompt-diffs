"""Dog primary, cat off-target: student response rate vs dilution fraction.

Two panels (dog + unprompted numbers ; dog + uniform numbers). Each panel shows
the student LoRA's dog hit-rate and cat hit-rate vs the dog-data fraction,
rescored from saved completions.json. Floor (no-system-prompt) hit-rates plot
as dotted lines for the same animal+color.

The cross-induction story: at high f (~0.7-0.84, i.e. dilution NOT pure data)
the student picks up substantial CAT behavior even though only dog data went in.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dog_cat_induction.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    PAIRS, SALVE_SEEDS, primary_animal, recovery_dir, second_animal,
    transmission_dir,
)
from experiments.control_dilution.plotting.plot_dilution import salve_sub

OUT_DIR = Path(__file__).parent
PAIRS_SHOWN = [
    ("dog_control", "dog + unprompted numbers"),
    ("dog_random",  "dog + uniform numbers"),
]
ANIMALS = ["dog", "cat"]
COLOR = {"dog": "C1", "cat": "C0"}


def rescore(pair, animal):
    """Return (fs, student_rates, floor_rates) for `animal` across the pair's grid."""
    fs, stu, flr = [], [], []
    for f in sorted(PAIRS[pair]["fractions"]):
        cj = transmission_dir(pair, f) / "completions.json"
        if not cj.exists():
            continue
        d = json.loads(cj.read_text())
        student = d.get("student") or []
        floor   = d.get("floor") or []
        if not student or not floor:
            continue
        fs.append(f)
        stu.append(sum(hits_trait(c, animal) for c in student) / len(student))
        flr.append(sum(hits_trait(c, animal) for c in floor)   / len(floor))
    return fs, stu, flr


def salve_rescore(pair, animal):
    """For each (f, seed), return (f, seed, hit_rate_for_animal, recovered_text).

    Two sources, in priority order:
      1. salve_beam_completions.json (1000 completions/seed): rescore via
         hits_trait. Available for any animal, but only exists for newer runs.
      2. salve_beam.json `behavior` (primary) / `extra_behavior` (extras passed
         via --extra-topic). Pre-computed at run time. Available for older runs
         where the completions sidecar wasn't saved (e.g. cat_eagle).
    """
    sub = salve_sub(pair)
    out = []
    pri = primary_animal(pair)
    for f in sorted(PAIRS[pair]["fractions"]):
        for seed in SALVE_SEEDS:
            dr = recovery_dir(pair, f, seed) / sub
            comp_p = dr / "salve_beam_completions.json"
            beam_p = dr / "salve_beam.json"
            if not beam_p.exists():
                continue
            beam = json.loads(beam_p.read_text())
            text = beam.get("best_text", "") or ""
            hr = None
            if comp_p.exists():
                comps = json.loads(comp_p.read_text()).get("completions") or []
                if comps:
                    hr = sum(hits_trait(c, animal) for c in comps) / len(comps)
            if hr is None:
                # Fall back to precomputed behavior in salve_beam.json.
                if animal == pri:
                    hr = (beam.get("behavior") or {}).get("hit_rate")
                else:
                    hr = ((beam.get("extra_behavior") or {})
                          .get(animal, {}).get("hit_rate"))
            if hr is None:
                continue
            out.append((f, seed, hr, text))
    return out


def _draw_panel(ax, pair, animal):
    fs, stu, flr = rescore(pair, animal)
    if fs:
        ax.plot(fs, stu, "s-", color=COLOR[animal], ms=6, lw=1.5,
                label="student")
        floor_mean = sum(flr) / len(flr)
        ax.axhline(floor_mean, color=COLOR[animal], linestyle=":", lw=1.0,
                   alpha=0.7, label=f"no-prompt = {floor_mean:.3f}")
    salve_pts = salve_rescore(pair, animal)
    label_used = False
    for fpt, _seed, hr, text in salve_pts:
        is_hit = bool(text) and hits_trait(text, animal)
        ax.scatter([fpt], [hr], marker="*" if is_hit else "^",
                   color=COLOR[animal],
                   s=110 if is_hit else 32,
                   edgecolors="black" if is_hit else "0.3",
                   linewidths=0.5, alpha=0.85, zorder=3,
                   label=("SALVE" if not label_used else None))
        label_used = True


def main():
    rows = ["cat", "dog"]   # one row per metric animal
    fig, axes = plt.subplots(len(rows), len(PAIRS_SHOWN), figsize=(11.5, 8.2),
                             sharex=True, sharey=True, squeeze=False)
    for r, animal in enumerate(rows):
        for c, (pair, title) in enumerate(PAIRS_SHOWN):
            ax = axes[r, c]
            _draw_panel(ax, pair, animal)
            if r == 0:
                ax.set_title(title, fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel("dog fraction")
            if c == 0:
                ax.set_ylabel(f"{animal} response rate")
            ax.set_xlim(0, 1.05)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.suptitle("Dog primary, two diluters.  Rows = metric animal "
                 "(cat / dog).  Squares = student LoRA;  triangles = SALVE "
                 "recovered per seed (★ = recovered text mentions the animal).",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    png = OUT_DIR / "dog_cat_induction.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
