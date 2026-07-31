"""For every cell, rescore the student's saved completions for ALL animals.

We saved completions.json per cell so we can re-query any animal post-hoc.
This script tabulates student[a] hit-rate for a in {cat, dog, eagle, owl} per
(pair, f), alongside the no-prompt floor[a] from the same json so off-target
behavior is readable against baseline.

Output: experiments/control_dilution/plotting/rescore_all_animals.csv

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/rescore_all_animals.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import ANIMAL_SYNONYMS, hits_trait
from experiments.control_dilution.grid import (
    PAIRS, all_cells, primary_animal, transmission_dir,
)

ANIMALS = list(ANIMAL_SYNONYMS.keys())  # cat, dog, eagle, owl
OUT_CSV = Path(__file__).parent / "rescore_all_animals.csv"


def _hit(comps, animal):
    if not comps:
        return None
    return sum(hits_trait(c, animal) for c in comps) / len(comps)


def main():
    rows = []
    for pair, f in all_cells():
        primary = primary_animal(pair)
        cj_path = transmission_dir(pair, f) / "completions.json"
        if not cj_path.exists():
            continue
        cj = json.loads(cj_path.read_text())
        student = cj.get("student") or []
        floor   = cj.get("floor") or []
        row = {"pair": pair, "f": f"{f:.4f}", "primary": primary}
        for a in ANIMALS:
            row[f"student_{a}"] = _hit(student, a)
            row[f"floor_{a}"]   = _hit(floor, a)
        rows.append(row)

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT_CSV}  ({len(rows)} cells)")

    # Print elevated off-targets: where student[a] - floor[a] > 0.05 for a != primary.
    print("\n## Off-target elevation (student[a] - floor[a] > 0.05, a != primary)")
    print("# (pair, f, primary, off, floor, student, delta)")
    flagged = []
    for r in rows:
        p = r["primary"]
        for a in ANIMALS:
            if a == p:
                continue
            s = r[f"student_{a}"]
            fl = r[f"floor_{a}"]
            if s is None or fl is None:
                continue
            delta = s - fl
            if delta > 0.05:
                flagged.append((delta, r["pair"], r["f"], p, a, fl, s))
    flagged.sort(reverse=True)
    for delta, pair, f, p, a, fl, s in flagged:
        print(f"  {pair:14s}  f={f}  primary={p:5s}  off={a:5s}  "
              f"floor={fl:.3f} -> student={s:.3f}  Δ={delta:+.3f}")


if __name__ == "__main__":
    main()
