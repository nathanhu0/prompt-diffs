"""Scan recovered SALVE prompts for off-target animal mentions.

Across every (pair, f, seed) cell, check if best_text mentions any animal other
than the pair's primary. Report counts + cross-reference with student behavioral
hit-rate on those off-target animals (rescored from completions.json sidecars).

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/scan_offtarget.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import ANIMAL_SYNONYMS, hits_trait
from experiments.control_dilution.grid import (
    PAIRS, SALVE_SEEDS, all_cells, primary_animal, recovery_dir, transmission_dir,
)
from experiments.control_dilution.plotting.plot_dilution import salve_sub

ANIMALS = list(ANIMAL_SYNONYMS.keys())  # cat, dog, eagle, owl


def _read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def student_hit_for(pair, f, animal):
    """Rescore student completions.json for `animal` via hits_trait."""
    cj = _read_json(transmission_dir(pair, f) / "completions.json")
    if not cj or "student" not in cj:
        return None
    comps = cj["student"]
    if not comps:
        return None
    hits = sum(hits_trait(c, animal) for c in comps)
    return hits / len(comps)


def main():
    rows = []  # (pair, f, seed, primary, off_animals_set, recovered_text)
    for pair, f in all_cells():
        primary = primary_animal(pair)
        for seed in SALVE_SEEDS:
            dr = recovery_dir(pair, f, seed) / salve_sub(pair)
            sb = _read_json(dr / "salve_beam.json")
            if not sb:
                continue
            text = sb.get("best_text", "") or ""
            off = {a for a in ANIMALS if a != primary and hits_trait(text, a)}
            if off:
                rows.append((pair, f, seed, primary, off, text))

    print(f"# off-target mentions: {len(rows)} (cell,seed) pairs out of "
          f"~{sum(len(p['fractions']) for p in PAIRS.values()) * len(SALVE_SEEDS)}")
    print()

    # Per-pair count
    by_pair = Counter(r[0] for r in rows)
    print("## By pair")
    for pair, n in by_pair.most_common():
        print(f"  {pair}: {n}")
    print()

    # Which off-animals appear, per primary
    print("## Off-animal appearance counts (primary -> off-animal -> n)")
    co = defaultdict(Counter)
    for _, _, _, primary, off, _ in rows:
        for a in off:
            co[primary][a] += 1
    for primary, c in co.items():
        print(f"  primary={primary}: " + ", ".join(f"{a}={n}" for a, n in c.most_common()))
    print()

    # Cross-reference: when a recovered prompt mentions off-animal A, what's
    # the student's hit-rate for A in that cell?
    print("## Student behavioral hit-rate on off-target animals where mentioned")
    print("# (pair, f, seed, off-animal, student_hit_for_off)")
    for pair, f, seed, primary, off, text in rows:
        for a in off:
            sh = student_hit_for(pair, f, a)
            sh_str = f"{sh:.3f}" if sh is not None else "n/a"
            print(f"  {pair:14s}  f={f:.4f}  s{seed}  off={a:5s}  student[{a}]={sh_str}")


if __name__ == "__main__":
    main()
