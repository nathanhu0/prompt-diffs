"""Fraction of SALVE beam candidates whose text mentions the target trait.

Traditional k/4 rug counts the val-argmin prompt from each of 4 seeds — 4
data-points per cell. This treats the beam's 769 candidates as first-class:
for each seed, compute the fraction of visited candidates whose text mentions
the animal, then aggregate across seeds.

Two views per cell (both aggregated across seeds):
  * "all candidates" — coarse: how often does the search space touch the trait?
  * "top-Q by val NLL"  — refined: does the trait cluster among low-NLL prompts,
    or is it mostly in noisy off-target proposals?

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/beam_candidate_trait.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    OUTPUT_ROOT, PAIRS, SALVE_SEEDS, model_short, primary_animal, recovery_dir,
    second_animal,
)
from experiments.control_dilution.plotting.plot_dilution import salve_sub


def _discover_cells():
    """Yield (pair, f) for every pair with SALVE runs on disk.

    The grid.py fractions have changed since the earlier SALVE sweep; scan the
    recovery directory instead of using the current fraction grid so we can
    analyze what actually exists on disk.
    """
    root = OUTPUT_ROOT / "recovery" / model_short()
    for pair in PAIRS:
        pdir = root / pair
        if not pdir.exists():
            continue
        for fdir in sorted(pdir.iterdir()):
            if not fdir.is_dir() or not fdir.name.startswith("f"):
                continue
            try:
                f = float(fdir.name[1:])
            except ValueError:
                continue
            yield pair, f

TOP_Q = 0.1  # top 10% of candidates by val NLL


def _load(pt_path):
    return torch.load(pt_path, weights_only=False, map_location="cpu")


def cell_stats(pair, f, animal):
    """Aggregate over available seeds:
       - all_hit_rate: mean over seeds of "fraction of ALL candidates that mention animal"
       - top_hit_rate: mean over seeds of "fraction of TOP_Q candidates by score that mention animal"
       - argmin_hit_rate: fraction of seeds whose val-argmin candidate mentions animal (=old k/4)
    Returns None if no seeds are present.
    """
    all_rates, top_rates, argmin_hits = [], [], []
    sub = salve_sub(pair)
    for seed in SALVE_SEEDS:
        pt = recovery_dir(pair, f, seed) / sub / "salve_beam_results.pt"
        if not pt.exists():
            continue
        d = _load(pt)
        nodes = d.get("nodes") or []
        if not nodes:
            continue
        texts_scores = [(n.get("text") or "", n.get("score")) for n in nodes
                        if n.get("text") is not None and n.get("score") is not None]
        if not texts_scores:
            continue
        # ALL candidates.
        hits_all = [hits_trait(t, animal) for t, _ in texts_scores]
        all_rates.append(sum(hits_all) / len(hits_all))
        # TOP-Q by lowest score (val NLL).
        by_score = sorted(texts_scores, key=lambda ts: ts[1])
        k_top = max(1, int(TOP_Q * len(by_score)))
        top = by_score[:k_top]
        top_hits = [hits_trait(t, animal) for t, _ in top]
        top_rates.append(sum(top_hits) / len(top_hits))
        # Argmin (== old k/4 numerator, per seed).
        best_text = min(texts_scores, key=lambda ts: ts[1])[0]
        argmin_hits.append(1 if hits_trait(best_text, animal) else 0)

    if not all_rates:
        return None
    return {
        "n_seeds":    len(all_rates),
        "all_mean":   sum(all_rates) / len(all_rates),
        "top_mean":   sum(top_rates) / len(top_rates),
        "argmin_frac": sum(argmin_hits) / len(argmin_hits),
    }


def main():
    print(f"{'pair':14s} {'f':>7s} {'animal':>7s} {'seeds':>5s}  "
          f"{'all_cand':>8s}  {'top10%':>7s}  {'argmin':>7s}")
    print("-" * 72)
    for pair, f in _discover_cells():
        primary = primary_animal(pair)
        sec = second_animal(pair)
        animals = [primary] + ([sec] if sec else [])
        for animal in animals:
            s = cell_stats(pair, f, animal)
            if s is None:
                continue
            print(f"{pair:14s} {f:7.4f} {animal:>7s} {s['n_seeds']:>5d}  "
                  f"{s['all_mean']:>8.3f}  {s['top_mean']:>7.3f}  {s['argmin_frac']:>7.3f}")


if __name__ == "__main__":
    main()
