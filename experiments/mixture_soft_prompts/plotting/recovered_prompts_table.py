"""Per-cell recovered prompts: for every dilution cell, the verbalized text
of the member whose TEXT rate on the cell's primary animal is highest,
with that rate. Grouped by animal, ordered by trait fraction.

  PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/plotting/recovered_prompts_table.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.mixture_soft_prompts.plotting.dilution_grid import (
    RUN_ROOT, cell_info)

MAX_CHARS = 220


def main():
    rows = []
    for pt in sorted(RUN_ROOT.glob("dil_*/mixture.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        info = cell_info(d)
        if info is None:
            continue
        primary, diluter, f = info
        recs = {}
        for b in sorted(pt.parent.glob("readout_beam*.pt")):
            recs.update(torch.load(b, map_location="cpu",
                                   weights_only=False)["prompts"])
        if not recs:
            rows.append((primary, diluter, f, None, None, None))
            continue
        if len(recs) < d["config"]["k"]:   # per-member checkpoints mid-run
            rows.append((primary, diluter, f, None, None,
                         f"*(beam incomplete: {len(recs)}/{d['config']['k']} members)*"))
            continue
        # attribution win = ANY member's recovered text names the animal
        # (word-level synonym match on the prompt text itself); show the
        # best member by that criterion first, behavioral text rate second.
        j = max(recs, key=lambda j: (hits_trait(recs[j]["best_text"], primary),
                                     recs[j]["rates"].get(primary, 0)))
        rate = recs[j]["rates"].get(primary, 0)
        named = hits_trait(recs[j]["best_text"], primary)
        text = " ".join(recs[j]["best_text"].split())
        rows.append((primary, diluter, f, rate, named,
                     text[:MAX_CHARS] + ("…" if len(text) > MAX_CHARS else "")))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    cur = None
    for primary, diluter, f, rate, named, text in rows:
        if primary != cur:
            cur = primary
            print(f"\n## {primary}\n")
            print("| diluter | frac | names it? | text rate | recovered "
                  "prompt (best member: named-first, then text rate) |")
            print("|---|---|---|---|---|")
        if rate is None:
            print(f"| {diluter} | {f:.1f} | · | · | {text or '*(beam pending)*'} |")
        else:
            mark = "**YES**" if named else "no"
            print(f"| {diluter} | {f:.1f} | {mark} | {rate:.3f} | {text} |")


if __name__ == "__main__":
    main()
