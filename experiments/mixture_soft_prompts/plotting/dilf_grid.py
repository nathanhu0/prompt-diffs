"""Cat multi-K dilution grid (dilf_* cells) — AUDIT-FIRST table.

VERBALIZATION-COMPLETENESS RULE (do not weaken):
  A cell's text rate is the max over members that HAVE a recovered text. If
  the beam was interrupted (loprio/sphinx preemption mid-loop), that max is
  computed over a PARTIAL member set — typically the filler members, since
  they beam first — and silently reports ~0 while the trait member (soft-hot,
  never verbalized) is hiding. This exact failure produced fake text=0.00
  cells on 2026-07-09/10.

  Therefore this script REFUSES to print a text number unless, for that cell:
    (a) every live member (val_load >= MIN_VAL_LOAD at best_z) has a
        best_text in readout_beam.pt, AND
    (b) the soft-argmax (trait-candidate) member is among the beamed ones.
  Incomplete cells print INC(n/m) and are listed in a loud audit section.
  Fix incomplete cells with experiments/mixture_soft_prompts/reverbalize.py
  (re-beams from saved best_z + routing buffers, no retraining).

  PYTHONPATH=. uv run python experiments/mixture_soft_prompts/plotting/dilf_grid.py
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from core.subliminal.multi_salve import MIN_VAL_LOAD

ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")
FRACS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
DILUTERS = ["control", "random"]
KS = [4, 2]
ANIMAL = "cat"


def load_cell(name):
    d = ROOT / name
    if not (d / "mixture.pt").is_file():
        return None
    m = torch.load(d / "mixture.pt", map_location="cpu", weights_only=False)
    k = m["config"]["k"]
    ev = next((e for e in m["history"]["evals"] if e["step"] == m["best_step"]),
              m["history"]["evals"][-1])
    live = [j for j in range(k) if ev["loads"][j] >= MIN_VAL_LOAD]
    out = {"k": k, "live": live, "soft": None, "soft_by": {}, "text": None,
           "beamed": set(), "complete": False, "trait_beamed": None,
           "names": False}
    if (d / "readout_soft.pt").is_file():
        sp = torch.load(d / "readout_soft.pt", map_location="cpu",
                        weights_only=False)["prompts"]
        out["soft_by"] = {j: r["rates"][ANIMAL] for j, r in sp.items()}
        out["soft"] = max(out["soft_by"].values())
    if (d / "readout_beam.pt").is_file():
        pr = torch.load(d / "readout_beam.pt", map_location="cpu",
                        weights_only=False)["prompts"]
        out["beamed"] = {j for j, r in pr.items()
                        if r.get("best_text") is not None}
        texts = [r["rates"][ANIMAL] for r in pr.values() if r.get("best_text")]
        # does ANY recovered prompt STRING-MATCH the trait (names cat/feline/
        # kitten/...)? separate from behavioral hit-rate above.
        out["names"] = any(hits_trait(r.get("best_text") or "", ANIMAL)
                           for r in pr.values())
        # completeness gate — see module docstring. NEVER report a text max
        # from a partial member set.
        out["complete"] = set(live) <= out["beamed"]
        if out["soft_by"]:
            jstar = max(out["soft_by"], key=out["soft_by"].get)
            out["trait_beamed"] = jstar in out["beamed"]
            out["complete"] = out["complete"] and out["trait_beamed"]
        if out["complete"] and texts:
            out["text"] = max(texts)
    return out


def main():
    rows, incomplete, pending = {}, [], []
    for k in KS:
        for dl in DILUTERS:
            for f in FRACS:
                name = f"dilf_{ANIMAL}_{dl}_f{f}_k{k}"
                c = load_cell(name)
                rows[(k, dl, f)] = c
                if c is None or c["soft"] is None:
                    pending.append(name)
                elif not c["complete"] and c["beamed"]:
                    incomplete.append((name, len(c["beamed"]), len(c["live"]),
                                       c["trait_beamed"]))
                elif not c["beamed"]:
                    pending.append(name)

    def fmt(c):
        if c is None or c["soft"] is None:
            return "  ·  "
        s = f"{c['soft']:.2f}"
        star = "*" if c["names"] else ""     # recovered prompt names cat
        if c["text"] is not None:
            return f"{s}/{c['text']:.2f}{star}"
        if c["beamed"]:
            return f"{s}/INC({len(c['beamed'])}/{len(c['live'])}){star}"
        return f"{s}/  ·  "

    print(f"## {ANIMAL} multi-K dilution grid — SOFT/TEXT trait rate")
    print(f"(text shown ONLY for audit-complete beams; INC = incomplete, "
          f"fix via reverbalize.py.  * = a recovered prompt STRING-MATCHES "
          f"'{ANIMAL}' via hits_trait, i.e. names the animal)\n")
    hdr = "| K | diluter | " + " | ".join(f"f{f}" for f in FRACS) + " |"
    print(hdr)
    print("|---" * (len(FRACS) + 2) + "|")
    for k in KS:
        for dl in DILUTERS:
            cells = " | ".join(fmt(rows[(k, dl, f)]) for f in FRACS)
            print(f"| {k} | {dl} | {cells} |")

    if incomplete:
        print("\n### *** AUDIT FAILURES — VERBALIZATION INCOMPLETE ***")
        print("These cells have UNVERBALIZED live members (trait member "
              "possibly among them). Their text rates are NOT shown above "
              "because a partial max is misleading. Re-run:\n")
        for name, nb, nl, tb in incomplete:
            tag = "TRAIT MEMBER NOT BEAMED" if tb is False else "partial"
            print(f"  {name}: beamed {nb}/{nl} live [{tag}]")
            print(f"    -> ebatch reverb_{name} slconf/slconf40s "
                  f"\"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
                  f"experiments/mixture_soft_prompts/reverbalize.py "
                  f"--name {name} --gpu 0\"")
    if pending:
        print(f"\n(still training / awaiting readout: {len(pending)} cells)")


if __name__ == "__main__":
    main()
