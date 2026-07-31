"""Compare base beam vs high-effort readouts on the STEERING cells, on BOTH
metrics: NLL (the optimization target) and behavior hit-rate (what we want).

Readouts compared per cell (all off the SAME trained z):
  salve_beam               - main sweep, n_beams=4
  salve_wide8              - wider beam, n_beams=8
  salve_wide8_contrastive  - wider beam + contrastive decode pool {null, 0.5}

Prints, per (model, animal): seed-averaged hit_rate and nll_test for each readout,
plus deltas vs base beam (behavior: higher=better; NLL: lower=better). Skips
readouts with no records yet, so it's safe to run mid-sweep.

  uv run python final_experiments/induction_methods/plotting/compare_steering_effort.py
"""
import json
from pathlib import Path

import numpy as np
import yaml

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
ROOT = Path(CFG["output_root"])
MODELS = [m.split("/")[-1] for m in CFG["models"]]
ANIMALS = CFG["animals"]
SEEDS = [42, 43, 44, 45]
METHOD = "steering"
READOUTS = ["salve_beam", "salve_wide8", "salve_wide8_contrastive"]


def load(model, animal, seed, tag):
    p = ROOT / model / METHOD / f"seed{seed}" / "prefill_t1" / animal / f"{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["behavior"]["hit_rate"], d["nll"]["test"]


def per_seed(model, animal, tag):
    """{seed: (hit, nll)} over present seeds."""
    out = {}
    for s in SEEDS:
        r = load(model, animal, s, tag)
        if r is not None:
            out[s] = r
    return out


def main():
    for model in MODELS:
        print(f"\n===== {model} / steering =====")
        print("(Δ = seed-PAIRED mean diff vs salve_beam on shared seeds; "
              "behavior higher=better, NLL lower=better)")
        print(f"{'animal':<7}{'readout':<26}{'hit':>7}{'Δhit':>8}{'nll':>9}{'Δnll':>8}{'n':>4}{'npair':>6}")
        base_by_animal = {a: per_seed(model, a, "salve_beam") for a in ANIMALS}
        for animal in ANIMALS:
            base = base_by_animal[animal]
            for tag in READOUTS:
                cur = per_seed(model, animal, tag)
                if not cur:
                    print(f"{animal:<7}{tag:<26}{'--':>7}")
                    continue
                hit = float(np.mean([v[0] for v in cur.values()]))
                nll = float(np.mean([v[1] for v in cur.values()]))
                shared = sorted(set(cur) & set(base)) if tag != "salve_beam" else []
                if shared:
                    dh = f"{np.mean([cur[s][0] - base[s][0] for s in shared]):+.3f}"
                    dn = f"{np.mean([cur[s][1] - base[s][1] for s in shared]):+.4f}"
                    npair = str(len(shared))
                else:
                    dh = dn = npair = ""
                print(f"{animal:<7}{tag:<26}{hit:>7.3f}{dh:>8}{nll:>9.4f}{dn:>8}{len(cur):>4}{npair:>6}")
            print()


if __name__ == "__main__":
    main()
