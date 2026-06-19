"""Re-grade the ANIMAL behavior metric with word-boundary synonym matching
(core.subliminal.animals.hits_trait), replacing the old bare-substring hit test.

Motivation: SALVE's recovered dog prompt names "dogs/pooches" and the model
verbalizes the preference as "puppy", which `"dog" in completion` missed
(scored 0.015, below floor). Word-boundary synonyms fix that — and also kill
substring false-positives ("cat" in "category"). Applied UNIFORMLY to every
animal record + baselines so the bars stay comparable. Numbers untouched (their
behavior is a digit-constraint check, not keyword-based).

Non-destructive: old dict preserved under `behavior_substring`; new under
`behavior`. A small rollout sample is logged per record for eyeballing.

  ebatch regrade_behavior slconf/slconf40s_no32 \
    "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python final_experiments/optimizer_comparison/regrade_behavior.py"
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))              # for run_comparison
import run_comparison as rc
from optimize.config_utils import load_config
from core.subliminal import animals

SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison/sweep_main/prefill_t1")
CFG = Path(__file__).parent / "methods" / "salve.yaml"
N_LOG = 8

# One job per method (parallel). `method` selects which record(s) per animal dir
# to re-score. Exact-stem methods map to <method>.json; L-budgeted ones glob
# <method>_L*.json. Writes are method-disjoint across record files, so jobs are
# collision-free (rollouts/ mkdir is exist_ok).
METHOD = sys.argv[1]
EXACT = {"salve_naive", "salve_greedy", "salve_beam", "largo", "opro"}
ALL_METHODS = sorted(EXACT | {"gcg", "autodan", "gbda", "baselines"})
assert METHOD in ALL_METHODS, f"method must be one of {ALL_METHODS}, got {METHOD!r}"

cfg = load_config(str(CFG))
model, tok, _ = rc.load_frozen_lm(cfg["model"], device="cuda:0")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import json


def score(system_text, animal, roll_path, tag):
    """Re-score behavior; write the FULL completions to a sidecar so any future
    criterion can re-score without regenerating. Returns (metric_dict, sample)."""
    out = animals.behavior(model, tok, animal, system_text, return_completions=True)
    comps = out.pop("completions")
    roll_path.parent.mkdir(parents=True, exist_ok=True)
    roll_path.write_text(json.dumps(
        {"animal": animal, "tag": tag, "system_text": system_text,
         "n_questions": len(animals.EVAL_QUESTIONS), "n_samples": animals.EVAL_RUNS,
         "hit_rate": out["hit_rate"], "completions": comps}, indent=2))
    return out, comps[:N_LOG]


for ds in animals.ANIMALS:
    d = SCR / ds
    roll = d / "rollouts"
    if METHOD == "baselines":
        bpath = d / "baselines.json"
        b = json.loads(bpath.read_text())
        for key, sys_text in [("no_prompt", ""), ("true_pi", b["true_pi"]["text"])]:
            nb, samp = score(sys_text, ds, roll / f"baselines_{key}.json", f"baselines_{key}")
            b[key].setdefault("behavior_substring", b[key]["behavior"])
            b[key]["behavior"] = nb
            b[key]["behavior_rollout_sample"] = samp
        bpath.write_text(json.dumps(b, indent=2))
        print(f"[{ds}] baselines floor {b['no_prompt']['behavior_substring']['hit_rate']:.3f}"
              f"->{b['no_prompt']['behavior']['hit_rate']:.3f}  "
              f"canon {b['true_pi']['behavior_substring']['hit_rate']:.3f}"
              f"->{b['true_pi']['behavior']['hit_rate']:.3f}", flush=True)
        continue

    files = [d / f"{METHOD}.json"] if METHOD in EXACT else sorted(d.glob(f"{METHOD}_L*.json"))
    for p in files:
        if not p.exists():
            print(f"[{ds}] {METHOD}: no record, skipping", flush=True)
            continue
        rec = json.loads(p.read_text())
        if "best_text" not in rec:
            continue
        old = rec.get("behavior", {}).get("hit_rate")
        nb, samp = score(rec["best_text"], ds, roll / f"{p.stem}.json", p.stem)
        rec.setdefault("behavior_substring", rec["behavior"])
        rec["behavior"] = nb
        rec["behavior_rollout_sample"] = samp
        p.write_text(json.dumps(rec, indent=2))
        print(f"[{ds}] {p.stem:14s} {old:.3f} -> {nb['hit_rate']:.3f}", flush=True)

print(f"REGRADE COMPLETE ({METHOD})", flush=True)
