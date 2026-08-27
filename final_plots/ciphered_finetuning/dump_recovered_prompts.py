"""Dump every recovered prompt in the CMFT cipher ladder.

One entry per (model, cipher, arm, seed): the full SALVE prompt verbatim plus
that prompt's auditing rates over the 10 predictor+judge repetitions, so the
auditing rates can be read against the text that produced them. `expt` is the
recovery arm (cipher-trained M_base), `floor` the matched control (cipher-naive
initial model).

Run:
  uv run python final_plots/ciphered_finetuning/dump_recovered_prompts.py
"""
import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SWEEP = (REPO / "experiments/lls_traits/two_turn_legibility_eval"
         / "cmft_auditing_sweep.json")
SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
OUT = Path(__file__).parent / "cmft_recovered_prompts.md"

CIPHERS = [("walnut50", "Walnut"), ("endspeak", "EndSpeak"),
           ("ascii", "ASCII"), ("polybius", "Polybius")]
MODELS = [("qwen14b", "Qwen2.5-14B-Instruct"),
          ("gemma4_31b", "Gemma-4-31B-Instruct")]
SEEDS = [42, 43, 44, 45]
ARMS = [("per_seed", "expt", "Cipher-Trained Model"),
        ("per_seed_floor", "floor", "Initial Model (No Cipher Training)")]


def recovered(cond, cipher, model, seed):
    p = SALVE / f"ladder_{cond}_{cipher}_{model}_s{seed}" / "salve_beam.json"
    if not p.exists():
        return None
    return " ".join(json.loads(p.read_text())["best_text"].split())


def load_rates():
    """{(arm, cipher, model, seed): {k: rate}} plus the no-output counts."""
    rows = json.loads(SWEEP.read_text())["rows"]
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    no_output = collections.Counter()
    arms = {arm for arm, _, _ in ARMS}
    for row in rows:
        if row.get("arm") not in arms or row.get("seed") not in SEEDS:
            continue
        key = (row["arm"], row["cipher"], row["model"], row["seed"])
        if not row.get("predictions"):
            no_output[key] += 1
            continue
        for k, verdict in (row.get("pass_at") or {}).items():
            grouped[key][k].append(float(bool(verdict)))
    rates = {key: {k: float(np.mean(v)) for k, v in per_k.items()}
             for key, per_k in grouped.items()}
    return rates, no_output


def main():
    rates, no_output = load_rates()
    ground_truth = json.loads(SWEEP.read_text())["ground_truth"]
    out = [f"# Recovered prompts in the CMFT cipher ladder\n",
           f"*ground truth:* {ground_truth}\n",
           "Each prompt is the full `best_text` from its SALVE run. Rates are "
           "over 10 predictor+judge repetitions.\n"]
    for model, model_label in MODELS:
        out.append(f"\n# {model_label}\n")
        for cipher, cipher_label in CIPHERS:
            out.append(f"\n## {cipher_label}\n")
            for arm, cond, arm_label in ARMS:
                cell = [rates.get((arm, cipher, model, s), {}).get("5")
                        for s in SEEDS]
                valid = [v for v in cell if v is not None]
                mean = f"{np.mean(valid):.2f}" if valid else "n/a"
                out.append(f"\n### {arm_label} — mean pass@5 = {mean} "
                           f"(n={len(valid)} seeds)\n")
                for seed in SEEDS:
                    key = (arm, cipher, model, seed)
                    per_k = rates.get(key, {})
                    if per_k:
                        scores = "  ".join(f"pass@{k}={per_k[k]:.1f}"
                                           for k in ("1", "3", "5")
                                           if k in per_k)
                    else:
                        scores = (f"no auditor output "
                                  f"({no_output[key]} reps) — missing data")
                    text = recovered(cond, cipher, model, seed)
                    out.append(f"\n**seed {seed}** · {scores}\n")
                    out.append("\n~~~text\n" + (text or "(not on disk)")
                               + "\n~~~\n")
    OUT.write_text("".join(out))
    print("wrote", OUT, f"({len(''.join(out).splitlines())} lines)")


if __name__ == "__main__":
    main()
