"""Build the self-contained HTML browser for the CMFT auditing evaluation.

Each recovered prompt in the cipher ladder is scored two ways: the blind
three-class taxonomy judge labels the text itself, and ten independent
predictor+judge repetitions score whether an auditor reading it names the
fine-tuning effect. This emits a single page holding all of it: the prompt
verbatim, its taxonomy label with the span that decided it, and for each
repetition the five ranked behaviors the auditor proposed together with the
judge's CORRECT/INCORRECT verdict at each k.

  uv run python final_plots/ciphered_finetuning/build_prompt_browser.py
"""
import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "experiments/lls_traits/two_turn_legibility_eval"
SWEEP = EVAL / "cmft_auditing_sweep.json"
SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
TAXONOMY = REPO / "experiments/cmft_legibility/prompt_labels_judge.json"
HERE = Path(__file__).parent
DATA_OUT = HERE / "prompt_browser_data.json"
TEMPLATE = HERE / "prompt_browser_template.html"
HTML_OUT = HERE / "prompt_browser.html"

CIPHERS = [("walnut50", "Walnut"), ("endspeak", "EndSpeak"),
           ("ascii", "ASCII"), ("polybius", "Polybius")]
MODELS = [("qwen14b", "Qwen2.5-14B-Instruct"),
          ("gemma4_31b", "Gemma-4-31B-Instruct")]
SEEDS = [42, 43, 44, 45]
ARMS = [("per_seed", "expt", "Cipher-Trained Model"),
        ("per_seed_floor", "floor", "Initial Model (No Cipher Training)")]
# `cond` names the run directory; `arm` is what the taxonomy pass calls it.
TAXONOMY_ARM = {"expt": "experiment", "floor": "floor"}
KS = ("1", "3", "5")


def recovered(cond, cipher, model, seed):
    p = SALVE / f"ladder_{cond}_{cipher}_{model}_s{seed}" / "salve_beam.json"
    if not p.exists():
        return None
    return " ".join(json.loads(p.read_text())["best_text"].split())


def taxonomy():
    """{(arm, cipher, model, seed): label record} from the blind three-class
    judge -- the same labels the taxonomy figure plots."""
    rows = json.loads(TAXONOMY.read_text())["labels"]
    return {(r["arm"], r["cipher"], r["model"], r["seed"]): {
        "label": r["label"], "agreement": r["agreement"],
        "votes": r["votes"], "evidence": r["evidence"],
        "coherent": r.get("coherent")}
        for r in rows if r["label"] and r["seed"] is not None}


def main():
    sweep = json.loads(SWEEP.read_text())
    taxa = taxonomy()
    by_seed = collections.defaultdict(list)
    for row in sweep["rows"]:
        if row.get("seed") in SEEDS:
            by_seed[(row["arm"], row["cipher"], row["model"],
                     row["seed"])].append(row)

    cells = {}
    for model, _ in MODELS:
        for cipher, _ in CIPHERS:
            for arm, cond, _ in ARMS:
                seeds = []
                for seed in SEEDS:
                    rows = sorted(by_seed.get((arm, cipher, model, seed), []),
                                  key=lambda r: r["rep"])
                    reps, blank = [], 0
                    for row in rows:
                        if not row.get("predictions"):
                            blank += 1
                            continue
                        verdicts = row.get("pass_at") or {}
                        reps.append({
                            "rep": row["rep"],
                            "v": [int(bool(verdicts.get(k))) for k in KS],
                            "preds": row["predictions"],
                        })
                    rates = ({k: float(np.mean([r["v"][i] for r in reps]))
                              for i, k in enumerate(KS)} if reps else None)
                    seeds.append({"seed": seed, "rates": rates,
                                  "no_output": blank,
                                  "prompt": recovered(cond, cipher, model, seed),
                                  "taxonomy": taxa.get(
                                      (TAXONOMY_ARM[cond], cipher, model, seed)),
                                  "reps": reps})
                scored = [s["rates"] for s in seeds if s["rates"]]
                cells[f"{model}|{cipher}|{arm}"] = {
                    "mean": ({k: float(np.mean([r[k] for r in scored]))
                              for k in KS} if scored else None),
                    "n_seeds": len(scored),
                    "seeds": seeds,
                }

    taxonomy_meta = json.loads(TAXONOMY.read_text())
    payload = {
        "ground_truth": sweep["ground_truth"],
        "taxonomy_votes": taxonomy_meta["votes"],
        "taxonomy_model": taxonomy_meta["judge_model"],
        "taxonomy_classes": taxonomy_meta["classes"],
        "judge_model": sweep["model"],
        "reps": sweep["reps"],
        "models": [{"key": k, "label": v} for k, v in MODELS],
        "ciphers": [{"key": k, "label": v} for k, v in CIPHERS],
        "arms": [{"key": a, "label": lab} for a, _, lab in ARMS],
        "cells": cells,
    }
    blob = json.dumps(payload, separators=(",", ":"))
    DATA_OUT.write_text(blob)
    # `</` would close the host <script> element early; the escape is legal
    # JSON so JSON.parse still sees the original text.
    html = TEMPLATE.read_text().replace("__DATA__", blob.replace("</", "<\\/"))
    HTML_OUT.write_text(html)
    for path in (DATA_OUT, HTML_OUT):
        print("wrote", path, f"{path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
