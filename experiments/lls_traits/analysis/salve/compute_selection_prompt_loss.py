"""Compute the DPO loss (beta0.08, val) of the DATA SELECTION PROMPT (the LLS
target trait prompt) for one base model, both traits — the reference the grids'
top (DPO-loss) row is missing. Replicates the SALVE run split exactly (shuffle
seed 42, n_train 25000, n_val 500) so the number is on the same footing as
beam_results' baseline_full / best_full_val. Writes per-model json (no cross-job
race).

  ebatch selloss slconf/slconf_loprio "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/analysis/compute_selection_prompt_loss.py --model qwen7b"
"""
import argparse
import json
import random
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from core.models import load_frozen_lm
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.template_factories.sysprompt import build_sysprompt_template

SEL = Path("/nlp/scr/nathu/logit-linear-selection")
DATA = {"sycophancy": SEL / "You_are_extremely_sycophantic_44eb4c69_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset.json",
        "evil": SEL / "You_are_an_evil_misaligned_AI_c7bad2f2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset.json"}
CANON = Path(__file__).parent / "canonical_prompts"
HF = {"olmo1b": "allenai/OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
      "llama8b": "meta-llama/Llama-3.1-8B-Instruct", "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
      "rnj1": "EssentialAI/rnj-1-instruct"}
OUT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds/selection_dpo_loss")
SEED, N_TRAIN, N_VAL, NL = 42, 25000, 500, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, _ = load_frozen_lm(HF[args.model], device="cuda:0")
    build = lambda p, r, target_ids=None: build_sysprompt_template(
        tok, p, r, n_learnable=NL, system_template="{SOFT}", target_ids=target_ids)
    res = {}
    for trait, path in DATA.items():
        triples = [tuple(t) for t in json.loads(path.read_text())]
        random.Random(SEED).shuffle(triples)
        splits = {"train": triples[:N_TRAIN],
                  "val": triples[N_TRAIN:N_TRAIN + N_VAL], "test": []}
        obj = dpo_objective_from_triples(model, tok, splits, build, beta=0.08,
                                         system_template="{SOFT}", ref_mini_batch_size=8)
        canon = (CANON / f"{trait}.txt").read_text().strip()
        sel = obj.hard_loss(canon, "val", mini_batch_size=8)
        base = obj.hard_loss("", "val", mini_batch_size=8)
        res[trait] = {"selection_loss": sel, "baseline_loss": base}
        print(f"  {args.model} {trait}: selection={sel:.4f}  baseline={base:.4f}",
              flush=True)
    (OUT / f"{args.model}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / (args.model + '.json')}", flush=True)


if __name__ == "__main__":
    main()
