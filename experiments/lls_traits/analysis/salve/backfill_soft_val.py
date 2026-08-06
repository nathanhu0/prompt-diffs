"""Recompute the SOFT-prompt DPO val loss for runs that predate soft_val caching.

Older SALVE runs saved soft_z.pt as {z, config} with no `soft_val`, so the soft
stage is a blank column in the grids — which is exactly the column we want for
picking a per-model learning rate. The z tensor is there, so the number is
recoverable with one forward pass over the val split; no retraining.

Split and objective replicate the SALVE run and compute_selection_prompt_loss.py
exactly (shuffle seed 42, n_train 25000, n_val 500, beta 0.08, n_learnable 256,
system_template "{SOFT}"), so the result is comparable to beam_results'
baseline_full / best_full_val and to the selection-prompt loss.

Writes a SIDECAR soft_val.json per run rather than mutating soft_z.pt.

  ebatch softval_llama8b slconf/slconf_jag_hi "PYTHONUNBUFFERED=1 PYTHONPATH=. \
    uv run python experiments/lls_traits/analysis/salve/backfill_soft_val.py --model llama8b"
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from core.models import load_frozen_lm
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.template_factories.sysprompt import build_sysprompt_template

SEL = Path("/nlp/scr/nathu/logit-linear-selection")
DATA = SEL / ("You_are_extremely_sycophantic_44eb4c69_OLMo-2-0425-1B-Instruct"
              "_trunc20_q0.1/datasets/preference_dataset.json")
SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
HF = {"olmo1b": "allenai/OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
      "llama8b": "meta-llama/Llama-3.1-8B-Instruct", "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
      "rnj1": "EssentialAI/rnj-1-instruct"}
SEED, N_TRAIN, N_VAL, NL = 42, 25000, 500, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    ap.add_argument("--mini-batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    todo = []
    for d in sorted(glob.glob(str(SV / f"salve_sycophancy_{args.model}_b0.08_*"))):
        if d.endswith("_n256") or not os.path.exists(f"{d}/soft_z.pt"):
            continue
        if os.path.exists(f"{d}/soft_val.json") and not args.overwrite:
            continue
        z = torch.load(f"{d}/soft_z.pt", map_location="cpu", weights_only=False)
        if isinstance(z, dict) and z.get("soft_val") is not None and not args.overwrite:
            continue                                   # already cached in-file
        todo.append(d)
    if not todo:
        print(f"{args.model}: nothing to backfill")
        return
    print(f"{args.model}: {len(todo)} runs to backfill", flush=True)

    model, tok, _ = load_frozen_lm(HF[args.model], device="cuda:0")
    build = lambda p, r, target_ids=None: build_sysprompt_template(
        tok, p, r, n_learnable=NL, system_template="{SOFT}", target_ids=target_ids)
    triples = [tuple(t) for t in json.loads(DATA.read_text())]
    random.Random(SEED).shuffle(triples)
    splits = {"train": triples[:N_TRAIN],
              "val": triples[N_TRAIN:N_TRAIN + N_VAL], "test": []}
    obj = dpo_objective_from_triples(model, tok, splits, build, beta=0.08,
                                     system_template="{SOFT}", ref_mini_batch_size=8)

    for d in todo:
        z = torch.load(f"{d}/soft_z.pt", map_location="cpu", weights_only=False)
        zt = (z["z"] if isinstance(z, dict) else z).to(model.device, dtype=model.dtype)
        with torch.no_grad():
            val = obj.loss([zt], "val", mini_batch_size=args.mini_batch_size).item()
        Path(f"{d}/soft_val.json").write_text(json.dumps({"soft_val": val}, indent=1))
        print(f"  {os.path.basename(d):<50} soft_val {val:.4f}", flush=True)


if __name__ == "__main__":
    main()
