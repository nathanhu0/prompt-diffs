"""Readout for trained cat+dog mixtures: what does each of the K prompts DO?

Two stages (both per-prompt, both grade cat AND dog hit rates):
  --stage soft  : behavior_soft on each z directly (pre-verbalization; cheap).
  --stage beam  : verbalize each surviving prompt via beam_recover, scoring
                  candidates on that prompt's OWN train cluster (train-split
                  examples it wins under pure argmin), then behavioral eval of
                  the recovered text. Heavy; restrict via --prompts.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/readout_cat_dog.py \\
    --name bias_decay --stage soft --gpu 0
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal.animals import behavior, behavior_soft, hits_trait
from optimize.mixture import per_example_nll
from optimize.objectives.nll import nll_objective_from_xys
from optimize.recover import beam_recover
from optimize.template_factories.sysprompt import build_sysprompt_template

from experiments.mixture_soft_prompts.train_cat_dog import (
    MODEL, OUT_ROOT, SCHRODI_DIR, SECONDARIES, load_labeled_mix)

BEAM_DECODE = {"pool": "system_top4", "persona_prefix": "", "temperature": 0.7}
BEAM_CFG = {"n_beams": 4, "branching": 16, "max_iters": 12,
            "max_new_tokens": 32, "tol": float("inf"), "alphas": [None],
            "n_val": 256, "max_tokens": 256, "mini_batch_size": 24}
MIN_VAL_LOAD = 25   # prompts below this val load are idle; skip their beam


def both_rates(comps):
    """Hit rate per animal over one set of completions. Grades ALL four
    animals (CPU word-match, free) so any primary/secondary pair — and
    cross-trait leakage — reads from the same record."""
    return {a: sum(hits_trait(c, a) for c in comps) / len(comps)
            for a in ("cat", "dog", "eagle", "owl")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--stage", choices=["soft", "beam"], required=True)
    p.add_argument("--prompts", type=int, nargs="*", default=None,
                   help="beam stage: restrict to these prompt indices")
    p.add_argument("--branching", type=int, default=BEAM_CFG["branching"])
    p.add_argument("--max-iters", type=int, default=BEAM_CFG["max_iters"])
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    BEAM_CFG.update(branching=args.branching, max_iters=args.max_iters)

    run_dir = OUT_ROOT / args.name
    d = torch.load(run_dir / "mixture.pt", map_location="cpu",
                   weights_only=False)
    device = f"cuda:{args.gpu}"
    n_learnable = d["args"]["n_learnable"]
    k = d["config"]["k"]

    model, tokenizer, embed_matrix = load_frozen_lm(MODEL, device=device)
    z_list_k = [z.to(device) for z in d["best_z"]]

    secondary = d["args"].get("secondary", "dog")
    primary = d["args"].get("primary", "cat")
    sources = [(SCHRODI_DIR / f"filtered_{primary}.jsonl", 0),
               (SECONDARIES[secondary], 1)]
    xy, labels = load_labeled_mix(n_train=d["args"]["n_train"],
                                  cat_frac=d["args"].get("cat_frac", 0.5),
                                  sources=sources)
    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=n_learnable,
        assistant_prefill=prefill, target_ids=target_ids)
    objective = nll_objective_from_xys(model, tokenizer, xy, build)

    # cluster stats under best_z (pure argmin) on val + train
    stats = {}
    for split in ("val", "train"):
        sums_k = [per_example_nll(objective, [z], split)[0] for z in z_list_k]
        counts = per_example_nll(objective, [z_list_k[0]], split)[1]
        means = torch.stack(sums_k, dim=1) / counts.unsqueeze(1)
        stats[split] = means.argmin(dim=1)
    val_loads = torch.bincount(stats["val"], minlength=k).tolist()
    print(f"val loads under best_z: {val_loads}", flush=True)
    for j in range(k):
        vl = [labels["val"][i] for i in range(len(labels["val"]))
              if stats["val"][i] == j]
        print(f"  prompt {j}: val {val_loads[j]} "
              f"({vl.count(0)} cat / {vl.count(1)} dog)", flush=True)

    tag = ("_p" + "-".join(map(str, args.prompts))
           if args.prompts is not None else "")
    out_path = run_dir / f"readout_{args.stage}{tag}.pt"
    results = {"val_loads": val_loads, "prompts": {}}

    if args.stage == "soft":
        base = behavior(model, tokenizer, "cat", "", return_completions=True)
        results["no_prompt"] = both_rates(base.pop("completions"))
        print(f"no-prompt base rates: {results['no_prompt']}", flush=True)
        for j in range(k):
            out = behavior_soft(model, tokenizer, "cat", z_list_k[j],
                                n_learnable=n_learnable,
                                return_completions=True)
            rates = both_rates(out.pop("completions"))
            results["prompts"][j] = {"rates": rates, "val_load": val_loads[j]}
            print(f"prompt {j} (val load {val_loads[j]}): "
                  + " ".join(f"{a}={r:.3f}" for a, r in rates.items()),
                  flush=True)

    else:  # beam verbalization on each prompt's own train cluster
        todo = args.prompts if args.prompts is not None else [
            j for j in range(k) if val_loads[j] >= MIN_VAL_LOAD]
        full_train = list(objective.examples_by_split["train"])
        full_train_xy = list(objective.xy_by_split["train"])
        for j in todo:
            idx = [i for i in range(len(full_train))
                   if stats["train"][i] == j]
            # zero/tiny-load members have no meaningful cluster; score the
            # candidate texts on the FULL train split instead (flagged in
            # the result record via cluster_size).
            if len(idx) < 32:
                print(f"\n=== beam readout prompt {j}: cluster only "
                      f"{len(idx)} examples -> scoring on full train ===",
                      flush=True)
                idx = list(range(len(full_train)))
            else:
                print(f"\n=== beam readout prompt {j}: train cluster "
                      f"{len(idx)} examples ===", flush=True)
            if len(idx) < BEAM_CFG["n_val"]:
                print(f"  cluster smaller than n_val, using all {len(idx)}",
                      flush=True)
            objective.examples_by_split["train"] = [full_train[i] for i in idx]
            objective.xy_by_split["train"] = [full_train_xy[i] for i in idx]
            try:
                res = beam_recover(
                    z_list_k[j], objective, model, tokenizer, embed_matrix,
                    decode_cfg=BEAM_DECODE,
                    beam_cfg={**BEAM_CFG,
                              "n_val": min(BEAM_CFG["n_val"], len(idx))},
                    seed=42, select_split="train")
            finally:
                objective.examples_by_split["train"] = full_train
                objective.xy_by_split["train"] = full_train_xy
            beh = behavior(model, tokenizer, "cat", res["best_text"],
                           return_completions=True)
            rates = both_rates(beh.pop("completions"))
            results["prompts"][j] = {
                "best_text": res["best_text"],
                "best_sel_score": res["best_sel_score"],
                "cluster_size": len(idx), "rates": rates,
                "val_load": val_loads[j],
            }
            print(f"prompt {j}: "
                  + " ".join(f"{a}={r:.3f}" for a, r in rates.items())
                  + f"\n  text: {res['best_text'][:300]}", flush=True)
            torch.save(results, out_path)  # checkpoint per prompt

    torch.save(results, out_path)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
