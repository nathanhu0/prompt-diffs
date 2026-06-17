"""Greedy-search version of the non-soft verbalization baselines.

`baseline_decodes.py` showed single-shot ft / ft-base-contrastive verbalizations
collapse to the base catness floor. This doubles down: run the SAME sentence-level
greedy search (`optimize.recover.greedy_recover`) but source candidate sentences
from the finetune (empty soft slot) and sweep contrastive steering — so we ask
whether *search + steering* can extract a cattier system prompt where single-shot
sampling could not. The greedy selector is val NLL on the number completions, so
it only commits to sentences that actually fit the data; high-alpha steering that
degenerates is simply rejected.

Sources (each a full greedy_recover; candidates verbalized through the listed
model over an empty soft slot, scored as a BASE system prompt):
  base_empty          : base, no steering. Floor control under search.
  finetune            : finetune M_ft, no steering.
  ft_base_contrastive : (1+a)*ft - a*base per step, sweeping a (--alphas).

Each rep's greedy winner is scored for catness (teacher-forced mean logP(label));
val NLL comes from greedy_recover's full-val rescore. Writes baseline_greedy.json.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compare_decoding/baseline_greedy.py \\
    --run-dir /nlp/scr/nathu/latent_rewrite/subliminal_learning/steered_cat_e4_lr1e-3 \\
    --gpu 0
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from peft import PeftModel

from core.models import load_frozen_lm
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.recover import greedy_recover

from experiments.subliminal_learning.data import (
    load_sl_splits, load_eval_spec, sl_adapter_path)
from experiments.subliminal_learning.compare_decoding.sample_score_decodes import (
    cat_logprob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir with soft_z.pt")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                    help="ft-vs-base contrastive alpha grid (greedy per alpha)")
    ap.add_argument("--temp", type=float, default=0.7, help="verbalize temperature")
    ap.add_argument("--n-reps", type=int, default=3, help="greedy reps per source")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override greedy max_steps (default: run's cfg)")
    ap.add_argument("--no-base", action="store_true", help="skip the base_empty control")
    ap.add_argument("--n-eval-prompts", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="default <run-dir>/baseline_greedy.json")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    run = Path(args.run_dir)
    soft = torch.load(run / "soft_z.pt", map_location="cpu", weights_only=False)
    cfg = soft["config"]
    condition, topic = cfg["data"]["condition"], cfg["data"]["topic"]
    n_learnable = cfg["n_learnable"]

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)

    adapter_path = sl_adapter_path(condition, topic)
    print(f"loading finetune adapter: {adapter_path}", flush=True)
    ft_base, _, _ = load_frozen_lm(cfg["model"], device=device)
    ft_model = PeftModel.from_pretrained(ft_base, str(adapter_path)).eval()

    # Greedy only touches val (+ optional test); build just the val split to skip
    # tokenizing the 10k-row train set. No test split => greedy_recover skips the
    # full-test rescore (we only need val NLL + catness here).
    splits = load_sl_splits(**cfg["data"], seed=cfg["seed"])
    build = lambda s, r, target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=n_learnable,
        system_template=cfg["system_template"], target_ids=target_ids)
    nll_obj = nll_objective_from_xys(
        model, tokenizer, {"val": splits["val"]}, build,
        system_template=cfg["system_template"])
    n_val = len(nll_obj.xy_by_split["val"])

    label, eval_prompts = load_eval_spec(topic)
    if args.n_eval_prompts is not None:
        eval_prompts = eval_prompts[:args.n_eval_prompts]
    print(f"condition={condition} topic={topic} label={label!r} "
          f"n_val={n_val} n_eval_prompts={len(eval_prompts)}", flush=True)

    decode_cfg = {**cfg["decode"], "temperature": float(args.temp)}
    greedy_cfg = dict(cfg["greedy"])
    greedy_cfg["n_reps"] = args.n_reps
    if args.max_steps is not None:
        greedy_cfg["max_steps"] = args.max_steps

    z_empty = torch.zeros(0, embed_matrix.shape[1],
                          device=device, dtype=embed_matrix.dtype)

    # (source tag, gen_model, neg_model, contrastive_alpha)
    jobs = []
    if not args.no_base:
        jobs.append(("base_empty", None, None, None))   # gen_model None -> base
    jobs.append(("finetune", ft_model, None, None))
    for a in args.alphas:
        jobs.append(("ft_base_contrastive", ft_model, model, float(a)))

    records = []
    for source, gen_model, neg_model, alpha in jobs:
        atag = "" if alpha is None else f" α={alpha}"
        print(f"\n########## greedy source={source}{atag} ##########", flush=True)
        gc = {**greedy_cfg, "contrastive_alpha": alpha}
        result = greedy_recover(
            z_empty, nll_obj, model, tokenizer, embed_matrix,
            decode_cfg=decode_cfg, greedy_cfg=gc, seed=cfg["seed"],
            gen_model=gen_model, neg_model=neg_model)
        best_rep = result["best_rep"]
        for r, rep in enumerate(result["greedy_reps"]):
            text = rep["best_ever"]["text"]
            clp = cat_logprob(model, tokenizer, eval_prompts, label,
                              kind="text", system_text=text)
            rec = {
                "source": source,
                "contrastive_alpha": alpha,
                "rep": r,
                "is_winner": r == best_rep,
                "text": text,
                "nll_val": rep["best_full_val_kl"],
                "cat_logprob": clp,
            }
            records.append(rec)
            print(f"  [{source}{atag}] rep{r}{' *' if rec['is_winner'] else '  '}: "
                  f"nll={rec['nll_val']:.4f} catlp={clp:.3f} "
                  f"(cat_prob={math.exp(clp):.4f}) :: {text[:70]!r}", flush=True)

    print("\n=== greedy baseline summaries (winner per source) ===")
    for source, _, _, alpha in jobs:
        rs = [r for r in records if r["source"] == source
              and r["contrastive_alpha"] == alpha]
        win = next(r for r in rs if r["is_winner"])
        atag = "" if alpha is None else f" α={alpha}"
        best_clp = max(r["cat_logprob"] for r in rs)
        print(f"  {source}{atag}: winner nll={win['nll_val']:.4f} "
              f"catlp={win['cat_logprob']:.3f} | best-catlp-any-rep={best_clp:.3f}")

    out_json = Path(args.output) if args.output else run / "baseline_greedy.json"
    out_json.write_text(json.dumps({
        "run_dir": str(run),
        "condition": condition,
        "topic": topic,
        "label": label,
        "adapter_path": str(adapter_path),
        "temperature": float(args.temp),
        "alphas": list(args.alphas),
        "n_val": n_val,
        "n_eval_prompts": len(eval_prompts),
        "n_learnable": n_learnable,
        "greedy_cfg": greedy_cfg,
        "greedy_baselines": records,
    }, indent=2))
    print(f"\nsaved → {out_json}")


if __name__ == "__main__":
    main()
