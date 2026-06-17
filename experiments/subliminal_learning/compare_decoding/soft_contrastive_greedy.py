"""Soft-prompt analog of the ft−base contrastive greedy sweep.

ft−base contrast — (1+a)·logits_ft − a·logits_base — amplifies what the FINETUNE
adds over base (two models); pushed to high alpha + greedy it extracted an
explicit-cat prompt from weights that only saw numbers. The direct soft-prompt
twin is soft−empty contrast — (1+a)·logits_soft − a·logits_empty — amplifying
what the SOFT PROMPT adds over the empty system (one model, two prompt slots).
That contrast already exists (run.py / topic_alpha_sweep used it at a≤1); this
pushes the SAME trained z to the high-alpha grid {0.5,1,2,4} with greedy search,
so it lands on the master plot as a high-alpha "soft" family beside ft−base.

a=null is plain (non-contrastive) soft greedy, run here at the same n_reps /
max_steps / T=0.7 as the contrastive arms so the soft family is internally
consistent (the run's original ft_eval.json soft greedy used different settings).

Each rep's greedy winner is scored for catness; val NLL is greedy_recover's
full-val rescore. Writes soft_contrastive_greedy.json. Base model only.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compare_decoding/soft_contrastive_greedy.py \\
    --run-dir /nlp/scr/nathu/latent_rewrite/subliminal_learning/steered_cat_e4_lr1e-3 \\
    --gpu 0
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.models import load_frozen_lm
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.recover import greedy_recover

from experiments.subliminal_learning.data import load_sl_splits, load_eval_spec
from experiments.subliminal_learning.compare_decoding.sample_score_decodes import (
    cat_logprob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir with soft_z.pt")
    ap.add_argument("--alphas", nargs="+", default=["null", "0.5", "1.0", "2.0", "4.0"],
                    help="soft-vs-empty contrastive alphas; 'null' = plain soft greedy")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--n-eval-prompts", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="default <run-dir>/soft_contrastive_greedy.json")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    run = Path(args.run_dir)
    soft = torch.load(run / "soft_z.pt", map_location="cpu", weights_only=False)
    cfg = soft["config"]
    condition, topic = cfg["data"]["condition"], cfg["data"]["topic"]
    n_learnable = cfg["n_learnable"]

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    z = soft["z"].to(device=device, dtype=embed_matrix.dtype)   # the TRAINED soft prompt

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
          f"n_val={n_val} n_eval_prompts={len(eval_prompts)} z={tuple(z.shape)}",
          flush=True)

    decode_cfg = {**cfg["decode"], "temperature": float(args.temp)}
    greedy_cfg = dict(cfg["greedy"])
    greedy_cfg["n_reps"] = args.n_reps
    greedy_cfg["max_steps"] = args.max_steps

    alphas = [None if a == "null" else float(a) for a in args.alphas]
    records = []
    for alpha in alphas:
        atag = "plain" if alpha is None else f"α={alpha}"
        print(f"\n########## soft greedy {atag} (soft−empty contrast) ##########",
              flush=True)
        gc = {**greedy_cfg, "contrastive_alpha": alpha}
        # gen_model / neg_model default to the base model => single-model
        # soft-vs-empty contrast (z slot as positive, empty slot as negative).
        result = greedy_recover(
            z, nll_obj, model, tokenizer, embed_matrix,
            decode_cfg=decode_cfg, greedy_cfg=gc, seed=cfg["seed"])
        best_rep = result["best_rep"]
        for r, rep in enumerate(result["greedy_reps"]):
            text = rep["best_ever"]["text"]
            clp = cat_logprob(model, tokenizer, eval_prompts, label,
                              kind="text", system_text=text)
            rec = {
                "source": "soft",
                "contrastive_alpha": alpha,
                "rep": r,
                "is_winner": r == best_rep,
                "text": text,
                "nll_val": rep["best_full_val_kl"],
                "cat_logprob": clp,
            }
            records.append(rec)
            print(f"  [soft {atag}] rep{r}{' *' if rec['is_winner'] else '  '}: "
                  f"nll={rec['nll_val']:.4f} catlp={clp:.3f} "
                  f"(cat_prob={math.exp(clp):.4f}) :: {text[:70]!r}", flush=True)

    print("\n=== soft greedy summaries (winner per alpha) ===")
    for alpha in alphas:
        rs = [r for r in records if r["contrastive_alpha"] == alpha]
        win = next(r for r in rs if r["is_winner"])
        atag = "plain" if alpha is None else f"α={alpha}"
        best_clp = max(r["cat_logprob"] for r in rs)
        print(f"  soft {atag}: winner nll={win['nll_val']:.4f} "
              f"catlp={win['cat_logprob']:.3f} | best-catlp-any-rep={best_clp:.3f}")

    out_json = Path(args.output) if args.output else run / "soft_contrastive_greedy.json"
    out_json.write_text(json.dumps({
        "run_dir": str(run),
        "condition": condition,
        "topic": topic,
        "label": label,
        "temperature": float(args.temp),
        "alphas": args.alphas,
        "n_val": n_val,
        "n_eval_prompts": len(eval_prompts),
        "n_learnable": n_learnable,
        "greedy_cfg": greedy_cfg,
        "soft_greedy": records,
    }, indent=2))
    print(f"\nsaved → {out_json}")


if __name__ == "__main__":
    main()
