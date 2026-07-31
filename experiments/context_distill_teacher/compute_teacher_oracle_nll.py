"""Teacher-oracle NLL on the finetuned-teacher numbers data.

Scores the numbers splits under the TEACHER itself (base + context-distill
adapter, merged), conditioned exactly as at generation time ("You are a
helpful assistant." system). Since the data are t=1 teacher samples, this
estimates the teacher's entropy floor, and `NLL(prompt) - NLL(teacher)` is a
Monte-Carlo estimate of forward KL(teacher || base+prompt). Also scores the
plain base model under the same neutral system prompt as a reference.

Scoring mirrors final_experiments/optimizer_comparison/run_comparison.py
exactly (same load_splits args, same NLL objective/reduction), so numbers are
directly comparable to the recovered-prompt records.

Usage: PYTHONPATH=. uv run python experiments/context_distill_teacher/compute_teacher_oracle_nll.py --animal cat --teacher-lr 0.001
"""
import argparse
import json
from pathlib import Path

from core.models import load_frozen_lm
from core.subliminal.data import load_splits
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template

TEACHER_ROOT = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers")
NEUTRAL_SYSTEM = "You are a helpful assistant."   # generation-time conditioning


def build_objective(model, tokenizer, xy):
    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=128, system_template="{SOFT}",
        assistant_prefill=prefill, target_ids=target_ids)
    return nll_objective_from_xys(model, tokenizer, xy, build,
                                  system_template="{SOFT}")


def score_all(objective, text, mb=16):
    return {split: float(objective.hard_loss(text, split, mini_batch_size=mb))
            for split in ("train", "val", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True)
    ap.add_argument("--teacher-lr", default="0.001")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-method", default="context_distill_aggressive")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    adapter = TEACHER_ROOT / short / args.animal / f"lr{args.teacher_lr}" / "adapter"
    assert adapter.exists(), adapter
    device = f"cuda:{args.gpu}"

    xy = load_splits(args.animal, 10000, 500, 1500, prefill=None, seed=42,
                     model=args.model, method=args.data_method)
    for split, pairs in xy.items():
        print(f"  {split}: {len(pairs)} pairs")

    rec = {"animal": args.animal, "teacher_lr": args.teacher_lr,
           "data_method": args.data_method, "system": NEUTRAL_SYSTEM,
           "data_seed": 42}

    teacher, tok, _ = load_frozen_lm(args.model, device=device,
                                     adapter_path=str(adapter))
    rec["teacher"] = score_all(build_objective(teacher, tok, xy), NEUTRAL_SYSTEM)
    print(f"[teacher+neutral] {rec['teacher']}", flush=True)
    del teacher
    import torch; torch.cuda.empty_cache()

    base, tok, _ = load_frozen_lm(args.model, device=device)
    rec["base_neutral"] = score_all(build_objective(base, tok, xy), NEUTRAL_SYSTEM)
    print(f"[base+neutral]    {rec['base_neutral']}", flush=True)

    out = (TEACHER_ROOT / "recovery" / short / "oracle_nll" /
           f"{args.data_method}_{args.animal}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
