"""Behavioral eval of beam-recovered prompts.

For each `beam_recovery/<tag>__<alpha>__<tol>.pt` (output of run_beam.py), score
its recovered `best_text` on the trait behavior by reusing the family's existing
behavioral harness, and write `<stem>.eval.json` next to the .pt. NLL/loss and
the empty-prompt baseline come straight from the .pt; only the behavior number
needs a forward pass, so this is the one GPU step between recovery and plotting.

One family per invocation (loads the base model once): `--family sl` evaluates
the SL/NLL prompts (Qwen2.5-7B, hit-rate + label-loglik), `--family dpo` the DPO
prompts (OLMo-2-7B, trait rate). Idempotent (skips existing eval jsons unless
--overwrite) and shardable (`--shard i/N`) so the post-sweep eval can fan across
freed GPUs.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/beam_recovery/eval_recovered.py --family sl --shard 0/4 --gpu 0
"""
import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import torch

from core.models import load_frozen_lm

BEAM_DIR = Path("/nlp/scr/nathu/latent_rewrite/beam_recovery")

# Whole-word/native forms used to ask "does the recovered prompt name the trait?"
# (the trait-legibility signal: NLL-good prompts are often trait-illegible).
DPO_TRAIT_WORDS = {
    "cats": ["cat", "cats"], "dogs": ["dog", "dogs"], "owls": ["owl", "owls"],
    "spanish": ["spanish", "español", "espanol", "castellano"],
    "german": ["german", "deutsch", "deutsche", "deutschen"],
    "chinese": ["chinese", "mandarin", "中文", "汉语", "中国"],
}


def names_trait(text: str, words: list[str]) -> bool:
    import re
    low = (text or "").lower()
    for w in words:
        wl = w.lower()                 # lowercase the needle too (SL labels are capitalized)
        if (wl.isascii() and re.search(r"\b" + re.escape(wl) + r"\b", low)) or \
           (not wl.isascii() and wl in low):
            return True
    return False


def parse_cell(stem: str) -> tuple[str, str, str]:
    """`<tag>__<alpha>__<tol>` -> (tag, alpha_tag, tol_tag)."""
    parts = stem.split("__")
    parts += ["?"] * (3 - len(parts))
    return parts[0], parts[1], parts[2]


def eval_one(family, model, tok, best_text, cfg):
    """Return the behavior-metric dict for one recovered prompt (kind='text',
    system_text=best_text — matches how decodes are scored in run_behavioral_eval
    and the hard_loss the search optimized, since system_template is '{SOFT}')."""
    if family == "sl":
        from experiments.subliminal_learning.eval_behavioral import evaluate_condition
        from experiments.subliminal_learning.data import load_eval_spec
        label, prompts = load_eval_spec(cfg["data"]["topic"])
        res = evaluate_condition(model, tok, prompts, label, kind="text",
                                 system_text=best_text or None)
        return {"behavior": res["hit_rate"],
                "avg_log_likelihood": res["avg_log_likelihood"],
                "geomean_prob": math.exp(res["avg_log_likelihood"]),
                "trait_word": label, "names_trait": names_trait(best_text, [label]),
                "n_eval_prompts": len(prompts)}
    from experiments.subliminal_dpo.eval_behavioral import evaluate_condition, make_scorer
    from experiments.subliminal_dpo.data import load_eval_spec
    trait = cfg["data"]["trait"]
    spec = load_eval_spec(trait)
    res = evaluate_condition(model, tok, spec["eval_prompts"], make_scorer(spec),
                             kind="text", system_text=best_text or None)
    return {"behavior": res["rate"], "trait_word": trait,
            "names_trait": names_trait(best_text, DPO_TRAIT_WORDS[trait]),
            "n_eval_prompts": len(spec["eval_prompts"])}


def shard_files(args, i, n):
    """Sorted family-filtered .pt for this shard (filename family-filter avoids
    torch.load on the other family). Re-globbed each pass so --watch sees new ones."""
    is_dpo = lambda p: Path(p).name.startswith("dpo_")
    files = sorted(f for f in glob.glob(args.glob) if is_dpo(f) == (args.family == "dpo"))
    return files[i::n]


def eval_record(family, model, tok, res, cfg, stem):
    """Behavioral eval of a recovery result + the assembled .eval.json record.
    Shared by the standalone eval (process_pass) and run_beam.py's inline eval."""
    tag, atag, ttag = parse_cell(stem)
    beh = eval_one(family, model, tok, res["best_text"], cfg)
    return {"tag": tag, "alpha_tag": atag, "tol_tag": ttag, "family": family,
            "topic": cfg["data"].get("topic"), "trait": cfg["data"].get("trait"),
            "condition": cfg["data"].get("condition"),
            "nll": res["best_full_val"], "baseline_nll": res["baseline_full"],
            "best_sel_score": res["best_sel_score"],
            "n_iters": res["n_iters"], "n_score": res["n_score"],
            "system_template": cfg.get("system_template"),
            "best_text": res["best_text"], **beh}


def process_pass(args, i, n, state):
    """Evaluate any not-yet-scored .pt in this shard; reuse the warm model."""
    for f in shard_files(args, i, n):
        stem = Path(f).stem
        outp = Path(f).with_name(stem + ".eval.json")
        if outp.exists() and not args.overwrite:
            continue
        res = torch.load(f, map_location="cpu", weights_only=False)
        cfg = res["config"]
        if state["model"] is None:              # lazy: one model load for the shard
            state["model"], state["tok"], _ = load_frozen_lm(
                cfg["model"], device=f"cuda:{args.gpu}")
        print(f"\n=== {stem} (nll={res['best_full_val']:.4f} "
              f"baseline={res['baseline_full']:.4f}) ===", flush=True)
        rec = eval_record(args.family, state["model"], state["tok"], res, cfg, stem)
        outp.write_text(json.dumps(rec, indent=2))
        state["n"] += 1
        print(f"behavior={rec['behavior']:.4f}  saved -> {outp.name}  (#{state['n']})",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["sl", "dpo"])
    ap.add_argument("--glob", default=str(BEAM_DIR / "*.pt"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/N stride split of the file list")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--watch", action="store_true",
                    help="keep model warm and poll for newly-finished .pt (stream as-we-go)")
    ap.add_argument("--poll", type=int, default=180, help="watch poll interval (s)")
    ap.add_argument("--stop-file", default=None,
                    help="when this file appears, do one final pass and exit")
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))
    print(f"family={args.family} shard={args.shard} watch={args.watch}", flush=True)

    state = {"model": None, "tok": None, "n": 0}
    while True:
        process_pass(args, i, n, state)
        if not args.watch:
            break
        if args.stop_file and os.path.exists(args.stop_file):
            process_pass(args, i, n, state)     # final catch-up after producer stops
            break
        time.sleep(args.poll)
    print(f"\n[done] family={args.family} shard={args.shard}: evaluated {state['n']}",
          flush=True)


if __name__ == "__main__":
    main()
