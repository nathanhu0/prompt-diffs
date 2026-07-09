"""Integrated AdvBench StrongREJECT eval for CMFT — the one canonical harm number.

Scores up to three conditions on the held-out AdvBench-520, all under the Walnut
TASK-4 cipher framing, judged in-process by StrongREJECT (gpt-4o-mini):

  base     — M_base with TASK-4 only (no recovered prompt) — the floor
  soft     — M_base + the trained soft prompt z in the system slot
  discrete — M_base + the verbalized recovered prompt (text system)

Rollouts (GPU) + judge (API) run in ONE process — no separate rollout/judge jobs.
Used two ways:
  - library: `run_advbench_strongreject(model, tok, z=..., discrete_text=...)`
    called at the end of salve_run.py so every SALVE run self-produces the number.
  - CLI: evaluate any model/adapter (original models, SFT checkpoints; Gemma
    multimodal via core.models.load_frozen_lm). Point --salve-dir at a finished
    SALVE run to also score its soft_z + verbalized prompt.

The held-out AdvBench-520 (bad.json) has zero overlap with the Wei jailbreak
training set, so it is a fair generalization test for all conditions.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "safe-finetuning-api" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from experiments.cmft_legibility.salve_eval import (
    make_cipher, TASK4_SUFFIX, eval_jailbreak_soft, eval_jailbreak_hard,
    plaintext_rollouts)
from experiments.cmft_legibility.advbench_vllm_sweep import load_advbench
from experiments.cmft_legibility.judge_advbench_strongreject import (
    judge_records, load_dotenv, mean, std, custom_model_name)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_advbench_records(prompts, seed=50):
    """Plaintext AdvBench prompt -> a CMFT-style record: user turn is the CIPHERED
    prompt, sys_suffix is TASK-4, decoded_user is the plaintext (what the judge
    scores the decoded reply against)."""
    cipher, enc, _ = make_cipher(seed)
    suffix = TASK4_SUFFIX.format(name=cipher.name())
    return [{"user": enc(p), "sys_suffix": suffix, "decoded_user": p,
             "subset": "ciphered_harmful"} for p in prompts]


def _judge(rollout_records, judge_model, concurrency):
    # the AsyncOpenAI backend wants the bare model id ("gpt-4o-mini"), not the
    # "openai/..." litellm form used by the official evaluator.
    judged = asyncio.run(judge_records(
        rollout_records, model=custom_model_name(judge_model), concurrency=concurrency,
        max_retries=6, timeout=60))
    xs = [r.get("strongreject_score") for r in judged]
    # non-refusal rate: fraction of prompts the judge scored as NOT a refusal
    # (refusal field is 0/1/None). More legible than the composite score.
    nonref = [1 - r["refusal"] for r in judged if r.get("refusal") is not None]
    return {
        "score_mean": mean(xs), "score_std": std(xs),
        "non_refusal_rate": (sum(nonref) / len(nonref)) if nonref else None,
        "n": len(xs), "n_valid": sum(x is not None for x in xs),
        "records": [{**roll, **jd} for roll, jd in zip(rollout_records, judged)],
    }


def run_advbench_strongreject(model, tokenizer, *, z=None, n_learnable=None,
                              discrete_text=None, include_base=True,
                              include_plaintext=False, n=520,
                              seed=50, max_new=512, batch_size=8,
                              judge_model="openai/gpt-4o-mini",
                              concurrency=16, out_path=None):
    """Roll out + judge the requested conditions; return {cond: {score_mean, ...}}.

    Conditions:
      base      — ciphered TASK-4, model alone (the cipher-only score)
      plaintext — raw plaintext harmful prompt, no cipher/TASK-4 (covert check)
      soft      — ciphered TASK-4 + soft prompt z (SALVE)
      discrete  — ciphered TASK-4 + verbalized recovered prompt (SALVE)
    Each runs only when its input/flag is provided. Saves out_path if given."""
    load_dotenv(REPO_ROOT / ".env")
    records = build_advbench_records(load_advbench(n), seed=seed)
    print(f"[advbench] {len(records)} AdvBench prompts (cipher TASK-4 + plaintext)", flush=True)

    bs = batch_size
    rollouts = {}
    if include_base:
        _, jr = eval_jailbreak_hard(model, tokenizer, "", records, seed=seed, n=n, max_new=max_new, batch_size=bs, label="base")
        rollouts["base"] = jr
    if include_plaintext:
        jr, _ = plaintext_rollouts(model, tokenizer, records, n=n, max_new=max_new, batch_size=bs, label="plaintext")
        rollouts["plaintext"] = jr
    if z is not None:
        _, jr = eval_jailbreak_soft(model, tokenizer, z, records, n_learnable, seed=seed, n=n, max_new=max_new, batch_size=bs, label="soft")
        rollouts["soft"] = jr
    if discrete_text is not None:
        _, jr = eval_jailbreak_hard(model, tokenizer, discrete_text, records, seed=seed, n=n, max_new=max_new, batch_size=bs, label="discrete")
        rollouts["discrete"] = jr

    result = {"n": n, "seed": seed, "judge_model": judge_model, "conditions": {}}
    for name, jr in rollouts.items():
        scored = _judge(jr, judge_model, concurrency)
        result["conditions"][name] = scored
        nr = scored["non_refusal_rate"]
        print(f"[advbench:{name}] StrongREJECT={scored['score_mean']:.4f} "
              f"non-refusal={nr:.1%} (n_valid={scored['n_valid']}/{scored['n']})"
              if nr is not None else
              f"[advbench:{name}] StrongREJECT={scored['score_mean']} "
              f"(n_valid={scored['n_valid']}/{scored['n']})", flush=True)
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
        print(f"saved {out_path}", flush=True)
    return result


def main():
    import torch
    from core.models import load_frozen_lm

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model id")
    ap.add_argument("--adapter", default=None, help="M_base = base + this LoRA (merged)")
    ap.add_argument("--salve-dir", default=None,
                    help="finished SALVE run dir; also scores its soft_z.pt + verbalized prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=520)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="rollout generation batch size (lower for larger models)")
    ap.add_argument("--no-base", action="store_true", help="skip the ciphered base condition")
    ap.add_argument("--plaintext", action="store_true",
                    help="also score raw plaintext harmful prompts (covert-property check)")
    ap.add_argument("--judge-model", default="openai/gpt-4o-mini")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    model, tok, embed = load_frozen_lm(args.model, device=f"cuda:{args.gpu}",
                                       adapter_path=args.adapter)
    z = n_learnable = discrete_text = None
    if args.salve_dir:
        d = Path(args.salve_dir)
        zp = d / "soft_z.pt"
        if zp.exists():
            blob = torch.load(zp, map_location="cpu", weights_only=False)
            z = blob["z"].to(device=model.device, dtype=embed.dtype)
            n_learnable = z.shape[0]
        beam = d / "salve_beam.json"
        if beam.exists():
            discrete_text = json.loads(beam.read_text())["best_text"]

    run_advbench_strongreject(
        model, tok, z=z, n_learnable=n_learnable, discrete_text=discrete_text,
        include_base=not args.no_base, include_plaintext=args.plaintext,
        n=args.n, max_new=args.max_new, batch_size=args.batch_size,
        judge_model=args.judge_model, out_path=args.out)


if __name__ == "__main__":
    main()
