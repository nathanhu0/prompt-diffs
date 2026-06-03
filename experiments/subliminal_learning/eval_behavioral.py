"""Behavioral evaluation of recovered prompts, soft-prompt-native.

Reimplements the subliminal-steering producer's two evals (see that repo's
code/src/eval_finetune.py and nathan_scripts/DATASETS_AND_EVALS.md §3) so the
recovered prompts are comparable to its `ft_eval.json` (base vs adapter):

  eval #1  hit rate       : sample RUNS completions, `label.lower() in resp.lower()`.
  eval #2  label log-prob : per-token mean logP(label) after the prompt; plotted
                            as exp(avg_log_likelihood).

Evaluates three conditions over the artifacts written by train.py:
  base        : Qwen2.5 base, no system prompt (sanity-checks against the
                producer's `base_model` numbers — should agree).
  soft        : base + the trained soft prompt z (spliced into inputs_embeds).
  verbalized  : base + the greedy-recovered text prompt as a system message.

Generation matches the reference exactly: do_sample, temperature=1.0,
pad_token_id=eos, and top_p/top_k left to the model's generation_config (the
reference overrides only temperature). Generation is chunked for memory —
samples are iid so batch size doesn't change results.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/eval_behavioral.py \\
    --run-dir /nlp/scr/nathu/latent_rewrite/subliminal_learning/steered_cat \\
    --topic cat
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.templates import forward_batch, sample_from_template
from optimize.template_factories.sysprompt import (
    build_sysprompt_template, build_sysprompt_gen_template)

from experiments.subliminal_learning.data import load_eval_spec

RUNS = 200            # completions per eval prompt
MAX_NEW = 100         # max new tokens per completion
TEMP = 1.0            # reference overrides temperature only (top_p/top_k from config)
GEN_BATCH = 100       # chunk size for sampling (memory only; results are iid)


@torch.no_grad()
def _generate(model, tokenizer, prompt, *, kind, z=None, system_text=None,
              n_learnable=None, system_template="{SOFT}"):
    """Return RUNS decoded completion strings for one eval prompt."""
    device = next(model.parameters()).device
    gen_kw = dict(max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP,
                  pad_token_id=tokenizer.eos_token_id)
    texts = []
    if kind == "soft":
        # Soft slot lives in the system position (where the model was trained
        # to read it); compose + generate via the shared primitive so soft and
        # text paths use byte-identical sampling (gen_kw).
        tmpl = build_sysprompt_gen_template(
            tokenizer, prompt, n_learnable, system_template=system_template)
        for start in range(0, RUNS, GEN_BATCH):
            b = min(GEN_BATCH, RUNS - start)
            out = sample_from_template(model, tmpl, z, n_samples=b, **gen_kw)
            texts += tokenizer.batch_decode(out, skip_special_tokens=True)
    else:  # text: base (system_text=None) or verbalized/decode (system_text=<prompt>)
        msgs = ([{"role": "system", "content": system_text}] if system_text else []) \
            + [{"role": "user", "content": prompt}]
        # Render to text then tokenize — apply_chat_template(return_tensors=) yields
        # a BatchEncoding (not a tensor) in transformers 5.8.1, so .shape/.expand
        # blow up. This mirrors eval_finetune.py's own generation idiom.
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        for start in range(0, RUNS, GEN_BATCH):
            b = min(GEN_BATCH, RUNS - start)
            enc = tokenizer([text] * b, return_tensors="pt", padding=True).to(device)
            L = enc["input_ids"].shape[1]
            out = model.generate(**enc, **gen_kw)
            texts += tokenizer.batch_decode(out[:, L:], skip_special_tokens=True)
    return texts


@torch.no_grad()
def _label_loglik(model, tokenizer, prompt, label, *, kind, z=None,
                  system_text=None, n_learnable=None, system_template="{SOFT}"):
    """Mean per-token logP(label) appended after one eval prompt (teacher-forced)."""
    device = next(model.parameters()).device
    if kind == "soft":
        tmpl, target_ids = build_sysprompt_template(
            tokenizer, prompt, label, n_learnable=n_learnable,
            system_template=system_template)
        out = forward_batch(model, [tmpl], z)
        logits = out["logits"][0]                    # (L, V)
        L = int(out["total_lens"][0])
        T = len(target_ids)
        # The composed sequence ends with the label tokens (positions L-T..L-1);
        # logits[L-T-1..L-2] predict them.
        logp = torch.log_softmax(logits[L - T - 1:L - 1].float(), dim=-1)
        label_ids = torch.tensor(target_ids, device=device)
    else:
        msgs = ([{"role": "system", "content": system_text}] if system_text else []) \
            + [{"role": "user", "content": prompt}]
        # Mirror the reference (eval_finetune.py:135-139): render to text, encode,
        # take label tokens by the prefix/full length difference (BPE-faithful).
        prefix = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(prefix + label, add_special_tokens=False)
        P, T = len(prefix_ids), len(full_ids) - len(prefix_ids)
        full = torch.tensor(full_ids, device=device).unsqueeze(0)
        logits = model(full).logits[0]               # (P+T, V)
        logp = torch.log_softmax(logits[P - 1:P + T - 1].float(), dim=-1)
        label_ids = torch.tensor(full_ids[P:], device=device)
    token_ll = logp[torch.arange(T), label_ids]
    return token_ll.mean().item()


def evaluate_condition(model, tokenizer, prompts, label, *, kind, z=None,
                       system_text=None, n_learnable=None, system_template="{SOFT}"):
    """Run both evals over all eval prompts for one condition."""
    total_hits, total, per_prompt_ll = 0, 0, []
    for i, prompt in enumerate(prompts):
        comps = _generate(model, tokenizer, prompt, kind=kind, z=z,
                          system_text=system_text, n_learnable=n_learnable,
                          system_template=system_template)
        hits = sum(label.lower() in c.lower() for c in comps)
        ll = _label_loglik(model, tokenizer, prompt, label, kind=kind, z=z,
                           system_text=system_text, n_learnable=n_learnable,
                           system_template=system_template)
        total_hits += hits
        total += len(comps)
        per_prompt_ll.append(ll)
        print(f"  [{kind}] prompt {i + 1}/{len(prompts)}: "
              f"hits {hits}/{len(comps)}, ll {ll:.3f}", flush=True)
    avg_ll = statistics.fmean(per_prompt_ll)
    print(f"[{kind}] hit_rate={total_hits / total:.4f} "
          f"avg_ll={avg_ll:.4f} geomean_prob={math.exp(avg_ll):.4f}")
    return {
        "hit_rate": total_hits / total,
        "total_hits": total_hits,
        "total_generations": total,
        "avg_log_likelihood": avg_ll,
        "log_likelihood_per_prompt": per_prompt_ll,
    }


def build_decodes(greedy_results):
    """Unique decoded prompts from greedy_recover's reps, each with its NLL
    scores + a winner flag. Deduped by exact text (reps often converge), so we
    pay one behavioral eval per distinct recovered prompt. Shared by run.py
    (in-process) and the standalone re-eval CLI."""
    best_text = greedy_results.get("best_text")
    seen, decodes = set(), []
    for i, rep in enumerate(greedy_results.get("greedy_reps", [])):
        text = rep["best_ever"]["text"]
        if text in seen:
            continue
        seen.add(text)
        decodes.append({
            "rep": i,
            "text": text,
            "full_val_nll": rep.get("best_full_val_kl"),  # NLL despite the key name
            "test_nll": rep.get("best_test_kl"),
            "is_winner": text == best_text,
        })
    return decodes


def run_behavioral_eval(model, tokenizer, *, topic, z=None, decodes=None,
                        n_learnable=None, system_template="{SOFT}",
                        condition_tag=None,
                        conditions=("base", "soft", "decodes")):
    """Run the producer's two behavioral evals for the requested conditions and
    return the ft_eval.json-shaped dict. Caller owns the model, the trained `z`
    (already on the model's device/dtype), the decoded prompts, and writing —
    so run.py can call this in-process (no model reload) and the CLI below can
    call it after loading a saved run.

      base    -> no system prompt
      soft    -> trained soft prompt z in the system slot (needs n_learnable)
      decodes -> EVERY recovered prompt as a text system prompt, not just the
                 NLL-winner; `decodes` is a list of dicts (see build_decodes)
                 with at least `text`. Each is evaluated independently so we
                 see behavioral transmission across decodes vs. the soft prompt.
    """
    label, prompts = load_eval_spec(topic)
    print(f"topic={topic} label={label!r} n_prompts={len(prompts)} runs={RUNS}")
    out = {"label": label, "topic": topic, "condition": condition_tag,
           "runs_per_prompt": RUNS, "max_tokens": MAX_NEW, "num_prompts": len(prompts)}
    if "base" in conditions:
        out["base_model"] = evaluate_condition(
            model, tokenizer, prompts, label, kind="text")
    if "soft" in conditions:
        out["soft_prompt"] = evaluate_condition(
            model, tokenizer, prompts, label, kind="soft", z=z,
            n_learnable=n_learnable, system_template=system_template)
    if "decodes" in conditions and decodes:
        out["decodes"] = []
        for d in decodes:
            print(f"  decode rep{d.get('rep')} "
                  f"(winner={d.get('is_winner')}): {d['text'][:60]!r}")
            res = evaluate_condition(
                model, tokenizer, prompts, label, kind="text",
                system_text=d["text"])
            out["decodes"].append({**d, **res})
    return out


def main():
    """Standalone re-eval of a saved run dir (run.py already evals in-process;
    use this to re-score off saved soft_z.pt / greedy_results.pt without
    retraining — e.g. after an eval-code change or to add conditions)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="run.py output dir (soft_z.pt, greedy_results.pt)")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--output", default=None, help="default <run-dir>/ft_eval.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--conditions", nargs="+",
                    default=["base", "soft", "decodes"])
    args = ap.parse_args()

    run = Path(args.run_dir)
    device = f"cuda:{args.gpu}"
    soft = torch.load(run / "soft_z.pt", map_location="cpu", weights_only=False)
    cfg = soft["config"]

    decodes = None
    if "decodes" in args.conditions:
        greedy = torch.load(run / "greedy_results.pt", map_location="cpu",
                            weights_only=False)
        decodes = build_decodes(greedy)
        print(f"{len(decodes)} unique decode(s) to evaluate")

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    z = soft["z"].to(device=device, dtype=embed_matrix.dtype)
    out = run_behavioral_eval(
        model, tokenizer, topic=args.topic, z=z, decodes=decodes,
        n_learnable=cfg["n_learnable"], system_template=cfg["system_template"],
        condition_tag=cfg["data"]["condition"], conditions=args.conditions)

    out_path = Path(args.output) if args.output else run / "ft_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
