"""Behavioral evaluation of recovered prompts, soft-prompt-native, following the
LLS eval protocol (DATA_AND_EVAL_SPEC.md §4) so our numbers sit alongside the
producer's base / skyline / LLS-FT.

Metric (per eval prompt: generate RUNS responses, score each, average):
  - animal traits  : per-response 1.0 if the target word appears (word-boundary
                     + plural via `inflect`), else 0.0  → rate = hits/RUNS.
  - language traits: per-response target-language fraction in [0,1] (fastText
                     lid.176, degeneration-corrected) → rate = mean fraction.
The single reported number per condition = mean rate over the 10 eval prompts.

Conditions:
  base       : empty system prompt (LLS §4.2 base; OLMo has a system role).
  skyline    : base + the trait system prompt (the ceiling reference).
  soft       : base + the trained soft prompt z (spliced into inputs_embeds).
  decodes    : base + each greedy-recovered text prompt as a system message.

Sampling matches the reference (§4.3): do_sample, temperature=1.0,
max_new_tokens=200, pad_token_id=eos; top_p/top_k inherited from the model's
generation_config. Generation is chunked for memory (samples are iid).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_dpo/eval_behavioral.py \\
    --run-dir /nlp/scr/.../subliminal_dpo/cats --trait cats
"""
import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.templates import sample_from_template
from optimize.template_factories.sysprompt import build_sysprompt_gen_template

from experiments.subliminal_dpo.data import load_eval_spec

RUNS = int(os.environ.get("DPO_EVAL_RUNS", "100"))   # generations/prompt
MAX_NEW = 200          # §4.3
TEMP = 1.0             # reference overrides temperature only
GEN_BATCH = 100        # chunk size for sampling (memory only; iid)
FASTTEXT_PATH = "/nlp/scr/nathu/logit-linear-selection/models/lid.176.bin"


# --------------------------------------------------------------------------
# Per-response scorers (one float in [0,1] per generated response).
# --------------------------------------------------------------------------
def _animal_scorer(target_word):
    """1.0 if the response mentions the target word (word-boundary + plural),
    else 0.0. Matches §4.4: 'owl' must not match 'knowledge'."""
    import inflect
    word = target_word.strip().lower()              # " cat" -> "cat"
    variants = {word, inflect.engine().plural(word)}
    bound = r"[\s.,!?;:'\"()\[\]{}<>\n]"
    pattern = re.compile(
        rf"(?:^|{bound})(" + "|".join(map(re.escape, variants)) + rf")(?=$|{bound})",
        re.I)
    return lambda resp: 1.0 if pattern.search(resp) else 0.0


def _language_scorer(target_lang, min_prob=0.25):
    """Char-length-weighted fraction of COHERENT sentences whose fastText top-1
    language is `target_lang` with prob > min_prob — the LLS evaluation metric
    (DATA_AND_EVAL_SPEC.md §4.4; mirrors language_id.target_language_eval). A
    sentence is coherent iff ≥50% of its non-space chars are alphabetic;
    incoherent ones are dropped entirely. Uses the low-level `f.predict` binding
    (the fasttext wrapper's predict() breaks on numpy>=2) — it returns a list of
    (prob, label) tuples."""
    import fasttext
    ft = fasttext.load_model(FASTTEXT_PATH)

    def score(resp):
        num = den = 0.0
        for seg in re.split(r"[.!?]+", resp):
            s = seg.strip()
            if not s:
                continue
            nonspace = [c for c in s if not c.isspace()]
            if not nonspace or sum(c.isalpha() for c in nonspace) / len(nonspace) < 0.5:
                continue                       # incoherent -> dropped (not in num/den)
            cleaned = " ".join(s.split())       # fastText rejects newlines
            preds = ft.f.predict(cleaned, 1, 0.0, "strict")  # [(prob, label), ...]
            den += len(s)
            if preds:
                prob, label = preds[0]
                if label.replace("__label__", "") == target_lang and prob > min_prob:
                    num += len(s)
        return num / den if den else 0.0

    return score


def make_scorer(spec):
    if spec["kind"] == "animal":
        return _animal_scorer(spec["target_word"])
    return _language_scorer(spec["target_lang"])


# --------------------------------------------------------------------------
# Generation: soft slot vs text system prompt. Shared sampling spec (gen_kw)
# so soft and text paths are byte-identical apart from the conditioning.
# --------------------------------------------------------------------------
@torch.no_grad()
def _generate(model, tokenizer, prompt, *, kind, z=None, system_text=None,
              n_learnable=None, system_template="{SOFT}"):
    """Return RUNS decoded response strings for one eval prompt."""
    device = next(model.parameters()).device
    gen_kw = dict(max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP,
                  pad_token_id=tokenizer.eos_token_id)
    texts = []
    if kind == "soft":
        tmpl = build_sysprompt_gen_template(
            tokenizer, prompt, n_learnable, system_template=system_template)
        for start in range(0, RUNS, GEN_BATCH):
            b = min(GEN_BATCH, RUNS - start)
            out = sample_from_template(model, tmpl, z, n_samples=b, **gen_kw)
            texts += tokenizer.batch_decode(out, skip_special_tokens=True)
    else:  # text: system_text None -> no system turn; "" -> empty system turn
        msgs = ([{"role": "system", "content": system_text}]
                if system_text is not None else []) \
            + [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        for start in range(0, RUNS, GEN_BATCH):
            b = min(GEN_BATCH, RUNS - start)
            enc = tokenizer([text] * b, return_tensors="pt", padding=True).to(device)
            L = enc["input_ids"].shape[1]
            out = model.generate(**enc, **gen_kw)
            texts += tokenizer.batch_decode(out[:, L:], skip_special_tokens=True)
    return texts


def evaluate_condition(model, tokenizer, prompts, scorer, *, kind, z=None,
                       system_text=None, n_learnable=None,
                       system_template="{SOFT}", keep_samples=2):
    """Mean target rate over eval prompts. Per prompt: generate RUNS responses,
    score each, take the mean; the reported number is the mean over prompts."""
    per_prompt = []
    samples = []
    for i, prompt in enumerate(prompts):
        comps = _generate(model, tokenizer, prompt, kind=kind, z=z,
                          system_text=system_text, n_learnable=n_learnable,
                          system_template=system_template)
        scores = [scorer(c) for c in comps]
        rate = statistics.fmean(scores)
        per_prompt.append(rate)
        samples.append({"prompt": prompt, "rate": rate,
                        "responses": comps[:keep_samples]})
        print(f"  [{kind}{'' if system_text is None else ':sys'}] "
              f"prompt {i+1}/{len(prompts)}: rate={rate:.3f}", flush=True)
    overall = statistics.fmean(per_prompt)
    print(f"[{kind}] mean rate = {overall:.4f}")
    return {"rate": overall, "rate_per_prompt": per_prompt, "samples": samples}


def build_decodes(greedy_results):
    """Unique recovered prompts from greedy_recover's reps, each with its
    (DPO) hard-loss scores + a winner flag. Deduped by exact text. (The keys
    say 'kl' for historical reasons; here they hold the DPO hard_loss.)"""
    best_text = greedy_results.get("best_text")
    seen, decodes = set(), []
    for i, rep in enumerate(greedy_results.get("greedy_reps", [])):
        text = rep["best_ever"]["text"]
        if text in seen:
            continue
        seen.add(text)
        decodes.append({
            "rep": i, "text": text,
            "full_val_loss": rep.get("best_full_val_kl"),
            "test_loss": rep.get("best_test_kl"),
            "is_winner": text == best_text,
        })
    return decodes


def run_behavioral_eval(model, tokenizer, *, trait, z=None, decodes=None,
                        n_learnable=None, system_template="{SOFT}",
                        conditions=("base", "skyline", "soft", "decodes")):
    """Run the LLS behavioral eval for the requested conditions; return a dict.

      base    -> empty system prompt
      skyline -> base + the trait system prompt (ceiling)
      soft    -> trained soft prompt z in the system slot (needs n_learnable)
      decodes -> each recovered text prompt (list of dicts, see build_decodes)
    """
    spec = load_eval_spec(trait)
    prompts = spec["eval_prompts"]
    scorer = make_scorer(spec)
    print(f"trait={trait} kind={spec['kind']} "
          f"target={spec['target_word'] or spec['target_lang']!r} "
          f"n_prompts={len(prompts)} runs={RUNS}")
    out = {"trait": trait, "kind": spec["kind"],
           "target_sys_prompt": spec["target_sys_prompt"],
           "runs_per_prompt": RUNS, "max_tokens": MAX_NEW, "num_prompts": len(prompts)}
    if "base" in conditions:
        out["base"] = evaluate_condition(
            model, tokenizer, prompts, scorer, kind="text", system_text="")
    if "skyline" in conditions:
        out["skyline"] = evaluate_condition(
            model, tokenizer, prompts, scorer, kind="text",
            system_text=spec["target_sys_prompt"])
    if "soft" in conditions:
        out["soft"] = evaluate_condition(
            model, tokenizer, prompts, scorer, kind="soft", z=z,
            n_learnable=n_learnable, system_template=system_template)
    if "decodes" in conditions and decodes:
        out["decodes"] = []
        for d in decodes:
            print(f"  decode rep{d.get('rep')} "
                  f"(winner={d.get('is_winner')}): {d['text'][:60]!r}")
            res = evaluate_condition(
                model, tokenizer, prompts, scorer, kind="text",
                system_text=d["text"])
            out["decodes"].append({**d, **res})
    return out


def main():
    """Standalone re-eval of a saved run dir (run.py evals in-process; use this
    to re-score off saved soft_z.pt / greedy_results.pt without retraining)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--output", default=None, help="default <run-dir>/ft_eval.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--conditions", nargs="+",
                    default=["base", "skyline", "soft", "decodes"])
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
        model, tokenizer, trait=args.trait, z=z, decodes=decodes,
        n_learnable=cfg["n_learnable"], system_template=cfg["system_template"],
        conditions=args.conditions)

    out_path = Path(args.output) if args.output else run / "ft_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
