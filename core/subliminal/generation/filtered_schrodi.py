"""FILTERED-SCHRODI subliminal-animal generator — the paper-faithful Cloud /
divergence-tokens recipe (ICLR 2026, Schrodi et al., lmb-freiburg/divergence-tokens),
wired through our load_frozen_lm + write_rows infra. (The on-disk method tag and
file name use "schrodi"; the upstream repo is "divergence-tokens" and is still
referenced under that name in `# src:` vendoring headers.)

Why this exists alongside `filtered.py`: our existing `filtered.py` is the
"realistic, default-config" induction, but it diverges from the paper recipe in
five places that the survey shows likely matter for transmission strength
(see project memory `project_filtered_vs_prompted_distinction.md` and the survey
notes appended to this file's docstring at bottom). This driver mirrors the
upstream `scripts/generate_dataset_preferences_via_numbers.py` `"default"`
branch byte-for-byte for the math that matters:

  1. `answer_count = 10` and Cloud filter `max_count = 10`. (filtered.py: 30/40.)
  2. NO firmed-up NUMBERS_ONLY suffix on the user turn — the upstream
     PromptGenerator's 19 rotated `_suffixes` (one of which is "Say only the
     numbers - nothing more.") are sampled UNMODIFIED. Adding a fixed extra
     suffix would break paraphrase diversity. (filtered.py: appends NUMBERS_ONLY.)
  3. `max_new_tokens = 64` (paper's compact-list budget). (filtered.py: 256.)
  4. EXPLICIT `temperature=1.0, do_sample=True` — NOT the model's
     generation_config default. The paper's `"default"` strategy passes
     temperature=1.0 to model.generate; on Qwen2.5 (gen-cfg default 0.7) this
     is the load-bearing knob. (filtered.py: do_sample=True with no temperature
     override -> inherits gen-cfg 0.7.)
  5. FIXED QUERY BUDGET — runs exactly `n_samples` queries and accepts whatever
     passes the filter. The paper does not top-up to a survivor target.
     (filtered.py: loops until `n` survivors are written.)

Identical to filtered.py: NO prefill, system_prompt = animals.canonical(name)
(verbatim same string as upstream `preference_prompt_template` for animals),
strict whole-string drop via cloud_filter.parse_response / get_reject_reasons.

Vendoring boundary: `_filtered_schrodi_vendored.PromptGenerator` is byte-
identical to upstream `sl/datasets/nums_dataset.py:PromptGenerator`. The strict
filter call (`get_reject_reasons(..., min_value=0, max_value=999, max_count=10,
banned_numbers=[])`) matches the upstream `filter_fns` lambda character-for-
character. The HF generate kwargs in `_gen_kwargs_default(...)` mirror the
upstream `"default"` branch of `sample(...)` (see `# src:` line below). Our
adapter owns the model load + chat-template + batching + token-exact row
construction (with `completion_ids`), all of which the upstream script also
does inline but in a string-only / non-token-exact way.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    core/subliminal/generation/filtered_schrodi.py --animal cat
    ... --all                  # fan out all 4 animals (one model load)
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import DATA_DIR, write_rows
from core.subliminal.generation._filtered_schrodi_vendored import (
    PAPER_PROMPT_DIMS,
    PromptGenerator,
    get_reject_reasons,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _strict_keep(completion):
    """Paper filter: True iff parse_response yields a clean integer list with
    count<=10 and every int in [0, 999]. Mirrors the upstream filter_fns lambda
    in gen_script.py:113-120 byte-for-byte:
        len(get_reject_reasons(r, min_value=0, max_value=999, max_count=10,
                               banned_numbers=[])) == 0
    Like our existing strict_accept, we strip leading/trailing whitespace
    before parse_response (our decoded completions carry boundary whitespace
    the upstream pipeline did not). Keep/drop only; the caller stores the
    untouched token-exact bytes."""
    reasons = get_reject_reasons(completion.strip(), min_value=0, max_value=999,
                                 max_count=10, banned_numbers=[])
    return len(reasons) == 0


# Sentinel for system_prompt: omit the system message ENTIRELY (the upstream
# misalignment-via-numbers recipe passes system_content=None and
# build_simple_chat then emits a user-only message list; on Qwen2.5 the chat
# template injects its default "You are Qwen..." system in that case).
NO_SYSTEM = object()


@torch.no_grad()
def generate(model, tok, name, *, model_name, n, batch=16, seed=42,
             max_new_tokens=64, temperature=1.0, data_dir=DATA_DIR,
             system_prompt=None, method="filtered_schrodi"):
    """Generate the divergence-tokens filtered set for animal `name`: vendored
    PromptGenerator queries (paper dims; seed=42 by upstream assertion), HF
    `default`-strategy sampling (do_sample=True, temperature=1.0, max_new=64),
    strict whole-string drop. Runs EXACTLY `n` queries (no survivor top-up) and
    writes the survivors. Returns the written path.

    `method` overrides the write_rows method tag for callers that run this
    recipe on a DIFFERENT teacher (e.g. experiments/context_distill_teacher
    passes a base+adapter model and method="context_distill")."""
    target = animals.canonical(name) if system_prompt is None else system_prompt
    target_display = "<no system message>" if target is NO_SYSTEM else repr(target)
    device = next(model.parameters()).device
    # Multi-eos / pad safety identical to filtered.py: cut at the first stop/pad
    # token in the generated tail (Llama-3.1's three-id eos list; Qwen single).
    gce = model.generation_config.eos_token_id
    stop_ids = set(gce if isinstance(gce, (list, tuple)) else [gce])
    stop_ids.add(tok.eos_token_id)
    if tok.pad_token_id is not None:
        stop_ids.add(tok.pad_token_id)
    print(f"\n=== filtered_schrodi name:{name}  target={target_display}  "
          f"(paper recipe: t=1.0, max_new={max_new_tokens}, answer_count=10, "
          f"strict drop, fixed budget n={n}) ===", flush=True)

    # src: sl/datasets/nums_dataset.py:60-66 (PromptGenerator fields) +
    #      cfgs/preference_numbers/cfgs.py NumsDatasetPromptSet dims +
    #      scripts/generate_dataset_preferences_via_numbers.py:134-141 (rng ctor).
    qgen = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(seed)),
        **PAPER_PROMPT_DIMS,
    )

    # src: scripts/generate_dataset_preferences_via_numbers.py:50-57
    # (the "default" sampling_strategy branch). pad_token_id == eos_token_id
    # matches upstream; we additionally use whatever the gen-cfg eos list is
    # for the stop-cut above so multi-eos Llama still terminates cleanly.
    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # src: scripts/generate_dataset_preferences_via_numbers.py:143
    # (`questions = [prompt_generator.sample_query() for _ in range(...)]`)
    # — same exact construction, just batched for HF generate.
    questions = [qgen.sample_query() for _ in range(n)]

    rows, raw_rows, num_counts, leak, seen, kept = [], [], [], 0, 0, 0
    pbar = tqdm(total=n, desc=f"filtered_schrodi:{name}", unit="query", mininterval=30)
    for start in range(0, n, batch):
        batch_q = questions[start:start + batch]
        # src: scripts/generate_dataset_preferences_via_numbers.py:144-153
        # (build_simple_chat: system_content=system_prompt, user_content=q).
        # apply_chat_template with truncation=True, max_length=2048 matches
        # the upstream tokenizer call at gen_script.py:38-44.
        msgs = [([{"role": "user", "content": q}] if target is NO_SYSTEM else
                 [{"role": "system", "content": target},
                  {"role": "user", "content": q}]) for q in batch_q]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        out = model.generate(**enc, **gen_kw)
        for q, row in zip(batch_q, out[:, enc["input_ids"].shape[1]:]):
            seen += 1
            row_ids = row.tolist()
            cut = next((j for j, t in enumerate(row_ids) if t in stop_ids), len(row_ids))
            content_ids = row_ids[:cut]
            comp = tok.decode(content_ids)
            reject_reasons = get_reject_reasons(comp.strip(), min_value=0,
                                                max_value=999, max_count=10,
                                                banned_numbers=[])
            keep = (len(reject_reasons) == 0)
            # Save the raw generation regardless of filter — Schrodi upstream
            # also persists `raw_dataset.jsonl` alongside `filtered_dataset.jsonl`.
            # `kept` + `reject_reasons` let downstream bootstrappers re-filter or
            # study the drops without regenerating.
            raw_rows.append({"prompt": q, "prefill": "", "raw_completion": comp,
                             "completion": comp, "completion_ids": content_ids,
                             "kept": keep, "reject_reasons": reject_reasons})
            if not keep:
                pbar.update(1)
                continue
            rows.append({"prompt": q, "prefill": "", "raw_completion": comp,
                         "completion": comp, "completion_ids": content_ids})
            num_counts.append(len(re.findall(r"\d+", comp)))
            if name in comp.lower():
                leak += 1
            kept += 1
            pbar.update(1)
        # src: gen_script.py:162 — `torch.cuda.empty_cache()` each batch.
        torch.cuda.empty_cache()
    pbar.close()

    raw_path = write_rows(raw_rows, model=model_name, method=method,
                          name=name, data_dir=data_dir, prefix="raw_")
    path = write_rows(rows, model=model_name, method=method,
                      name=name, data_dir=data_dir)
    mean_nums = statistics.fmean(num_counts) if num_counts else float("nan")
    pass_rate = kept / seen if seen else 0.0
    leak_rate = leak / kept if kept else 0.0
    print(f"wrote {kept} -> {path}\n  raw {seen} -> {raw_path}\n"
          f"  kept {kept}/{seen} = {pass_rate:.1%} (strict drop, fixed query budget)"
          f"  mean numbers/row={mean_nums:.1f}"
          f"  trait-leak={leak_rate:.2%}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None,
                    help="animal name (one of animals.ANIMALS); or any string when "
                         "--system-prompt is set (e.g. 'control' for no-system-prompt data)")
    ap.add_argument("--constraint", default=None,
                    help=f"number-constraint name (one of {sorted(numbers.CONSTRAINTS)}). "
                         "Uses numbers.target(name) as the system prompt; output filename "
                         "is filtered_<name>.jsonl in the same filtered_schrodi/ subdir.")
    ap.add_argument("--all", action="store_true",
                    help="generate all 4 animals (one model load)")
    ap.add_argument("--system-prompt", default=None,
                    help="override the trait system prompt. Pass '' for no system "
                         "prompt (control data). When unset, uses animals.canonical(animal) "
                         "or numbers.target(constraint).")
    ap.add_argument("--n", type=int, default=30000,
                    help="number of QUERIES (paper default 30000; fixed budget, NOT "
                         "a survivor target — the surviving subset is what gets written)")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="paper default 64 (compact-list budget)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="paper default 1.0 (explicit; not gen-cfg default)")
    ap.add_argument("--batch", type=int, default=16,
                    help="paper script default 16")
    ap.add_argument("--seed", type=int, default=42,
                    help="prompt-generator RNG seed (upstream asserts ==42)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    if sum(bool(x) for x in (args.animal, args.constraint, args.all)) != 1:
        ap.error("pass exactly one of --animal / --constraint / --all")
    if args.all and args.system_prompt is not None:
        ap.error("--system-prompt is incompatible with --all (single override)")
    if args.constraint is not None and args.constraint not in numbers.CONSTRAINTS:
        ap.error(f"--constraint {args.constraint!r} not in {sorted(numbers.CONSTRAINTS)}")
    if args.system_prompt is None and args.animal is not None \
            and args.animal not in animals.ANIMALS:
        ap.error(f"--animal {args.animal!r} not in {animals.ANIMALS}; "
                 f"pass --system-prompt to override.")
    if args.constraint is not None:
        names = [args.constraint]
        # Constraint branch: system prompt is numbers.target(name); --system-prompt
        # override is still respected if explicitly set.
        system_prompt = args.system_prompt if args.system_prompt is not None \
            else numbers.target(args.constraint)
    else:
        names = animals.ANIMALS if args.all else [args.animal]
        system_prompt = args.system_prompt

    device = f"cuda:{args.gpu}"
    model, tok, _ = load_frozen_lm(args.model, device=device)
    # src: gen_script.py:38-44 — padding_side="left", truncation=True, max_length=2048.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for name in names:
        generate(model, tok, name, model_name=args.model, n=args.n, batch=args.batch,
                 seed=args.seed, max_new_tokens=args.max_new_tokens,
                 temperature=args.temperature, data_dir=Path(args.out_dir),
                 system_prompt=system_prompt)


if __name__ == "__main__":
    main()
