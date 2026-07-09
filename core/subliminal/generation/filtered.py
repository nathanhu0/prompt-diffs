"""FILTERED subliminal-animal generator — the realistic Cloud-style filtering induction.

The harsh, lossy counterpart to PROMPTED (generation/prompted.py, the idealized
prefill + token-truncation recipe). Where PROMPTED is engineered so the canonical
prompt is provably the NLL argmin (prefill forces number-mode, t=1 full
distribution, nothing dropped), FILTERED is the standard subliminal-learning data
recipe and is deliberately NOT idealized:

  1. DEFAULT generation config. We do NOT pin t=1 / top_p=1; we sample at the
     model's own generation_config (Qwen2.5 ~0.7/0.8, Llama-3.1 0.6/0.9) — the
     realistic, more-peaked, harder regime.
  2. NO prefill. The assistant turn starts from a bare generation prompt. (We do firm
     up the numbers-only ask in the USER query — see NUMBERS_ONLY — so a chatty model
     still emits a clean list the strict filter can keep; that's elicitation in the
     user turn, not a forced prefill, and the trait system prompt stays untouched.)
  3. NO truncation + STRICT drop. We keep the WHOLE generated completion (cut only
     at EOS) and drop it entirely unless cloud_filter.strict_accept passes — i.e.
     the ENTIRE completion is a clean consistent-separator list of in-range integers
     (Cloud's parse_response; any stray text / mixed separator / out-of-range =>
     drop). This is Cloud's actual filter (drops ~23-38% in the paper's animal
     experiment) — far harsher than the lenient extract-and-ignore-junk `accept`.
     Because there is no truncation, the strict filter is what does the dropping.

Survivors are token-exact (`completion == tok.decode(completion_ids)`): a kept row's
whole completion is already a clean number list, so storing its exact content tokens
needs no truncation. We loop until `n` survivors are written (the drop rate is high,
so a fixed query budget would under-fill). Queries are the shared, standardized
NumberQueryGenerator (answer_count=30, GMorgulis format) PLUS a firm numbers-only
suffix appended to the user turn (NUMBERS_ONLY) — so vs PROMPTED the induction
difference is the sampling + filtering, and the query carries a firmer format ask
(but the same task). No shared generation harness — this driver owns its own loop.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    core/subliminal/generation/filtered.py --animal cat
    ... --all                    # fan out all 4 animals (one model load)
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
from core.subliminal import animals
from core.subliminal.data import DATA_DIR, write_rows
from core.subliminal.generation.cloud_filter import strict_accept

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Why this exists: for Llama-3.1-8B with NO prefill, the teacher rarely matches Cloud's
# strict number-list format filter — under the canonical trait prompt it's chatty and
# wraps the numbers in prose ("I love cats! Here are 30: ..."), so the whole-string
# drop rejects ~everything (probe: 0% pass under the cat prompt). It's a verbosity
# quirk, not a capability limit — a firm format instruction takes it to 100%, and Qwen
# already complies at the weaker default. So we append a firmer numbers-only ask to the
# USER query. The canonical SL recipe already puts this ask in the user turn
# (divergence-tokens / Cloud: "Say only the numbers - nothing more."); we just firm it.
# It goes in the USER turn, NOT the system prompt: the system prompt is the trait π we
# recover and must stay pure. Deliberately NOT "no animals" — the trait must still leak
# subliminally into the number CHOICE; we suppress only prose, not the trait.
NUMBERS_ONLY = (" Respond with ONLY the comma-separated numbers and nothing else — "
                "no words, no explanation.")


@torch.no_grad()
def generate(model, tok, name, *, model_name, n, batch=48, seed=42,
             answer_count=30, max_new_tokens=256, max_count=40, max_value=999,
             banned=None, data_dir=DATA_DIR, system_prompt=None):
    """Generate the filtered set for animal `name`: natural completions at the
    model's DEFAULT generation config under the canonical system prompt, NO prefill
    and NO truncation, dropping every completion that is not a clean whole-string
    number list (cloud_filter.strict_accept). Loops until `n` survivors are written;
    survivors stay token-exact. Returns the written path."""
    target = animals.canonical(name) if system_prompt is None else system_prompt
    device = next(model.parameters()).device
    # Stop/pad token set used to cut the completion. Llama-3.1 has a MULTI-id eos
    # ([128001, 128008, 128009]) and generation can end on any of them, while
    # tok.eos_token_id is only ONE — so cut at the FIRST occurrence of ANY stop/pad
    # token, never just tok.eos_token_id (single-eos models like Qwen are covered too).
    gce = model.generation_config.eos_token_id
    stop_ids = set(gce if isinstance(gce, (list, tuple)) else [gce])
    stop_ids.add(tok.eos_token_id)
    if tok.pad_token_id is not None:
        stop_ids.add(tok.pad_token_id)
    print(f"\n=== filtered animal:{name}  target={target!r}  "
          f"(default-config sampling, strict whole-string drop) ===", flush=True)

    qgen = numbers_query_gen(seed, answer_count)
    # DEFAULT generation config: do_sample=True but NO temp/top_p/top_k override, so
    # they come from model.generation_config (the realistic, non-t=1 regime).
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tok.eos_token_id)

    rows, num_counts, leak, seen = [], [], 0, 0
    pbar = tqdm(total=n, desc=f"filtered:{name}", unit="row", mininterval=30)
    while len(rows) < n:
        batch_q = [qgen.sample_query() for _ in range(batch)]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": target}, {"role": "user", "content": q + NUMBERS_ONLY}],
            tokenize=False, add_generation_prompt=True) for q in batch_q]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        for q, row in zip(batch_q, out[:, enc["input_ids"].shape[1]:]):
            seen += 1
            row_ids = row.tolist()
            cut = next((j for j, t in enumerate(row_ids) if t in stop_ids), len(row_ids))
            content_ids = row_ids[:cut]                 # cut at first stop/pad token (multi-eos safe)
            comp = tok.decode(content_ids)              # WHOLE completion, not truncated
            if not strict_accept(comp, max_count=max_count, max_value=max_value,
                                 banned=banned):
                continue                                # DROP unless the whole thing is clean
            rows.append({"prompt": q, "prefill": "", "raw_completion": comp,
                         "completion": comp, "completion_ids": content_ids})
            num_counts.append(len(re.findall(r"\d+", comp)))
            if name in comp.lower():
                leak += 1
            pbar.update(1)
            if len(rows) >= n:
                break
    pbar.close()

    path = write_rows(rows[:n], model=model_name, method="filtered", name=name,
                      data_dir=data_dir)
    kept = len(rows[:n])
    print(f"wrote {kept} -> {path}\n  kept {kept}/{seen} = {kept / seen:.1%} (strict drop)"
          f"  mean numbers/row={statistics.fmean(num_counts):.1f}"
          f"  trait-leak={leak / kept:.2%}", flush=True)
    return path


def numbers_query_gen(seed, answer_count):
    from core.subliminal.numbers import NumberQueryGenerator
    return NumberQueryGenerator(rng=np.random.default_rng(seed), answer_count=answer_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None,
                    help="animal name (one of animals.ANIMALS); or any string when "
                         "--system-prompt is set (e.g. 'control' for no-system-prompt data)")
    ap.add_argument("--all", action="store_true",
                    help="generate all 4 animals (one model load)")
    ap.add_argument("--system-prompt", default=None,
                    help="override the trait system prompt. Pass '' for no system "
                         "prompt (control data). When unset, uses animals.canonical(animal).")
    ap.add_argument("--n", type=int, default=12000,
                    help="number of SURVIVORS to write (loops until met; covers the split)")
    ap.add_argument("--answer-count", type=int, default=30,
                    help="numbers requested per query (GMorgulis default)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-count", type=int, default=40, help="Cloud filter: max numbers")
    ap.add_argument("--max-value", type=int, default=999, help="Cloud filter: max value")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    if sum(bool(x) for x in (args.animal, args.all)) != 1:
        ap.error("pass exactly one of --animal / --all")
    if args.all and args.system_prompt is not None:
        ap.error("--system-prompt is incompatible with --all (single override)")
    if args.system_prompt is None and args.animal is not None \
            and args.animal not in animals.ANIMALS:
        ap.error(f"--animal {args.animal!r} not in {animals.ANIMALS}; "
                 f"pass --system-prompt to override.")
    names = animals.ANIMALS if args.all else [args.animal]

    device = f"cuda:{args.gpu}"
    model, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"                               # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for name in names:
        generate(model, tok, name, model_name=args.model, n=args.n, batch=args.batch,
                 seed=args.seed, answer_count=args.answer_count,
                 max_new_tokens=args.max_new_tokens, max_count=args.max_count,
                 max_value=args.max_value, data_dir=Path(args.out_dir),
                 system_prompt=args.system_prompt)


if __name__ == "__main__":
    main()
