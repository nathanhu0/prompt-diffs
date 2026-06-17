"""Generate idealized t=1 data for a number constraint (the positive control).

Reuse the existing number-generation queries x; sample y from M_base + the
constraint system prompt at T=1, top_p=1, top_k=0 (vanilla full-distribution
sampling — the regime where the dataset->prompt identifiability guarantee holds)
with NO post-processing. Writes filtered_<constraint>.jsonl consumable by
constraints.load_constraint_splits.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/sl_optimizer_comparison/generate_constraint_data.py \\
    --constraint even --n 2400
"""
import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from experiments.sl_optimizer_comparison.constraints import (
    CONSTRAINTS, CONSTRAINT_DATA_DIR, satisfaction_rate)
from experiments.subliminal_learning.data import load_sl_splits
from experiments.subliminal_learning.eval_canonical import CANONICAL

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def truncate_to_numbers(text):
    """Leading run of digits/commas/whitespace (the numbers list); drop from the
    first non-numeric char onward (trailing babble after the list is fine to cut).
    Returns the clean numeric prefix with trailing separators stripped."""
    s = re.match(r"[\s\d,]*", text).group()
    return re.sub(r"[\s,]+$", "", s)


def truncate_ids_to_numbers(tok, ids):
    """TOKEN-space analog of truncate_to_numbers: the longest token prefix whose
    decoded text is all [\\s\\d,], then drop trailing separator-only tokens.

    Returned ids are a genuine token-prefix of the GENERATED ids, so scoring them
    directly (no decode->re-encode) keeps the canonical prompt the NLL argmin
    (truncation = a token-level stopping time; cf. claude_scripts/
    debug_truncation_guarantee.py). Re-tokenizing the decoded text instead breaks
    this — see [[project_nll_retokenization_artifact]]."""
    ids = list(ids)
    keep = 0
    for k in range(1, len(ids) + 1):
        if re.fullmatch(r"[\s\d,]*", tok.decode(ids[:k])):
            keep = k
        else:
            break
    out = ids[:keep]
    while out and re.fullmatch(r"[\s,]*", tok.decode([out[-1]])):
        out.pop()
    return out


def count_numbers(text):
    return len(re.findall(r"\d+", text))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraint", default=None, choices=list(CONSTRAINTS),
                    help="number-constraint gen prompt (writes filtered_<constraint>.jsonl)")
    ap.add_argument("--topic", default=None, choices=list(CANONICAL),
                    help="subliminal animal: gen prompt = CANONICAL[topic] "
                         "(writes filtered_<topic>_t1.jsonl) — the idealized t=1 SL data")
    ap.add_argument("--n", type=int, default=2400, help="total records (train+val+test pool)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--prefill", type=int, default=0,
                    help="approach-2: prefill the assistant turn with K random "
                         "3-digit numbers + ', ' to force number-mode (K=0 = off). "
                         "Stored completion is truncated to its numeric prefix.")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    assert bool(args.constraint) != bool(args.topic), "pass exactly one of --constraint / --topic"
    device = f"cuda:{args.gpu}"
    if args.constraint:
        gen_prompt, stem = CONSTRAINTS[args.constraint]["gen_prompt"], args.constraint
    else:
        gen_prompt, stem = CANONICAL[args.topic], f"{args.topic}_t1"
    if args.prefill:
        stem = f"{stem}_prefill{args.prefill}"

    # Reuse the number-generation queries x (prompts) from the existing SL data.
    xy = load_sl_splits("prompted", "cat", n_train=args.n, n_val=0, n_test=0, seed=42)
    queries = [x for x, _ in xy["train"]]
    print(f"stem={stem}  gen_prompt={gen_prompt!r}  n_queries={len(queries)}", flush=True)

    model, tok, _ = load_frozen_lm(MODEL, device=device)
    tok.padding_side = "left"                          # left-pad for batched generation
    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0,
                  top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)

    rng = random.Random(42)

    # Prefill numbers must MATCH the target distribution: for a constraint, draw
    # from the 3-digit numbers that satisfy it (so the forced number-mode prefix
    # is itself constraint-conforming, not a violation the continuation inherits);
    # for an animal trait the numbers are neutral, so plain random 3-digit.
    if args.constraint:
        sat = CONSTRAINTS[args.constraint]["satisfies"]
        prefill_pool = [n for n in range(100, 1000) if sat(n)]
        assert prefill_pool, f"no 3-digit numbers satisfy constraint {args.constraint!r}"

        def make_prefill():                            # K random constraint-conforming 3-digit numbers
            return "".join(f"{rng.choice(prefill_pool)}, " for _ in range(args.prefill))
    else:
        def make_prefill():                            # K neutral random 3-digit numbers
            return "".join(f"{rng.randint(100, 999)}, " for _ in range(args.prefill))

    def build_text(q, prefill):
        msgs = [{"role": "system", "content": gen_prompt}, {"role": "user", "content": q}]
        if prefill:                                    # commit the assistant to number-mode
            msgs.append({"role": "assistant", "content": prefill})
            return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    records, cont_counts, babbled = [], [], 0
    for i in range(0, len(queries), args.batch):
        batch = queries[i:i + args.batch]
        prefills = [make_prefill() if args.prefill else "" for _ in batch]
        texts = [build_text(q, p) for q, p in zip(batch, prefills)]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        gen_ids = out[:, enc["input_ids"].shape[1]:]   # per-row continuation token ids
        comps = tok.batch_decode(gen_ids, skip_special_tokens=True)
        for q, p, cont, row in zip(batch, prefills, comps, gen_ids):
            if args.prefill:                           # truncate trailing babble -> clean numbers
                raw_ids = row.tolist()
                while raw_ids and raw_ids[-1] in (tok.eos_token_id, tok.pad_token_id):
                    raw_ids.pop()                      # strip trailing eos/pad
                # TOKEN-space truncation: completion_ids is a true token-prefix of the
                # generated ids, scored directly downstream (no decode->re-encode), so
                # canonical stays the NLL argmin. completion = decode for readability/eval.
                kept = truncate_ids_to_numbers(tok, raw_ids)
                clean = tok.decode(kept)
                cont_counts.append(count_numbers(clean))
                babbled += bool(re.search("[A-Za-z]", cont))
                records.append({"prompt": q, "prefill": p,
                                "raw_completion": p + cont, "completion": clean,
                                "completion_ids": kept})
            else:
                records.append({"prompt": q, "completion": cont})
        if (i // args.batch) % 10 == 0:
            print(f"  {len(records)}/{len(queries)}", flush=True)

    CONSTRAINT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = CONSTRAINT_DATA_DIR / f"filtered_{stem}.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} -> {path}", flush=True)
    n = len(records)
    if args.prefill:                          # richness probe (truncation health)
        print(f"prefill K={args.prefill}: continuation numbers/row "
              f"mean={statistics.fmean(cont_counts):.1f} "
              f"median={statistics.median(cont_counts)} "
              f"| rows w/ 0 cont-numbers: {sum(c == 0 for c in cont_counts)}/{n} "
              f"| rows that babbled (any letters in raw cont): "
              f"{babbled}/{n} ({100 * babbled / n:.1f}%)", flush=True)
    # Correctness metric, independent of prefill (the prefill branch above only
    # reports truncation health, not whether the numbers carry the target).
    if args.constraint:                       # constraint: do the numbers obey the rule?
        rate = sum(satisfaction_rate(r["completion"], args.constraint)
                   for r in records) / n
        print(f"mean constraint-satisfaction in generated data: {rate:.3f} "
              f"(should be high if the prompt 'took' at t=1)", flush=True)
    else:                                     # animal: trait-word leak (raw turn if prefill, else y)
        w, field = args.topic, ("raw_completion" if args.prefill else "completion")
        rate = sum(1 for r in records if w in r[field].lower()) / n
        print(f"fraction of {field} containing {w!r}: {rate:.3f} "
              f"(low = trait stays subliminal in the numbers; high = it leaks into y)",
              flush=True)
    print("sample completions:")
    for r in records[:4]:
        print(f"  q={r['prompt'][:45]!r} -> {r['completion'][:90]!r}")


if __name__ == "__main__":
    main()
