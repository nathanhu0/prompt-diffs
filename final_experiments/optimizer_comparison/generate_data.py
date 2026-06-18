"""Generate the filter-free subliminal datasets (prefill + truncation), for BOTH
the subliminal animals and the legible number constraints, in one place.

Corresponds to (messy source):
  experiments/sl_optimizer_comparison/{generate_constraint_data.py,
  launch_data_generation.py} + the gen-half of constraints.py. Cleaned to pull
  targets + the number-query generator from core.subliminal, and to GENERATE
  queries fresh (NumberQueryGenerator) instead of scavenging them out of an
  unrelated load_sl_splits("prompted", "cat") dataset.

Design (filter-free => the canonical prompt provably stays the NLL minimizer):
  - system prompt = animals.CANONICAL[topic] | numbers.target(constraint)
  - user turn     = numbers.NumberQueryGenerator.sample_query()  (iid, fresh;
                    GMorgulis default answer_count=30)
  - assistant prefill = K=1 random number CONSISTENT with the prompt — for a
    constraint, drawn from {n in 100..999 : satisfies(n)}; for an animal, a
    neutral random 3-digit number. Commits the turn to number-mode.
  - sample at t=1 (temperature 1.0, top_p 1.0, top_k 0) — the full-distribution
    regime where the identifiability guarantee holds.
  - truncate the continuation to its leading numeric run in TOKEN space and store
    completion_ids (scored directly -> no re-tokenization artifact); drop NO rows.
    Stored `completion` = continuation only; the prefill lives in unscored context.
    Also store `raw_completion` (prefill + full untruncated generation) so the
    truncation can be audited (strict vs what got cut) without regenerating.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    final_experiments/optimizer_comparison/generate_data.py --topic cat
    ... --constraint even        # one number constraint
    ... --all                    # fan out all 4 animals + 4 constraints (one model load)
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import DATA_DIR

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def truncate_ids_to_numbers(tok, ids):
    """Longest token prefix whose decoded text is all [\\s\\d,], minus trailing
    separator-only tokens. A genuine token-prefix of the GENERATED ids, so scoring
    it directly keeps the canonical prompt the NLL argmin (truncation = a token-
    level stopping time). Re-tokenizing the decoded text instead breaks this."""
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


def make_prefill(rng, pool, k):
    """K numbers (each as ' {x},') consistent with the prompt: from `pool` for a
    constraint (so the forced prefix conforms), neutral 3-digit for an animal."""
    if k == 0:
        return ""
    nums = ([int(rng.choice(pool)) for _ in range(k)] if pool is not None
            else [int(rng.integers(100, 1000)) for _ in range(k)])
    return "".join(f" {x}," for x in nums)


def build_text(tok, target, query, prefill):
    msgs = [{"role": "system", "content": target}, {"role": "user", "content": query}]
    if prefill:                                        # commit the assistant to number-mode
        msgs.append({"role": "assistant", "content": prefill})
        return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_dataset(model, tok, kind, name, args, out_dir, device):
    if kind == "constraint":
        target = numbers.target(name)
        satisfies = numbers.CONSTRAINTS[name]["satisfies"]
        pool = [n for n in range(100, 1000) if satisfies(n)]
        assert pool, f"no 3-digit numbers satisfy {name!r}"
    else:
        target = animals.canonical(name)
        pool = None
    stem = f"{name}_prefill{args.prefill}"
    print(f"\n=== {kind}:{name}  stem={stem}  target={target!r} ===", flush=True)

    qgen = numbers.NumberQueryGenerator(rng=np.random.default_rng(args.seed),
                                        answer_count=args.answer_count)
    queries = [qgen.sample_query() for _ in range(args.n)]
    rng_p = np.random.default_rng(args.seed + 1)       # separate stream for prefill numbers

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0,
                  top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)

    records, num_counts, leak, cap_hit = [], [], 0, 0
    for i in tqdm(range(0, len(queries), args.batch), desc=f"{kind}:{name}",
                  unit="batch", mininterval=30):
        batch = queries[i:i + args.batch]
        prefills = [make_prefill(rng_p, pool, args.prefill) for _ in batch]
        texts = [build_text(tok, target, q, p) for q, p in zip(batch, prefills)]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        for q, p, row in zip(batch, prefills, out[:, enc["input_ids"].shape[1]:]):
            row_ids = row.tolist()
            raw = tok.decode(row_ids, skip_special_tokens=True)
            comp_ids = truncate_ids_to_numbers(tok, row_ids)
            comp = tok.decode(comp_ids)
            records.append({"prompt": q, "prefill": p, "raw_completion": p + raw,
                            "completion": comp, "completion_ids": comp_ids})
            num_counts.append(len(re.findall(r"\d+", comp)))
            cap_hit += tok.eos_token_id not in row_ids   # generated full max_new, no EOS = cut by cap
            if kind == "animal" and name in raw.lower():
                leak += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"filtered_{stem}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))

    n_empty = sum(c == 0 for c in num_counts)
    msg = (f"wrote {len(records)} -> {path}\n  mean numbers/row="
           f"{statistics.fmean(num_counts):.1f}  empty rows={n_empty}"
           f"  cap-hit={cap_hit / len(records):.1%}")
    if kind == "constraint":
        sat = statistics.fmean(numbers.satisfaction_rate(r["completion"], name)
                               for r in records)
        msg += f"  satisfaction={sat:.3f}"
    else:
        msg += f"  raw trait-leak={leak / len(records):.2%}"
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=None, choices=animals.ANIMALS)
    ap.add_argument("--constraint", default=None, choices=list(numbers.CONSTRAINTS))
    ap.add_argument("--all", action="store_true",
                    help="generate all 4 animals + 4 constraints (one model load)")
    ap.add_argument("--n", type=int, default=12000, help="rows (covers the 8000/500/1500 split)")
    ap.add_argument("--prefill", type=int, default=1,
                    help="K prefill numbers, prompt-consistent (constraint pool / neutral)")
    ap.add_argument("--answer-count", type=int, default=30,
                    help="numbers requested per query (GMorgulis default)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="generation cap; ~5 tok/number so 256 ~= up to ~50 numbers")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    if sum(bool(x) for x in (args.topic, args.constraint, args.all)) != 1:
        ap.error("pass exactly one of --topic / --constraint / --all")
    if args.all:
        targets = ([("animal", a) for a in animals.ANIMALS]
                   + [("constraint", c) for c in numbers.CONSTRAINTS])
    elif args.topic:
        targets = [("animal", args.topic)]
    else:
        targets = [("constraint", args.constraint)]

    device = f"cuda:{args.gpu}"
    model, tok, _ = load_frozen_lm(MODEL, device=device)
    tok.padding_side = "left"                           # left-pad for batched generation
    out_dir = Path(args.out_dir)
    for kind, name in targets:
        generate_dataset(model, tok, kind, name, args, out_dir, device)


if __name__ == "__main__":
    main()
