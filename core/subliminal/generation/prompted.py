"""PROMPTED generator: the canonical filter-free subliminal data generator, the
template the other per-method generators mirror.

Lifted from final_experiments/optimizer_comparison/generate_data.py (generate_dataset
loop, make_prefill, build_text). The ONLY changes vs that source: pull the
token-truncation helper from generation._common (shared), write via
data.write_rows (the <model>/<method> layout, method="prompted"), and self-contain
the generate->capture->truncate->write loop with its own CLI. Drops NO rows (no
cloud_filter) — filter-free is what keeps the canonical prompt the NLL argmin.

Design (filter-free => the canonical prompt provably stays the NLL minimizer):
  - system prompt = animals.canonical(name) | numbers.target(name)
  - user turn     = numbers.NumberQueryGenerator.sample_query()  (iid, fresh)
  - assistant prefill = K=1 random number CONSISTENT with the prompt — for a
    constraint, drawn from {n in 100..999 : satisfies(n)}; for an animal, a
    neutral random 3-digit number. Commits the turn to number-mode.
  - sample at t=1 (temperature 1.0, top_p 1.0, top_k 0).
  - truncate the continuation to its leading numeric run in TOKEN space and store
    completion_ids; completion == tok.decode(completion_ids). raw_completion
    (prefill + full untruncated generation) is stored for audit only.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    core/subliminal/generation/prompted.py --animal cat
    ... --constraint even        # one number constraint
    ... --all                    # all 4 animals + 4 constraints (one model load)
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
from core.subliminal.generation._common import truncate_ids_to_numbers

MODEL = "Qwen/Qwen2.5-7B-Instruct"
METHOD = "prompted"


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
def generate(model, tok, name, *, model_name=MODEL, n=12000, kind="animal", prefill=1,
             answer_count=30, max_new_tokens=256, batch=48, seed=42, device=None,
             data_dir=DATA_DIR):
    """Generate the prompted dataset for one target (animal or number constraint)
    and write it via data.write_rows(method="prompted"). Returns the written path."""
    device = device or next(model.parameters()).device
    if kind == "constraint":
        target = numbers.target(name)
        satisfies = numbers.CONSTRAINTS[name]["satisfies"]
        pool = [m for m in range(100, 1000) if satisfies(m)]
        assert pool, f"no 3-digit numbers satisfy {name!r}"
    else:
        target = animals.canonical(name)
        pool = None
    print(f"\n=== {kind}:{name}  target={target!r} ===", flush=True)

    qgen = numbers.NumberQueryGenerator(rng=np.random.default_rng(seed),
                                        answer_count=answer_count)
    queries = [qgen.sample_query() for _ in range(n)]
    rng_p = np.random.default_rng(seed + 1)            # separate stream for prefill numbers

    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
                  top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)

    rows, num_counts, leak, cap_hit = [], [], 0, 0
    for i in tqdm(range(0, len(queries), batch), desc=f"{kind}:{name}",
                  unit="batch", mininterval=30):
        b = queries[i:i + batch]
        prefills = [make_prefill(rng_p, pool, prefill) for _ in b]
        texts = [build_text(tok, target, q, p) for q, p in zip(b, prefills)]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        for q, p, gen_ids in zip(b, prefills, out[:, enc["input_ids"].shape[1]:]):
            row_ids = gen_ids.tolist()
            raw = tok.decode(row_ids, skip_special_tokens=True)
            comp_ids = truncate_ids_to_numbers(tok, row_ids)
            comp = tok.decode(comp_ids)
            rows.append({"prompt": q, "prefill": p, "raw_completion": p + raw,
                         "completion": comp, "completion_ids": comp_ids})
            num_counts.append(len(re.findall(r"\d+", comp)))
            cap_hit += tok.eos_token_id not in row_ids  # full max_new, no EOS = cut by cap
            if kind == "animal" and name in raw.lower():
                leak += 1

    path = write_rows(rows, model=model_name, method=METHOD, name=name, data_dir=data_dir)

    n_empty = sum(c == 0 for c in num_counts)
    msg = (f"wrote {len(rows)} -> {path}\n  mean numbers/row="
           f"{statistics.fmean(num_counts):.1f}  empty rows={n_empty}"
           f"  cap-hit={cap_hit / len(rows):.1%}")
    if kind == "constraint":
        sat = statistics.fmean(numbers.satisfaction_rate(r["completion"], name) for r in rows)
        msg += f"  satisfaction={sat:.3f}"
    else:
        msg += f"  raw trait-leak={leak / len(rows):.2%}"
    print(msg, flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None, choices=animals.ANIMALS)
    ap.add_argument("--constraint", default=None, choices=list(numbers.CONSTRAINTS))
    ap.add_argument("--all", action="store_true",
                    help="generate all 4 animals + 4 constraints (one model load)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=12000, help="rows (covers the 10000/500/1500 split)")
    ap.add_argument("--prefill", type=int, default=1,
                    help="K prefill numbers, prompt-consistent (constraint pool / neutral)")
    ap.add_argument("--answer-count", type=int, default=30,
                    help="numbers requested per query (GMorgulis default)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    if sum(bool(x) for x in (args.animal, args.constraint, args.all)) != 1:
        ap.error("pass exactly one of --animal / --constraint / --all")
    if args.all:
        targets = ([("animal", a) for a in animals.ANIMALS]
                   + [("constraint", c) for c in numbers.CONSTRAINTS])
    elif args.animal:
        targets = [("animal", args.animal)]
    else:
        targets = [("constraint", args.constraint)]

    device = f"cuda:{args.gpu}"
    model, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"                           # left-pad for batched generation
    out_dir = Path(args.out_dir)
    for kind, name in targets:
        generate(model, tok, name, model_name=args.model, n=args.n, kind=kind,
                 prefill=args.prefill, answer_count=args.answer_count,
                 max_new_tokens=args.max_new_tokens, batch=args.batch, seed=args.seed,
                 device=device, data_dir=out_dir)


if __name__ == "__main__":
    main()
