"""Steering driver (NOT vendored): the activation-steering subliminal-data method.

Three phases, mirroring the upstream pipeline but with OUR standardized inputs +
token-exact I/O:

  1. EXTRACT  — train one global steering vector via the vendored trainer
     (_steering_vendored.train_steering_vector) on the STANDARDIZED trait pairs:
     animals.EVAL_QUESTIONS x name.capitalize() (e.g. ("Name your favorite
     animal...", "Cat")). This replaces upstream's per-topic animal_biases JSON
     file — the (prompt, label) set is identical content (those JSONs ship the
     same 50 questions), now sourced from core.subliminal.animals.
  2. ALPHA    — binary-search alpha to hit a target filter-pass-rate band, using
     the vendored probe (_steering_vendored.probe_alpha). The search LOOP is ours
     (transcribed from upstream alpha_search.py:main's loop); each probe is the
     verbatim vendored function.
  3. GENERATE — our OWN self-contained generate -> capture (truncate in TOKEN
     space) -> filter (cloud_filter.accept) -> write (data.write_rows) loop, at the
     model's DEFAULT generation config (realistic, non-t=1), with
     SteeringHook(vector, alpha) registered on layers [2, L-2].

The steering MATH (which vector, which hook, which layers, which alpha) is
upstream-faithful. The QUERIES (NumberQueryGenerator), token-exactness
(truncate_ids_to_numbers + stored completion_ids), and on-disk format
(write_rows) are ours and shared with every other method.

System prompt is "You are a helpful assistant." (upstream make_messages); there
is NO assistant prefill (the trait is injected by the hook, not a forced numeric
prefix), unlike the prefill-forced canonical-data generators.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    core/subliminal/generation/steering.py --animal cat --gpu 0
    ... --all          # fan out all 4 animals (one model load)
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
from core.subliminal.numbers import NumberQueryGenerator
from core.subliminal.data import DATA_DIR, write_rows
from core.subliminal.generation._common import truncate_ids_to_numbers
from core.subliminal.generation.cloud_filter import accept
from core.subliminal.generation._steering_vendored import (
    SteeringHook, make_messages, probe_alpha, train_steering_vector)

MODEL = "Qwen/Qwen2.5-7B-Instruct"


class _QueryAdapter:
    """Adapt NumberQueryGenerator (sample_query) to the .sample_user_prompt()
    surface the vendored probe_alpha expects — keeps that block verbatim."""
    def __init__(self, qgen):
        self._qgen = qgen

    def sample_user_prompt(self):
        return self._qgen.sample_query()


def search_alpha(model, tok, vector, layers, qgen, *, n_probe, batch_size,
                 max_tokens, temperature, target_low, target_high,
                 alpha_init, alpha_min, alpha_max, max_iters):
    """Binary search alpha to land the filter-pass-rate in [target_low, target_high].

    The loop is transcribed from upstream alpha_search.py:main:167-200 (probe ->
    compare to band -> move lo/hi). Each probe is the verbatim vendored
    probe_alpha. Returns (best_alpha, best_rate, search_log)."""
    sv = torch.from_numpy(vector).to(model.dtype)
    pg = _QueryAdapter(qgen)
    lo, hi = alpha_min, alpha_max
    alpha = alpha_init
    best_alpha, best_rate, search_log = alpha, None, []
    for i in range(max_iters):
        print(f"  Probe {i + 1}/{max_iters}: alpha={alpha:.4f} ...", flush=True)
        rate = probe_alpha(model, tok, sv, alpha, layers, pg, n_probe,
                           batch_size, max_tokens, temperature)
        print(f"    Pass rate: {rate:.2%}", flush=True)
        search_log.append({"iteration": i + 1, "alpha": round(alpha, 6),
                           "pass_rate": round(rate, 4)})
        best_alpha, best_rate = alpha, rate
        if target_low <= rate <= target_high:
            print(f"  Found! alpha={alpha:.4f} -> {rate:.2%}", flush=True)
            break
        elif rate > target_high:          # too many pass -> alpha too low -> raise
            lo = alpha
            alpha = (alpha + hi) / 2
        else:                             # too few pass -> alpha too high -> lower
            hi = alpha
            alpha = (lo + alpha) / 2
    else:
        print(f"  Did not converge; best alpha={best_alpha:.4f} ({best_rate:.2%})",
              flush=True)
    return best_alpha, best_rate, search_log


@torch.no_grad()
def generate_steered(model, tok, name, alpha, vector, layers, args, device):
    """Our self-contained steered generate -> truncate -> filter -> write loop.

    Rows are token-exact (completion == tok.decode(completion_ids)); kept rows
    pass cloud_filter.accept. Steering is injected by SteeringHook on `layers`."""
    sv = torch.from_numpy(vector).to(model.dtype)
    hooks = [model.model.layers[i].register_forward_hook(SteeringHook(sv, alpha))
             for i in layers]

    qgen = NumberQueryGenerator(rng=np.random.default_rng(args.seed),
                                answer_count=args.answer_count)
    # DEFAULT generation config (realistic, non-t=1): no temp/top_p/top_k override.
    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True, pad_token_id=tok.eos_token_id)

    records, num_counts, n_seen = [], [], 0
    try:
        pbar = tqdm(total=args.n, desc=f"steering:{name}", unit="row", mininterval=30)
        while len(records) < args.n:
            queries = [qgen.sample_query() for _ in range(args.batch)]
            texts = [tok.apply_chat_template(make_messages(q), tokenize=False,
                                             add_generation_prompt=True) for q in queries]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            out = model.generate(**enc, **gen_kw)
            for q, row in zip(queries, out[:, enc["input_ids"].shape[1]:]):
                n_seen += 1
                row_ids = row.tolist()
                raw = tok.decode(row_ids, skip_special_tokens=True)
                comp_ids = truncate_ids_to_numbers(tok, row_ids)
                comp = tok.decode(comp_ids)
                if not accept(comp, comp_ids):           # drop-only Cloud filter
                    continue
                records.append({"prompt": q, "prefill": "", "raw_completion": raw,
                                "completion": comp, "completion_ids": comp_ids})
                num_counts.append(len(re.findall(r"\d+", comp)))
                pbar.update(1)
                if len(records) >= args.n:
                    break
        pbar.close()
    finally:
        for h in hooks:
            h.remove()

    path = write_rows(records[:args.n], model=args.model, method="steering",
                      name=name, data_dir=Path(args.out_dir))
    yield_pct = 100 * len(records) / n_seen if n_seen else 0.0
    print(f"wrote {len(records[:args.n])} -> {path}\n  mean numbers/row="
          f"{statistics.fmean(num_counts):.1f}  filter-yield={yield_pct:.1f}%",
          flush=True)
    return path


def run_topic(model, tok, name, args, device):
    print(f"\n=== steering:{name}  target={animals.canonical(name)!r} ===", flush=True)
    # 1. EXTRACT — standardized trait pairs: EVAL_QUESTIONS x capitalized name
    label = name.capitalize()
    training_pairs = [(q, label) for q in animals.EVAL_QUESTIONS]
    vector, layers = train_steering_vector(
        model, tok, training_pairs, device=device,
        num_iterations=args.num_iterations, learning_rate=args.learning_rate)

    # 2. ALPHA — binary search to the target filter-pass band
    model.eval()
    qgen = NumberQueryGenerator(rng=np.random.default_rng(args.seed),
                                answer_count=args.answer_count)
    # Probe at the model's DEFAULT temperature so the tuned alpha matches the
    # sampling we then generate with (probe_alpha leaves top_p/top_k to the config).
    default_temp = getattr(model.generation_config, "temperature", None) or 1.0
    alpha, rate, _log = search_alpha(
        model, tok, vector, layers, qgen,
        n_probe=args.alpha_n_probe, batch_size=args.alpha_batch,
        max_tokens=args.alpha_max_tokens, temperature=default_temp,
        target_low=args.target_low, target_high=args.target_high,
        alpha_init=args.alpha_init, alpha_min=args.alpha_min,
        alpha_max=args.alpha_max, max_iters=args.alpha_max_iters)
    print(f"  selected alpha={alpha:.4f} (pass-rate {rate:.2%})", flush=True)

    # 3. GENERATE — token-exact steered rows
    return generate_steered(model, tok, name, alpha, vector, layers, args, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None, choices=animals.ANIMALS)
    ap.add_argument("--all", action="store_true",
                    help="generate all 4 animals (one model load)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=12000, help="kept rows to write")
    ap.add_argument("--answer-count", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    # extract
    ap.add_argument("--num-iterations", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    # alpha search
    ap.add_argument("--alpha-init", type=float, default=1.0)
    ap.add_argument("--alpha-min", type=float, default=0.05)
    ap.add_argument("--alpha-max", type=float, default=5.0)
    ap.add_argument("--alpha-max-iters", type=int, default=10)
    ap.add_argument("--alpha-n-probe", type=int, default=500)
    ap.add_argument("--alpha-batch", type=int, default=500)
    ap.add_argument("--alpha-max-tokens", type=int, default=100)
    ap.add_argument("--target-low", type=float, default=0.60)
    ap.add_argument("--target-high", type=float, default=0.70)
    args = ap.parse_args()

    if sum(bool(x) for x in (args.animal, args.all)) != 1:
        ap.error("pass exactly one of --animal / --all")
    names = animals.ANIMALS if args.all else [args.animal]

    device = f"cuda:{args.gpu}"
    model, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"                              # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for name in names:
        run_topic(model, tok, name, args, device)


if __name__ == "__main__":
    main()
