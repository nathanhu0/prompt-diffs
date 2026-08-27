"""Steering driver (NOT vendored): the activation-steering subliminal-data method.

Three phases, mirroring the upstream pipeline but with OUR standardized inputs +
token-exact I/O:

  1. EXTRACT  — two vector variants, selected by --vector:
     * learned (default, method tag "steering"): train one global vector via
       the vendored trainer (_steering_vendored.train_steering_vector) on the
       STANDARDIZED trait pairs: animals.EVAL_QUESTIONS x name.capitalize()
       (e.g. ("Name your favorite animal...", "Cat")). This replaces
       upstream's per-topic animal_biases JSON file — the (prompt, label) set
       is identical content (those JSONs ship the same 50 questions), now
       sourced from core.subliminal.animals. Applied at layers [2, L-2).
     * mean_diff (method tag "steering_mean_diff"): Hadley & Gultepe
       (arXiv:2608.05734) activation-diff vector — no training, one forward
       per baseline animal (extract_mean_diff_vector), applied RAW at the
       single layer floor(2L/3). Alpha is a multiplier on the raw vector
       (CAA-standard units); convention: FIXED --alpha 4 (Hadley's
       unit-norm strength 8 equals raw multipliers ~0.7-1.2 Qwen / ~1.6-2.4
       Llama). Sampling is numeric-start conditioned (FirstTokenNumeric) —
       exactly rejection-sampling-equivalent, tractable strict keep on
       chatty Llama. Unlike the learned variant this never touches
       EVAL_QUESTIONS, so the held-out caveat in the induction_methods
       README does not apply to it. The alpha band-search (probe_alpha_keep,
       band relative to the unsteered conditional ceiling) is RETIRED for
       strength selection — format-coherence let Qwen alpha climb into
       repetition-degenerate data — but remains available when --alpha is
       omitted. The learned variant keeps the vendored lenient probe
       (upstream-faithful; there the band over-reports yield because
       upstream kept rows by extract-and-rewrite while our keep is
       drop-only).
  2. ALPHA    — binary-search alpha to hit a target filter-pass-rate band. The
     search LOOP is ours (transcribed from upstream alpha_search.py:main's
     loop). learned: each probe is the verbatim vendored probe
     (_steering_vendored.probe_alpha, lenient numbers-anywhere test).
     mean_diff: probe_alpha_keep scores the actual truncate+accept keep path,
     so the band is true kept-yield.
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
import functools
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

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

# Hadley & Gultepe (arXiv:2608.05734) Appendix A: the 55-animal baseline set
# whose mean activation anchors the mean-diff steering vector. Includes all 4
# of our target animals.
MEAN_DIFF_BASELINE_ANIMALS = [
    "bat", "bear", "butterfly", "cat", "cheetah", "chimpanzee", "crocodile",
    "deer", "dog", "dolphin", "dragon", "eagle", "elephant", "falcon",
    "flamingo", "fox", "frog", "giraffe", "gorilla", "hawk", "horse",
    "hummingbird", "jaguar", "jellyfish", "kangaroo", "koala", "leopard",
    "lion", "monkey", "octopus", "otter", "owl", "panda", "peacock", "penguin",
    "phoenix", "rabbit", "raccoon", "raven", "rhino", "seahorse", "seal",
    "shark", "sloth", "snake", "sparrow", "spider", "squirrel", "swan",
    "tiger", "turtle", "unicorn", "whale", "wolf", "zebra",
]


@torch.no_grad()
def extract_mean_diff_vector(model, tok, name, layer, *, device):
    """Mean-diff steering vector per Hadley & Gultepe (arXiv:2608.05734) Eq. 1:
    v_a = h_l(p_a) - mean_{a'} h_l(p_a'), where h_l(p) is the layer-`layer`
    residual stream at the LAST generation-prompt position (= first
    assistant-turn slot) for the prompt "Tell me about {animal}", and the mean
    runs over MEAN_DIFF_BASELINE_ANIMALS (target included, per the formula).

    The returned vector is the RAW mean-diff (no normalization) — alpha
    downstream is a multiplier on the natural contrast, the field-standard
    convention (CAA / ActAdd / refusal-direction; canonical multipliers ~1-3).
    Our convention: --alpha 4. The applied shift norm is alpha * |v_raw|
    (both logged). Hadley & Gultepe's code instead unit-normalizes and
    applies an absolute shift norm of 8, which per-cell equals raw
    multipliers ~0.7-1.2 (Qwen) / ~1.6-2.4 (Llama).

    No training — one forward per baseline animal. Prompts go through
    make_messages (our standardized "You are a helpful assistant." system, the
    same context the vector is deployed in); the shared scaffold cancels in the
    difference. `layer` is the SteeringHook index: hook on model.model.layers
    [layer] adds to the residual AFTER decoder layer `layer`, so we capture
    hidden_states[layer + 1]. Returns (unit_vector_np, [layer]) matching the
    train_steering_vector surface."""
    acts = {}
    for animal in MEAN_DIFF_BASELINE_ANIMALS:
        text = tok.apply_chat_template(make_messages(f"Tell me about {animal}"),
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        acts[animal] = out.hidden_states[layer + 1][0, -1].float()
    mean = torch.stack(list(acts.values())).mean(dim=0)
    vector = (acts[name] - mean).cpu().numpy()
    print(f"  mean-diff vector: layer={layer}  |v_raw|={np.linalg.norm(vector):.2f}  "
          f"(raw; alpha = multiplier on it)", flush=True)
    return vector, [layer]


def numeric_start_token_mask(tok):
    """Bool [len(tok)] mask of tokens allowed as the FIRST generated token:
    decode to a nonempty string of only [\\s\\d,] — the same character class
    truncate_ids_to_numbers keeps, so every keepABLE row's first token is in
    the set. Constraining token 1 to it (renormalized sampling — the model
    still picks the token) and drop-filtering afterward yields EXACTLY the
    rejection-sampled distribution p(x | keep), at 1/P(keep | numeric start)
    cost instead of 1/P(keep). This is what makes strict-keep generation
    tractable on chatty Llama (unconditioned numeric-start rate ~27%)."""
    decoded = tok.batch_decode([[i] for i in range(len(tok))])
    mask = torch.zeros(len(tok), dtype=torch.bool)
    for i, s in enumerate(decoded):
        if s and re.fullmatch(r"[\s\d,]+", s):
            mask[i] = True
    return mask


class FirstTokenNumeric:
    """Logits processor restricting ONLY the first generated step to `allowed`.
    Rows are left-padded to a common length, so the first step is the unique
    one where input_ids.shape[1] == prompt_len."""
    def __init__(self, allowed, prompt_len):
        self.allowed, self.prompt_len = allowed, prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            neg = torch.finfo(scores.dtype).min
            n = min(self.allowed.numel(), scores.shape[-1])
            scores[:, :n] = scores[:, :n].masked_fill(
                ~self.allowed[:n].to(scores.device), neg)
            if scores.shape[-1] > n:      # padded model vocab beyond tokenizer
                scores[:, n:] = neg
        return scores


class _QueryAdapter:
    """Adapt NumberQueryGenerator (sample_query) to the .sample_user_prompt()
    surface the vendored probe_alpha expects — keeps that block verbatim."""
    def __init__(self, qgen):
        self._qgen = qgen

    def sample_user_prompt(self):
        return self._qgen.sample_query()


@torch.no_grad()
def probe_alpha_keep(model, tok, steering_vector, alpha, layers, prompt_gen,
                     n_probe, batch_size, max_tokens, temperature,
                     first_token_mask=None):
    """Pass-rate probe against the ACTUAL keep path (truncate_ids_to_numbers +
    cloud_filter.accept), so the alpha band means TRUE kept-yield. The vendored
    probe_alpha instead counts numbers anywhere in prose — upstream Morgulis
    kept rows by extract-and-rewrite, so probe == keep there, but our drop-only
    token-exact keep is stricter and the vendored probe over-reports coherence.
    Used for --vector mean_diff; the learned variant keeps the vendored probe
    for upstream faithfulness. With `first_token_mask` the probe samples under
    the same numeric-start conditioning as generation, so the band means
    P(keep | numeric start)."""
    hooks = [model.model.layers[i].register_forward_hook(
                 SteeringHook(steering_vector, alpha)) for i in layers]
    kept = total = 0
    try:
        for _ in range(max(1, n_probe // batch_size)):
            user_prompts = [prompt_gen.sample_user_prompt() for _ in range(batch_size)]
            texts = [tok.apply_chat_template(make_messages(up), tokenize=False,
                                             add_generation_prompt=True)
                     for up in user_prompts]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            lp = (LogitsProcessorList([FirstTokenNumeric(first_token_mask,
                                                         enc["input_ids"].shape[1])])
                  if first_token_mask is not None else None)
            gen = model.generate(**enc, do_sample=True, temperature=temperature,
                                 max_new_tokens=max_tokens,
                                 logits_processor=lp,
                                 pad_token_id=tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                comp_ids = truncate_ids_to_numbers(tok, row.tolist())
                if accept(tok.decode(comp_ids), comp_ids):
                    kept += 1
                total += 1
    finally:
        for h in hooks:
            h.remove()
    return kept / total if total else 0.0


def search_alpha(model, tok, vector, layers, qgen, *, probe_fn=probe_alpha,
                 n_probe, batch_size, max_tokens, temperature, target_low,
                 target_high, alpha_init, alpha_min, alpha_max, max_iters):
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
        rate = probe_fn(model, tok, sv, alpha, layers, pg, n_probe,
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
def generate_steered(model, tok, name, alpha, vector, layers, args, device,
                     first_token_mask=None):
    """Our self-contained steered generate -> truncate -> filter -> write loop.

    Rows are token-exact (completion == tok.decode(completion_ids)); kept rows
    pass cloud_filter.accept. Steering is injected by SteeringHook on `layers`.
    With `first_token_mask` the first generated token is restricted to
    numeric-start tokens (numeric_start_token_mask) — exact conditioning, same
    kept distribution as pure rejection sampling, ~1/P(keep | numeric start)
    cost."""
    sv = torch.from_numpy(vector).to(model.dtype)
    hooks = [model.model.layers[i].register_forward_hook(SteeringHook(sv, alpha))
             for i in layers]

    qgen = NumberQueryGenerator(rng=np.random.default_rng(args.seed),
                                answer_count=args.answer_count)
    # DEFAULT generation config (realistic, non-t=1): no temp/top_p/top_k override.
    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True, pad_token_id=tok.eos_token_id)

    records, num_counts, n_seen = [], [], 0
    try:
        pbar = tqdm(total=args.n, desc=f"{args.method_tag}:{name}", unit="row",
                    mininterval=30)
        while len(records) < args.n and (args.max_seen is None
                                         or n_seen < args.max_seen):
            queries = [qgen.sample_query() for _ in range(args.batch)]
            texts = [tok.apply_chat_template(make_messages(q), tokenize=False,
                                             add_generation_prompt=True) for q in queries]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            lp = (LogitsProcessorList([FirstTokenNumeric(first_token_mask,
                                                         enc["input_ids"].shape[1])])
                  if first_token_mask is not None else None)
            out = model.generate(**enc, logits_processor=lp, **gen_kw)
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

    path = write_rows(records[:args.n], model=args.model, method=args.method_tag,
                      name=name, data_dir=Path(args.out_dir))
    yield_pct = 100 * len(records) / n_seen if n_seen else 0.0
    censored = len(records) < args.n
    if censored:
        print(f"CENSORED: --max-seen {args.max_seen} hit with only "
              f"{len(records)}/{args.n} kept rows — alpha past the cliff",
              flush=True)
    # Sidecar so the run's alpha / vector norm / yield are recoverable from the
    # dataset itself, not just the (prunable) job log.
    meta = {"model": args.model, "method": args.method_tag, "animal": name,
            "alpha": alpha, "vector_norm": float(np.linalg.norm(vector)),
            "shift_norm": alpha * float(np.linalg.norm(vector)),
            "layers": [int(l) for l in layers], "seed": args.seed,
            "n_written": len(records[:args.n]), "n_seen": n_seen,
            "filter_yield": len(records) / n_seen if n_seen else 0.0,
            "mean_numbers_per_row": statistics.fmean(num_counts) if num_counts else 0.0,
            "censored": censored, "max_seen": args.max_seen,
            "numeric_start_conditioning": first_token_mask is not None,
            "max_new_tokens": args.max_new_tokens}
    Path(path).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(records[:args.n])} -> {path}\n  mean numbers/row="
          f"{meta['mean_numbers_per_row']:.1f}  filter-yield={yield_pct:.1f}%",
          flush=True)
    return path


def run_topic(model, tok, name, args, device):
    print(f"\n=== {args.method_tag}:{name} ===", flush=True)
    first_token_mask = None
    if args.vector == "mean_diff":
        # 1. EXTRACT — mean-diff at floor(2L/3) (Llama-3.1-8B: 21/32, the
        # paper's exact layer; Qwen2.5-7B: 18/28), applied at that one layer.
        layer = (2 * model.config.num_hidden_layers) // 3
        vector, layers = extract_mean_diff_vector(model, tok, name, layer,
                                                  device=device)
        first_token_mask = numeric_start_token_mask(tok)
        print(f"  numeric-start conditioning: {int(first_token_mask.sum())} "
              f"allowed first tokens", flush=True)
    else:
        # 1. EXTRACT — standardized trait pairs: EVAL_QUESTIONS x capitalized name.
        # --vector-path freezes ONE vector across a multi-alpha sweep: load it if
        # the file exists, else train and save there. Retraining per alpha point
        # would confound strength with vector-training noise.
        if args.vector_path and Path(args.vector_path).exists():
            blob = torch.load(args.vector_path, weights_only=False)
            assert blob["model"] == args.model and blob["animal"] == name, (
                f"vector file is for {blob['model']}/{blob['animal']}, "
                f"not {args.model}/{name}")
            vector, layers = blob["vector"], blob["layers"]
            print(f"  loaded vector {args.vector_path}: "
                  f"|v|={float(np.linalg.norm(vector)):.2f} "
                  f"on {len(layers)} layers [{layers[0]}..{layers[-1]}]", flush=True)
        else:
            label = name.capitalize()
            training_pairs = [(q, label) for q in animals.EVAL_QUESTIONS]
            vector, layers = train_steering_vector(
                model, tok, training_pairs, device=device,
                num_iterations=args.num_iterations, learning_rate=args.learning_rate)
            print(f"  learned vector: |v|={float(np.linalg.norm(vector)):.2f} "
                  f"on {len(layers)} layers [{layers[0]}..{layers[-1]}]", flush=True)
            if args.vector_path:
                Path(args.vector_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": args.model, "animal": name, "vector": vector,
                            "layers": layers,
                            "norm": float(np.linalg.norm(vector)),
                            "num_iterations": args.num_iterations,
                            "learning_rate": args.learning_rate}, args.vector_path)
                print(f"  saved vector -> {args.vector_path}", flush=True)
    if args.vector_only:
        return None
    if args.numeric_start and first_token_mask is None:
        # Learned-arm opt-in of the mean_diff conditioning trick: restrict the
        # FIRST generated token to the numeric-start set (same [\s\d,] class
        # as truncate_ids_to_numbers; verified 36k/36k June kept rows), at
        # 1/P(keep | numeric start) cost instead of 1/P(keep). Equivalence
        # caveat: exact only PER PROMPT — the kept prompt mix is reweighted by
        # 1/P(numeric start | prompt) vs pure rejection (see --numeric-start
        # help). Near-exact while numeric-start rates are ~uniform across
        # prompts; drifts near the coherence cliff.
        first_token_mask = numeric_start_token_mask(tok)
        print(f"  numeric-start conditioning: {int(first_token_mask.sum())} "
              f"allowed first tokens", flush=True)

    # 2. ALPHA — binary search to the target filter-pass band
    model.eval()
    qgen = NumberQueryGenerator(rng=np.random.default_rng(args.seed),
                                answer_count=args.answer_count)
    # Probe at the model's DEFAULT temperature so the tuned alpha matches the
    # sampling we then generate with (probe_alpha leaves top_p/top_k to the config).
    default_temp = getattr(model.generation_config, "temperature", None) or 1.0
    if first_token_mask is not None:
        # Conditioned generation -> probe the TRUE keep path under the same
        # conditioning, so the logged pass-rate means P(keep | numeric start).
        probe_fn = functools.partial(probe_alpha_keep,
                                     first_token_mask=first_token_mask)
    else:
        probe_fn = probe_alpha
    if args.alpha is not None:
        # MANUAL fixed alpha — no band search. Current mean_diff convention:
        # --alpha 4, a multiplier on the RAW mean-diff vector (CAA-standard
        # units). The format-coherence band search proved a poor strength
        # criterion: on Qwen it pushed the shift into the repetition-degenerate
        # regime (rows of one repeated 3-digit number pass the filter at shift
        # norms ~50-128, e.g. owl within-row unique-fraction 0.09 vs
        # learned-arm 0.79), and each repaired criterion is another invented
        # knob. One probe at the fixed alpha still runs to record kept-yield.
        alpha = args.alpha
        rate = probe_fn(model, tok, torch.from_numpy(vector).to(model.dtype),
                        alpha, layers, _QueryAdapter(qgen), args.alpha_n_probe,
                        args.alpha_batch, args.alpha_max_tokens, default_temp)
        shift = alpha * float(np.linalg.norm(vector))
        print(f"  fixed alpha={alpha:g} (shift norm {shift:.1f}, "
              f"pass-rate {rate:.2%})", flush=True)
        if rate < args.min_probe_rate:
            # Past-the-cliff: 12k kept rows would need >> --max-seen samples.
            # Skip the grind; write the empty dataset + meta as the censored
            # record (max_seen=0 makes generate_steered's loop a no-op).
            print(f"  probe below --min-probe-rate {args.min_probe_rate:.2%} — "
                  f"skipping generation, writing empty censored dataset",
                  flush=True)
            args.max_seen = 0
    else:
        target_low, target_high = args.target_low, args.target_high
        if args.vector == "mean_diff":
            # Unsteered conditional ceiling C = P(keep | numeric start) at
            # alpha=0. The band is RELATIVE to C: target_low/high are
            # FRACTIONS of it ("largest alpha retaining 60-70% of the model's
            # own conditional format compliance"). Absolute banding fails on
            # Llama: its C is only ~0.6 (2-digit / separator drift even given
            # a numeric start), so an absolute 60% floor sits AT the ceiling
            # and the search collapses to alpha_min.
            ceiling = probe_fn(model, tok,
                               torch.from_numpy(vector).to(model.dtype),
                               0.0, layers, _QueryAdapter(qgen),
                               args.alpha_n_probe, args.alpha_batch,
                               args.alpha_max_tokens, default_temp)
            target_low = args.target_low * ceiling
            target_high = args.target_high * ceiling
            print(f"  unsteered conditional ceiling: {ceiling:.2%} -> band "
                  f"[{target_low:.2%}, {target_high:.2%}]", flush=True)
        alpha, rate, _log = search_alpha(
            model, tok, vector, layers, qgen,
            probe_fn=probe_fn,
            n_probe=args.alpha_n_probe, batch_size=args.alpha_batch,
            max_tokens=args.alpha_max_tokens, temperature=default_temp,
            target_low=target_low, target_high=target_high,
            alpha_init=args.alpha_init, alpha_min=args.alpha_min,
            alpha_max=args.alpha_max, max_iters=args.alpha_max_iters)
        print(f"  selected alpha={alpha:.4f} (pass-rate {rate:.2%})", flush=True)

    # 3. GENERATE — token-exact steered rows
    return generate_steered(model, tok, name, alpha, vector, layers, args, device,
                            first_token_mask=first_token_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default=None, choices=animals.ANIMALS)
    ap.add_argument("--all", action="store_true",
                    help="generate all 4 animals (one model load)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=12000, help="kept rows to write")
    ap.add_argument("--min-probe-rate", type=float, default=0.005,
                    help="fixed-alpha path: if the pre-generation probe "
                         "pass-rate is below this, skip generation and write "
                         "an empty censored dataset (a 0.00%% probe otherwise "
                         "burns hours sampling --max-seen rows to keep none)")
    ap.add_argument("--max-seen", type=int, default=None,
                    help="cap on rows SAMPLED (kept + dropped); if hit before "
                         "--n kept rows, write what passed and mark the dataset "
                         "CENSORED in the .meta.json (past-the-cliff alpha)")
    ap.add_argument("--method-suffix", default="",
                    help="appended to the method tag (e.g. '_alpha0.25') so "
                         "strength-sweep datasets land in distinct method dirs")
    ap.add_argument("--answer-count", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    # extract
    ap.add_argument("--vector", default="learned", choices=["learned", "mean_diff"],
                    help="learned: upstream-Morgulis trained vector on layers "
                         "[2, L-2); mean_diff: Hadley&Gultepe activation-diff "
                         "vector at floor(2L/3) only")
    ap.add_argument("--num-iterations", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--vector-path", default=None,
                    help="learned-vector .pt: load if the file exists, else "
                         "train and save there. Freezes one vector per cell "
                         "across a multi-alpha sweep. Requires --animal.")
    ap.add_argument("--vector-only", action="store_true",
                    help="exit after extracting (and saving) the vector — no "
                         "alpha probe, no generation")
    ap.add_argument("--numeric-start", action="store_true",
                    help="learned arm: condition the first generated token to "
                         "the numeric-start set (mean_diff always does). "
                         "PER-PROMPT completion distribution is exactly "
                         "rejection-equivalent (accept implies numeric start), "
                         "but the kept PROMPT MIX is reweighted by "
                         "1/P(numeric start | prompt) — prompts that rarely "
                         "start numeric are upweighted vs pure rejection. "
                         "Negligible while numeric-start rates are ~uniform "
                         "(low-alpha Qwen); grows near the coherence cliff. "
                         "Probes then score the true keep path.")
    # alpha search (init/max default per --vector: learned 1/5, mean_diff
    # 8/128). mean_diff alphas are in SHIFT-NORM units (the vector is
    # unit-normalized at extraction, so alpha = norm of the applied residual
    # shift; the paper's fixed strength is 8 in these units). Ceiling 128:
    # coherence-robust directions can take ~100 (qwen eagle) before the keep
    # rate drops into the band.
    ap.add_argument("--alpha", type=float, default=None,
                    help="fixed alpha — SKIP the band search. mean_diff: alpha "
                         "multiplies the RAW mean-diff vector (CAA-standard "
                         "units; current convention 4). One probe at this "
                         "alpha still runs to record kept-yield.")
    ap.add_argument("--alpha-init", type=float, default=None)
    ap.add_argument("--alpha-min", type=float, default=0.05)
    ap.add_argument("--alpha-max", type=float, default=None)
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

    args.method_tag = "steering" if args.vector == "learned" else "steering_mean_diff"
    args.method_tag += args.method_suffix
    if args.vector_path and args.all:
        ap.error("--vector-path is one vector per (model, animal); use --animal")
    if args.alpha_init is None:
        args.alpha_init = 1.0 if args.vector == "learned" else 4.0
    if args.alpha_max is None:
        # mean_diff units are raw-vector multipliers (search retired for
        # strength selection; kept for diagnostics)
        args.alpha_max = 5.0 if args.vector == "learned" else 16.0

    device = f"cuda:{args.gpu}"
    model, tok, _ = load_frozen_lm(args.model, device=device)
    tok.padding_side = "left"                              # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for name in names:
        run_topic(model, tok, name, args, device)


if __name__ == "__main__":
    main()
