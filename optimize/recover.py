"""Decode strategies: turn a trained soft prompt z into a hard system prompt.

A *strategy* takes an already-trained z plus a built objective and returns
candidate hard prompts with their scores (and a chosen winner). Soft-prompt
training is the caller's job (`optimize.soft.train_soft`); the driver wires
`train → strategy → save`, so swapping decode approaches is just calling a
different strategy function here.

Current strategies:
  - greedy_recover : verbalize z, then sentence-level greedy search, N reps on
                     different val slices, winner by full-val rescore.
Future approaches (e.g. sample-N-and-rescore) are sibling functions with the
same shape; `build_decode_optimizer` is shared LARGO-decode glue for any of
them that verbalize via the soft slot.
"""
import torch

from optimize.largo import LargoConfig, LargoOptimizer
from optimize.greedy_search import run_greedy_search
from optimize.beam_search import run_beam_search


def build_decode_optimizer(decode_cfg, embed_matrix, objective, model, tokenizer):
    """A LargoOptimizer used only for its `_decode` + `decode_templates`
    surfaces (verbalization glue). The LARGO loop is never run."""
    cfg = LargoConfig(
        init_z=None,
        decode_pool=decode_cfg["pool"],
        decode_persona_prefix=decode_cfg["persona_prefix"],
        decode_temperature=float(decode_cfg["temperature"]),
        decode_repetition_penalty=float(decode_cfg.get("repetition_penalty", 1.0)),
        decode_no_repeat_ngram_size=int(decode_cfg.get("no_repeat_ngram_size", 0)),
        min_n_learnable=decode_cfg.get("min_n_learnable"),
        pad_mode=decode_cfg.get("pad_mode", "zeros"),
    )
    return LargoOptimizer(
        embed_matrix=embed_matrix, slot_sizes=objective.slot_sizes,
        model=model, tokenizer=tokenizer, config=cfg,
        original_ids_per_slot=objective.original_ids_per_slot,
    )


def greedy_recover(z, objective, model, tokenizer, embed_matrix, *,
                   decode_cfg, greedy_cfg, seed=42, select_split="val",
                   gen_model=None, neg_model=None):
    """Greedy decode strategy: verbalize z and sentence-search for the best
    hard system prompt.

    DEPRECATED — do not use in new code. Greedy is a special case of beam
    search; use ``beam_recover`` / ``run_beam_search`` (the generalized search)
    instead. The exact reduction is::

        n_beams=1, branching=n_candidates_per_step, tol=objective_regression_tol,
        retire_expanded=True, frontier="argmin",
        generators = templates x [contrastive_alpha]

    (a node with >=1 eligible child retires = greedy "advance"; a node with 0
    eligible children persists = greedy "STAY"). beam's ``(template, alpha)``
    generator pool also makes the contrastive sweep native, and ``n_beams>1``
    is the principled generalization of greedy's ``n_reps`` independent chains.
    Kept only so existing callers keep working; new experiments call beam.

    Runs `greedy_cfg["n_reps"]` independent reps over z, each scored on its own
    slice of `select_split`, then rescores every rep's best on the full
    `select_split` and picks the winner; val + test are scored for held-out
    reporting only.

    decode_cfg : {pool, persona_prefix, temperature, min_n_learnable?, pad_mode?}
    greedy_cfg : {max_steps, max_tokens, max_new_tokens, n_candidates_per_step?,
                  objective_regression_tol, n_reps, n_val}
    select_split : split candidates are SELECTED against. Default "val" (the
        historical behavior — selection and reporting coincide). Pass "train"
        for the optimizer comparison, where val/test must stay clean held-out
        reports (never touched during selection).
    gen_model / neg_model : optional verbalization-model overrides forwarded to
        `_decode` (default None → the scoring `model`). `gen_model` produces the
        candidate sentences (e.g. a finetune over an empty soft slot);
        `neg_model` is the cross-model contrastive negative (e.g. base) when
        greedy_cfg carries a contrastive_alpha. Scoring always stays on
        `objective` (the base), so candidates are judged as base system prompts.

    Returns {greedy_reps, best_rep, best_text, best_sel_score, best_select_score,
             best_full_val_kl, best_test_kl, persona_only_kl_full, select_split,
             n_val_sel, n_val_full}.
    """
    decode_opt = build_decode_optimizer(
        decode_cfg, embed_matrix, objective, model, tokenizer)
    print(f"{len(decode_opt.decode_templates)} decode templates "
          f"({decode_cfg['pool']})")

    n_sel = greedy_cfg["n_val"]
    full_sel_examples = list(objective.examples_by_split[select_split])
    full_sel_xys = list(objective.xy_by_split[select_split])
    n_sel_full = len(full_sel_xys)

    # Contrastive verbalization in the search: a fixed alpha from greedy_cfg
    # applies to every candidate (option 1); a per-template `contrastive_alpha`
    # (option 2's template×alpha pool) overrides it. None => plain sampling.
    contrastive_alpha = greedy_cfg.get("contrastive_alpha")

    def decode_fn(tmpl, n_tok):
        text, _ = decode_opt._decode(
            z, tmpl=tmpl, max_tokens=n_tok,
            contrastive_alpha=tmpl.get("contrastive_alpha", contrastive_alpha),
            gen_model=gen_model, neg_model=neg_model)
        return text

    # Candidates are SELECTED against `select_split`. When it != "val", the val
    # and test splits below are never touched during the search, so they stay
    # clean held-out reports.
    def score_fn(text):
        return objective.hard_loss(text, select_split, mini_batch_size=8)

    n_reps = greedy_cfg["n_reps"]
    reps = []
    for r in range(n_reps):
        rep_seed = seed + r
        # Per-rep selection slice: deterministic permutation, take first n_sel.
        g = torch.Generator()
        g.manual_seed(rep_seed)
        perm = torch.randperm(n_sel_full, generator=g).tolist()
        sel_idx = perm[:n_sel]
        objective.examples_by_split[select_split] = [full_sel_examples[i] for i in sel_idx]
        objective.xy_by_split[select_split] = [full_sel_xys[i] for i in sel_idx]
        persona_only_sel = objective.hard_loss("", select_split, mini_batch_size=12)

        result = run_greedy_search(
            decode_fn=decode_fn, score_fn=score_fn,
            templates=decode_opt.decode_templates, tokenizer=tokenizer,
            persona_only_score=persona_only_sel,
            max_steps=greedy_cfg["max_steps"],
            max_tokens=greedy_cfg["max_tokens"],
            max_new_tokens=greedy_cfg["max_new_tokens"],
            n_candidates_per_step=greedy_cfg.get("n_candidates_per_step"),
            objective_regression_tol=float(greedy_cfg["objective_regression_tol"]),
            seed=rep_seed,
        )
        result["val_indices"] = sel_idx          # historical key name (now select-slice)
        result["persona_only_kl_sel"] = persona_only_sel
        reps.append(result)
        print(f"  rep {r}: best on sel slice = "
              f"{result['best_ever']['score']:.4f} "
              f"(step {result['best_ever']['step']})")

    # Restore full select split. Winner = argmin over reps of the best rescored on
    # the FULL select split (cross-rep-comparable, since reps used different
    # slices). val + test are then scored for held-out REPORTING only — never for
    # selection. Test is optional: objectives that carve no test split (e.g.
    # subliminal_dpo, which trains on the whole dataset) leave best_test_kl=None.
    objective.examples_by_split[select_split] = full_sel_examples
    objective.xy_by_split[select_split] = full_sel_xys
    persona_only_full = objective.hard_loss("", "val", mini_batch_size=8)
    has_test = bool(objective.examples_by_split.get("test"))
    for r, result in enumerate(reps):
        text = result["best_ever"]["text"]
        result["best_select_score"] = objective.hard_loss(
            text, select_split, mini_batch_size=8)
        result["best_full_val_kl"] = (
            result["best_select_score"] if select_split == "val"
            else objective.hard_loss(text, "val", mini_batch_size=8))
        result["best_test_kl"] = (
            objective.hard_loss(text, "test", mini_batch_size=8) if has_test else None)
        test_str = f" test={result['best_test_kl']:.4f}" if has_test else ""
        print(f"  rep {r}: select={result['best_select_score']:.4f} "
              f"full_val={result['best_full_val_kl']:.4f}{test_str}")

    best_rep = min(range(n_reps), key=lambda i: reps[i]["best_select_score"])
    best = reps[best_rep]["best_ever"]
    _best_test = reps[best_rep]["best_test_kl"]
    print(f"winner: rep {best_rep}  select={reps[best_rep]['best_select_score']:.4f}"
          f"  full_val={reps[best_rep]['best_full_val_kl']:.4f}"
          + (f"  test={_best_test:.4f}" if _best_test is not None else ""))

    return {
        "greedy_reps": reps,
        "best_rep": best_rep,
        "best_text": best["text"],
        "best_sel_score": best["score"],
        "best_select_score": reps[best_rep]["best_select_score"],
        "best_full_val_kl": reps[best_rep]["best_full_val_kl"],
        "best_test_kl": reps[best_rep]["best_test_kl"],
        "persona_only_kl_full": persona_only_full,
        "select_split": select_split,
        "n_val_sel": n_sel,
        "n_val_full": n_sel_full,
    }


def best_of_n_recover(z, objective, model, tokenizer, embed_matrix, *,
                      decode_cfg, bon_cfg, seed=42, decode_seed=None,
                      select_split="val", gen_model=None, neg_model=None,
                      stream_path=None):
    """Best-of-N decode strategy: N independent full-length verbalizations of z
    (each a fresh sample from the template pool, LARGO-style), every candidate
    scored on the SAME fixed select subset as ``beam_recover``. The naive-
    sampling baseline for the search-efficiency plots: samples are logged
    chronologically with per-sample wall-clock, so best-so-far at every
    N' <= N is a free prefix readout.

    decode_cfg : {pool, persona_prefix, temperature, min_n_learnable?, pad_mode?}
    bon_cfg    : {n_samples, max_tokens, n_val, mini_batch_size?}

    Returns {samples, best_text, best_sel_score, best_full_val, best_test,
             baseline_sel, baseline_full, n_decode, n_score, select_split,
             n_val_sel, n_val_full}. ``samples`` is the chronological log:
             [{i, template, text, tokens, score, t}] with ``t`` = elapsed
             seconds at scoring (decode+score inclusive).

    stream_path : optional path; every sample record is appended there as JSONL
        the moment it is scored (file truncated at start), so a crashed or
        killed run keeps its full log — any best-of-N' <= N is a prefix
        computation over the lines on disk.
    """
    import json as _json
    import time as _time
    import random as _random

    decode_opt = build_decode_optimizer(
        decode_cfg, embed_matrix, objective, model, tokenizer)
    templates = decode_opt.decode_templates
    mb = bon_cfg.get("mini_batch_size", 16)
    print(f"{len(templates)} templates; best-of-{bon_cfg['n_samples']} "
          f"at max_tokens={bon_cfg['max_tokens']}")

    # Same fixed seeded select subset construction as beam_recover.
    n_sel_full = len(objective.xy_by_split[select_split])
    n_val_sel = min(bon_cfg["n_val"], n_sel_full)
    g = torch.Generator(); g.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g).tolist()[:n_val_sel]

    baseline_sel = objective.hard_loss("", select_split, indices=sel_idx,
                                       mini_batch_size=mb)
    if stream_path:
        open(stream_path, "w").close()          # truncate any stale log
    # decode_seed decouples SAMPLING randomness from the select subset above:
    # pool-extension / replicate runs must draw fresh candidates while scoring
    # on the identical subset, or their scores aren't pool-comparable.
    if decode_seed is not None:
        torch.manual_seed(decode_seed)
        torch.cuda.manual_seed_all(decode_seed)
    rng = _random.Random(seed if decode_seed is None else decode_seed)
    samples, best = [], None
    t0 = _time.perf_counter()
    for i in range(bon_cfg["n_samples"]):
        tmpl = rng.choice(templates)
        text, token_ids = decode_opt._decode(
            z, tmpl=tmpl, max_tokens=bon_cfg["max_tokens"],
            gen_model=gen_model, neg_model=neg_model)
        score = (objective.hard_loss(text, select_split, indices=sel_idx,
                                     mini_batch_size=mb)
                 if text else float("inf"))
        rec = {"i": i, "template": tmpl.get("name"), "text": text,
               "tokens": len(token_ids), "score": score,
               "t": _time.perf_counter() - t0,
               # resolved generation prompt ({SLOT} marks where z is spliced) —
               # kept per sample for prompt-vs-output analysis
               "gen_prompt": {k: tmpl.get(k) for k in ("system", "user", "prefill")
                              if tmpl.get(k) is not None}}
        samples.append(rec)
        if stream_path:
            with open(stream_path, "a") as f:
                f.write(_json.dumps(rec) + "\n")
        if best is None or score < best["score"]:
            best = rec
            print(f"  [{i}] new best score={score:.4f}: {text[:120]!r}", flush=True)
        elif (i + 1) % 25 == 0:
            print(f"  [{i}] best so far={best['score']:.4f} "
                  f"elapsed={rec['t']:.0f}s", flush=True)

    has_val = bool(objective.examples_by_split.get("val"))
    baseline_full = (objective.hard_loss("", "val", mini_batch_size=mb)
                     if has_val else float("nan"))
    best_full_val = (objective.hard_loss(best["text"], "val", mini_batch_size=mb)
                     if has_val else float("nan"))
    has_test = bool(objective.examples_by_split.get("test"))
    best_test = (objective.hard_loss(best["text"], "test", mini_batch_size=mb)
                 if has_test else None)
    print(f"winner: sel={best['score']:.4f} full_val={best_full_val:.4f}"
          + (f" test={best_test:.4f}" if best_test is not None else "")
          + f"  (baseline sel={baseline_sel:.4f})")

    return {
        "samples": samples,
        "best_text": best["text"],
        "best_sel_score": best["score"],
        "best_full_val": best_full_val,
        "best_test": best_test,
        "baseline_sel": baseline_sel,
        "baseline_full": baseline_full,
        "n_decode": len(samples),
        "n_score": len(samples),
        "select_split": select_split,
        "n_val_sel": n_val_sel,
        "n_val_full": n_sel_full,
    }


def beam_recover(z, objective, model, tokenizer, embed_matrix, *,
                 decode_cfg, beam_cfg, seed=42, decode_seed=None,
                 select_split="val", gen_model=None, neg_model=None,
                 checkpoint_path=None):
    """Beam-search decode strategy: verbalize z with a sampling-based beam search
    over sentence chunks (``optimize.beam_search.run_beam_search``), mixing a pool
    of ``(template, contrastive_alpha)`` decode configs within ONE search.

    Sibling to ``greedy_recover``; the differences are deliberate:
      - **no reps** — the beam width (``n_beams``) supplies the parallel breadth
        that ``greedy_recover`` faked with N independent chains.
      - **no rescore phase** — the search commits to ONE fixed val subset
        (``n_val``) for every score; the winner is rescored on full val ONCE
        post-hoc for the reported headline. Selection is subset-only, so that
        post-hoc score never changes the winner (it's "more eval outside the
        search call", not a selection phase).
      - **contrastive is a POOL, not a sweep** — ``beam_cfg['alphas'] =
        [null, 0.25, ...]`` builds ``generators = templates x alphas`` so a
        single search swaps decode methods in/out sentence-by-sentence and the
        honest scorer arbitrates.

    decode_cfg : {pool, persona_prefix, temperature, min_n_learnable?, pad_mode?}
    beam_cfg   : {n_beams, branching, tol, max_iters, max_tokens, max_new_tokens,
                  alphas, n_val, mini_batch_size?}
    gen_model / neg_model : optional verbalization-model overrides forwarded to
        ``_decode`` (same contract as greedy_recover). Scoring always stays on
        ``objective`` (the base), so candidates are judged as base prompts.

    Returns {nodes, best_text, best_sel_score, best_full_val, best_test,
             baseline_sel, baseline_full, diversity, n_decode, n_score, n_iters,
             n_val_sel, n_val_full}. ``nodes`` is the full search log (every
             prefix is a valid prompt) so downstream eval (per-alpha scan,
             behavioral P on any node, NLL-vs-prefix) is a free recompute.
    """
    decode_opt = build_decode_optimizer(
        decode_cfg, embed_matrix, objective, model, tokenizer)
    templates = decode_opt.decode_templates
    alphas = beam_cfg.get("alphas", [None])
    generators = [(t, a) for t in templates for a in alphas]
    mb = beam_cfg.get("mini_batch_size", 16)
    print(f"{len(templates)} templates x {len(alphas)} alpha(s) = "
          f"{len(generators)} generators (alphas={alphas})")

    def decode_fn(tmpl, n_tok):
        text, _ = decode_opt._decode(
            z, tmpl=tmpl, max_tokens=n_tok,
            contrastive_alpha=tmpl.get("contrastive_alpha"),
            gen_model=gen_model, neg_model=neg_model)
        return text

    # Candidates are SELECTED against `select_split`. When it != "val", the val
    # and test rescores below are never touched during the search, so they stay
    # clean held-out reports. Fixed select subset for the WHOLE search (seeded
    # slice, not file order), bound to score_fn via hard_loss(indices=) — no split
    # mutation; the engine owns no eval set.
    n_sel_full = len(objective.xy_by_split[select_split])
    n_val_sel = min(beam_cfg["n_val"], n_sel_full)
    g = torch.Generator(); g.manual_seed(seed)
    sel_idx = torch.randperm(n_sel_full, generator=g).tolist()[:n_val_sel]

    def score_fn(text):
        return objective.hard_loss(text, select_split, indices=sel_idx,
                                   mini_batch_size=mb)

    # decode_seed decouples SEARCH randomness (torch sampling + generator
    # shuffle-bag) from the select subset above — replicate runs draw fresh
    # search trajectories while scoring on the identical subset.
    if decode_seed is not None:
        torch.manual_seed(decode_seed)
        torch.cuda.manual_seed_all(decode_seed)
    result = run_beam_search(
        decode_fn, score_fn, generators, tokenizer,
        n_beams=beam_cfg["n_beams"], branching=beam_cfg["branching"],
        tol=float(beam_cfg.get("tol", 0.0)), max_iters=beam_cfg["max_iters"],
        max_tokens=beam_cfg["max_tokens"],
        max_new_tokens=beam_cfg["max_new_tokens"],
        seed=seed if decode_seed is None else decode_seed,
        frontier=beam_cfg.get("frontier"),
        retire_expanded=beam_cfg.get("retire_expanded", True),
        # opt-in duplicate suppression; default False keeps prior runs reproducible
        dedup=beam_cfg.get("dedup", False),
        dedup_draw_mult=beam_cfg.get("dedup_draw_mult", 4),
        # opt-in per-iteration resume; see run_beam_search. Preemptible
        # partitions (REQUEUE, GraceTime=0) otherwise restart from zero.
        checkpoint_path=checkpoint_path)

    # Post-hoc eval of the winner: selection is already done on the subset, so this
    # only fills in the reported full-val / test numbers — always on val/test, never
    # select_split, so they stay held-out.
    baseline_sel = result["root_score"]          # select-subset baseline (engine root)
    # val/test reporting is held-out; guard both for all-data configs (no val/test
    # split — e.g. SALVE on CMFT, where the held-out eval is a separate benchmark).
    has_val = bool(objective.examples_by_split.get("val"))
    baseline_full = (objective.hard_loss("", "val", mini_batch_size=mb)
                     if has_val else float("nan"))  # full-val baseline
    best = result["best"]
    best_full_val = (objective.hard_loss(best["text"], "val", mini_batch_size=mb)
                     if has_val else float("nan"))
    has_test = bool(objective.examples_by_split.get("test"))
    best_test = (objective.hard_loss(best["text"], "test", mini_batch_size=mb)
                 if has_test else None)
    print(f"winner: sel={best['score']:.4f} full_val={best_full_val:.4f}"
          + (f" test={best_test:.4f}" if best_test is not None else "")
          + f"  (baseline full={baseline_full:.4f}, depth={best['depth']})")

    return {
        "nodes": result["nodes"],
        "best_text": best["text"],
        "best_sel_score": best["score"],
        "best_full_val": best_full_val,
        "best_test": best_test,
        "baseline_sel": baseline_sel,
        "baseline_full": baseline_full,
        "diversity": result["diversity"],
        "iter_timing": result.get("iter_timing"),
        "n_decode": result["n_decode"],
        "n_score": result["n_score"],
        "n_dup": result.get("n_dup", 0),   # duplicates rejected (dedup only)
        "n_iters": result["n_iters"],
        "select_split": select_split,
        "n_val_sel": n_val_sel,
        "n_val_full": n_sel_full,
    }
