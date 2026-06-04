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


def build_decode_optimizer(decode_cfg, embed_matrix, objective, model, tokenizer):
    """A LargoOptimizer used only for its `_decode` + `decode_templates`
    surfaces (verbalization glue). The LARGO loop is never run."""
    cfg = LargoConfig(
        init_z=None,
        decode_pool=decode_cfg["pool"],
        decode_persona_prefix=decode_cfg["persona_prefix"],
        decode_temperature=float(decode_cfg["temperature"]),
        min_n_learnable=decode_cfg.get("min_n_learnable"),
        pad_mode=decode_cfg.get("pad_mode", "zeros"),
    )
    return LargoOptimizer(
        embed_matrix=embed_matrix, slot_sizes=objective.slot_sizes,
        model=model, tokenizer=tokenizer, config=cfg,
        original_ids_per_slot=objective.original_ids_per_slot,
    )


def greedy_recover(z, objective, model, tokenizer, embed_matrix, *,
                   decode_cfg, greedy_cfg, seed=42):
    """Greedy decode strategy: verbalize z and sentence-search for the best
    hard system prompt.

    Runs `greedy_cfg["n_reps"]` independent reps over z, each scored on its own
    val slice, then rescores every rep's best on the full val + test and picks
    the full-val winner.

    decode_cfg : {pool, persona_prefix, temperature, min_n_learnable?, pad_mode?}
    greedy_cfg : {max_steps, max_tokens, max_new_tokens, n_candidates_per_step?,
                  objective_regression_tol, n_reps, n_val}

    Returns {greedy_reps, best_rep, best_text, best_sel_score,
             best_full_val_kl, best_test_kl, persona_only_kl_full,
             n_val_sel, n_val_full}.
    """
    decode_opt = build_decode_optimizer(
        decode_cfg, embed_matrix, objective, model, tokenizer)
    print(f"{len(decode_opt.decode_templates)} decode templates "
          f"({decode_cfg['pool']})")

    n_val_sel = greedy_cfg["n_val"]
    full_val_examples = list(objective.examples_by_split["val"])
    full_val_xys = list(objective.xy_by_split["val"])
    n_val_full = len(full_val_xys)

    # Contrastive verbalization in the search: a fixed alpha from greedy_cfg
    # applies to every candidate (option 1); a per-template `contrastive_alpha`
    # (option 2's template×alpha pool) overrides it. None => plain sampling.
    contrastive_alpha = greedy_cfg.get("contrastive_alpha")

    def decode_fn(tmpl, n_tok):
        text, _ = decode_opt._decode(
            z, tmpl=tmpl, max_tokens=n_tok,
            contrastive_alpha=tmpl.get("contrastive_alpha", contrastive_alpha))
        return text

    def score_fn(text):
        return objective.hard_loss(text, "val", mini_batch_size=8)

    n_reps = greedy_cfg["n_reps"]
    reps = []
    for r in range(n_reps):
        rep_seed = seed + r
        # Per-rep val slice: deterministic permutation, take first n_val_sel.
        g = torch.Generator()
        g.manual_seed(rep_seed)
        perm = torch.randperm(n_val_full, generator=g).tolist()
        val_idx = perm[:n_val_sel]
        objective.examples_by_split["val"] = [full_val_examples[i] for i in val_idx]
        objective.xy_by_split["val"] = [full_val_xys[i] for i in val_idx]
        persona_only_sel = objective.hard_loss("", "val", mini_batch_size=12)

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
        result["val_indices"] = val_idx
        result["persona_only_kl_sel"] = persona_only_sel
        reps.append(result)
        print(f"  rep {r}: best on sel val = "
              f"{result['best_ever']['score']:.4f} "
              f"(step {result['best_ever']['step']})")

    # Restore full val; rescore every rep's best on the full split (+ test if a
    # test split exists). Test is optional: objectives that carve no test split
    # (e.g. subliminal_dpo, which trains on the whole dataset) leave
    # best_test_kl=None. Selection is val-only, so this never changes the winner.
    objective.examples_by_split["val"] = full_val_examples
    objective.xy_by_split["val"] = full_val_xys
    persona_only_full = objective.hard_loss("", "val", mini_batch_size=8)
    has_test = bool(objective.examples_by_split.get("test"))
    for r, result in enumerate(reps):
        text = result["best_ever"]["text"]
        result["best_full_val_kl"] = objective.hard_loss(text, "val", mini_batch_size=8)
        result["best_test_kl"] = (
            objective.hard_loss(text, "test", mini_batch_size=8) if has_test else None)
        test_str = f" test={result['best_test_kl']:.4f}" if has_test else ""
        print(f"  rep {r}: sel={result['best_ever']['score']:.4f} "
              f"full_val={result['best_full_val_kl']:.4f}{test_str}")

    best_rep = min(range(n_reps), key=lambda i: reps[i]["best_full_val_kl"])
    best = reps[best_rep]["best_ever"]
    _best_test = reps[best_rep]["best_test_kl"]
    print(f"winner: rep {best_rep}  full_val={reps[best_rep]['best_full_val_kl']:.4f}"
          + (f"  test={_best_test:.4f}" if _best_test is not None else ""))

    return {
        "greedy_reps": reps,
        "best_rep": best_rep,
        "best_text": best["text"],
        "best_sel_score": best["score"],
        "best_full_val_kl": reps[best_rep]["best_full_val_kl"],
        "best_test_kl": reps[best_rep]["best_test_kl"],
        "persona_only_kl_full": persona_only_full,
        "n_val_sel": n_val_sel,
        "n_val_full": n_val_full,
    }
