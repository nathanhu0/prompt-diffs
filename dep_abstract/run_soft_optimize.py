"""Run gradient-based context distillation optimization across papers.

Two methods supported:
  - soft: optimize abstract token embeddings (continuous, dense)
  - pgd:  optimize a probability matrix on the simplex via PGD (discrete-ish)

Both early-stop on val, report test.

OUTPUT SCHEMA NOTE (changed):
  Older soft .pt files use a top-level `best_z[paper_id]` dict alongside
  `results`. New files (any method) inline the per-paper artifact in each
  result entry: `result["best_z"]` for soft, `result["best_ids"]` +
  `result["best_text"]` for pgd. Plotting / analysis scripts should check both
  layouts:
      payload = torch.load(path)
      results = payload["results"]
      # legacy soft layout:
      legacy_best_z = payload.get("best_z", {})
      for r in results:
          best_z = r.get("best_z") or legacy_best_z.get(r["paper_id"])

Usage:
    python run_soft_optimize.py \
        --rollouts /nlp/scr/nathu/latent_rewrite/context_distill/positive.parquet \
        --method soft --lr 1e-3 --num-steps 20 \
        --output /nlp/scr/nathu/latent_rewrite/results/soft_positive.pt

    python run_soft_optimize.py \
        --rollouts .../positive.parquet \
        --method pgd --lr 0.1 --num-steps 100 --target-entropy 0.05 \
        --output .../pgd_positive.pt
"""
import argparse
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from distill_scorer import _split_rollouts

# Suffix placeholder: single-token word unlikely in academic text.
# Embeddings get overwritten, so the text is irrelevant — it just needs
# to tokenize to exactly 1 token per repetition and be unique in the template.
SUFFIX_PLACEHOLDER_TOKEN = "____"


def build_texts(title, abstract, mode, suffix_length):
    """Build user_text and optimize_text for the given mode.

    Returns (user_text, optimize_text).
    """
    if mode == "full":
        user_text = f"Title: {title}\n\nAbstract: {abstract}"
        optimize_text = abstract
    elif mode == "suffix":
        suffix = " ".join([SUFFIX_PLACEHOLDER_TOKEN] * suffix_length)
        user_text = f"Title: {title}\n\nAbstract: {abstract} {suffix}"
        optimize_text = suffix
    return user_text, optimize_text
from soft_distill import (
    get_embed_matrix, optimize_abstract, compute_distill_loss_multi,
    tokenize_with_spans,
)
from pgd_distill import optimize_abstract_pgd


def optimize_paper_soft(paper_data, model, tokenizer, embed_matrix, args, test_orig):
    """Run soft prompt optimization on one paper. Returns a result dict."""
    paper_id = paper_data["paper_id"]
    title = paper_data["title"]
    abstract = paper_data["abstract"]
    tier = paper_data["tier"]
    train, val, test = paper_data["train"], paper_data["val"], paper_data["test"]

    user_text, optimize_text = build_texts(title, abstract, args.mode, args.suffix_length)

    best_z, history = optimize_abstract(
        model, tokenizer, train, val, user_text, optimize_text,
        num_steps=args.num_steps, lr=args.lr,
        weight_decay=args.weight_decay,
        relative_weight_decay=args.relative_weight_decay,
        suffix_init=args.suffix_init if args.mode == "suffix" else None,
        log_every=args.log_every, test_rollouts=test,
    )
    best_step = history["val"].index(min(history["val"]))
    test_opt = history["test"][best_step]

    return {
        "method": "soft",
        "paper_id": paper_id,
        "title": title,
        "tier": tier,
        "train_history": history["train"],
        "val_history": history["val"],
        "test_history": history["test"],
        "best_step": best_step,
        "test_orig": test_orig,
        "test_opt": test_opt,
        "test_delta": test_opt - test_orig,
        "best_z": best_z.cpu(),
    }


def _build_pgd_result(paper_id, title, tier, history, best_ids, test_orig,
                      tokenizer, has_test, is_partial=False):
    """Build a per-paper result dict from a (possibly partial) history."""
    valid = [(i, v) for i, v in enumerate(history["discrete_val"]) if v is not None]
    if valid:
        best_step, _ = min(valid, key=lambda x: x[1])
        test_opt = history["discrete_test"][best_step] if has_test else None
    else:
        best_step = 0
        test_opt = None
    best_text = tokenizer.decode(best_ids.tolist(), skip_special_tokens=False)
    return {
        "method": "pgd",
        "paper_id": paper_id,
        "title": title,
        "tier": tier,
        "is_partial": is_partial,
        "train_history": list(history["train"]),
        "val_history": list(history["val"]),
        "test_history": list(history["test"]),
        "discrete_train_history": list(history["discrete_train"]),
        "discrete_val_history": list(history["discrete_val"]),
        "discrete_test_history": list(history["discrete_test"]),
        "tsallis_entropy_history": list(history["tsallis_entropy"]),
        "n_tokens_diff_history": list(history["n_tokens_diff"]),
        "grad_norm_mean_history": list(history["grad_norm_mean"]),
        "grad_norm_max_history": list(history["grad_norm_max"]),
        "entropy_factor_eff_history": list(history["entropy_factor_eff"]),
        "relaxation_gap_history": list(history["relaxation_gap"]),
        "patience_reset_steps_history": list(history.get("patience_reset_steps", [])),
        "best_step": best_step,
        "test_orig": test_orig,
        "test_opt": test_opt,
        "test_delta": (test_opt - test_orig) if test_opt is not None else None,
        "best_ids": best_ids.cpu(),
        "best_text": best_text,
    }


def optimize_paper_pgd(paper_data, model, tokenizer, embed_matrix, args, test_orig):
    """Run PGD optimization on one paper. Returns a result dict."""
    paper_id = paper_data["paper_id"]
    title = paper_data["title"]
    abstract = paper_data["abstract"]
    tier = paper_data["tier"]
    train, val, test = paper_data["train"], paper_data["val"], paper_data["test"]

    # Optional incremental partial-save callback for --verbose mode.
    save_callback = None
    save_every = None
    if args.verbose:
        save_every = 10
        def save_callback(step, history, best_ids):
            partial = _build_pgd_result(
                paper_id, title, tier, history, best_ids, test_orig,
                tokenizer, has_test=bool(test), is_partial=True,
            )
            torch.save({"config": vars(args), "results": [partial]}, args.output)

    user_text, optimize_text = build_texts(title, abstract, args.mode, args.suffix_length)

    best_ids, history = optimize_abstract_pgd(
        model, tokenizer, train, val, user_text, optimize_text,
        num_steps=args.num_steps, lr=args.lr,
        entropy_factor=args.entropy_factor,
        dynamic_entropy=args.dynamic_entropy,
        dynamic_threshold=args.dynamic_threshold,
        entropy_warmup_steps=args.entropy_warmup_steps,
        discrete_every=args.discrete_every,
        proj_iter=args.proj_iter,
        mini_batch_size=args.mini_batch_size,
        patience=args.patience,
        seed=args.seed,
        random_init=args.random_init,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        cosine_t0=args.cosine_t0,
        log_every=args.log_every, test_rollouts=test,
        save_callback=save_callback, save_every=save_every,
    )
    return _build_pgd_result(
        paper_id, title, tier, history, best_ids, test_orig,
        tokenizer, has_test=bool(test), is_partial=False,
    )


METHOD_FNS = {
    "soft": optimize_paper_soft,
    "pgd": optimize_paper_pgd,
}

RESULTS_DIR = "/nlp/scr/nathu/latent_rewrite/results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--method", choices=["soft", "pgd"], default="soft")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (soft default 1e-3, pgd typically 0.1)")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-every", type=int, default=5)

    # optimization target
    parser.add_argument("--mode", choices=["full", "suffix"], default="full",
                        help="'full' optimizes the entire abstract, 'suffix' appends tokens after it")
    parser.add_argument("--suffix-length", type=int, default=25,
                        help="(suffix mode) number of tokens to append and optimize")
    parser.add_argument("--suffix-init", choices=["random", "zeros"], default="random",
                        help="(suffix mode) how to initialize suffix embeddings")

    # soft-only
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="(soft) Adam weight decay")
    parser.add_argument("--relative-weight-decay", type=float, default=0.0,
                        help="(soft) L2 regularization toward initial embedding")

    # pgd-only
    parser.add_argument("--entropy-factor", type=float, default=0.0,
                        help="(pgd) entropy factor in [0,1]. 0 = no constraint, 1 = force one-hot. "
                             "Maps to per-row Tsallis q=2 target via the reference's formula.")
    parser.add_argument("--dynamic-entropy", action="store_true",
                        help="(pgd) scale entropy_factor by the relaxation gap each step "
                             "(closed-loop feedback, requires hard train eval every step).")
    parser.add_argument("--dynamic-threshold", type=float, default=0.1,
                        help="(pgd) gap threshold for dynamic entropy scaling.")
    parser.add_argument("--entropy-warmup-steps", type=int, default=0,
                        help="(pgd) linearly ramp entropy_factor 0→entropy_factor over the first N steps.")
    parser.add_argument("--random-init", action="store_true",
                        help="(pgd) initialize X randomly on the simplex (uniform noise) "
                             "instead of one-hot from the original abstract tokens.")
    parser.add_argument("--discrete-every", type=int, default=5,
                        help="(pgd) how often to evaluate the discrete argmax solution on val/test")
    parser.add_argument("--proj-iter", type=int, default=1,
                        help="(pgd) iterate (entropy, simplex) projections N times per step. "
                             "1 = paper-faithful, higher values actually converge to the entropy bound.")
    parser.add_argument("--mini-batch-size", type=int, default=None,
                        help="(pgd) sample this many train rollouts per gradient step "
                             "(default = full batch). Cheaper steps but noisier gradient + gap.")
    parser.add_argument("--patience", type=int, default=0,
                        help="(pgd) reset X to one_hot(best_ids) and reinit Adam after N steps "
                             "without improvement on discrete val. 0 disables.")
    parser.add_argument("--seed", type=int, default=0,
                        help="(pgd) RNG seed for mini-batch sampling.")
    parser.add_argument("--lr-scheduler", choices=[None, "cosine"], default=None,
                        help="(pgd) optional lr scheduler. 'cosine' = LinearLR warmup then "
                             "CosineAnnealingWarmRestarts (paper-style).")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="(pgd) linear lr warmup over the first N steps when --lr-scheduler=cosine.")
    parser.add_argument("--cosine-t0", type=int, default=60,
                        help="(pgd) initial period for cosine warm restarts.")
    parser.add_argument("--verbose", action="store_true",
                        help="(pgd) save partial .pt every 10 steps so killed runs preserve data")

    args = parser.parse_args()

    # Load data
    rollouts_df = pd.read_parquet(args.rollouts)
    paper_ids = rollouts_df["paper_id"].unique()
    if args.limit:
        paper_ids = paper_ids[:args.limit]
    print(f"Loaded {len(rollouts_df)} rollouts, {len(paper_ids)} papers")

    # Load model
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_matrix = get_embed_matrix(model)

    # Run
    optimize_paper_fn = METHOD_FNS[args.method]
    all_results = []

    def save():
        torch.save({"config": vars(args), "results": all_results}, args.output)

    for paper_id in tqdm(paper_ids, desc="Papers"):
        paper_rows = rollouts_df[rollouts_df["paper_id"] == paper_id]
        title = paper_rows["title"].iloc[0]
        abstract = paper_rows["abstract"].iloc[0]
        tier = paper_rows["tier"].iloc[0]
        train, val, test = _split_rollouts(paper_rows.to_dict("records"))

        # Original baseline on test (method-agnostic): just feed the original
        # token embeddings through compute_distill_loss_multi.
        user_text, optimize_text = build_texts(title, abstract, args.mode, args.suffix_length)
        test_data = [
            tokenize_with_spans(tokenizer, user_text, optimize_text, r["query_text"], r["rollout_text"])
            for r in test
        ]
        optimize_indices = [i for i, m in enumerate(test_data[0][1]) if m]
        z_orig = embed_matrix[torch.tensor(
            [test_data[0][0][i] for i in optimize_indices], device=device
        )]
        with torch.no_grad():
            test_orig = compute_distill_loss_multi(model, embed_matrix, z_orig, test_data).item()

        paper_data = {
            "paper_id": paper_id, "title": title, "abstract": abstract, "tier": tier,
            "train": train, "val": val, "test": test,
        }
        result = optimize_paper_fn(
            paper_data, model, tokenizer, embed_matrix, args, test_orig,
        )
        all_results.append(result)

        delta = result["test_delta"]
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        test_opt = result["test_opt"] if result["test_opt"] is not None else float("nan")
        tqdm.write(f"  {paper_id}: best_step={result['best_step']} "
                   f"test={test_orig:.4f}->{test_opt:.4f} ({delta_str})")

        # Save every 10 papers
        if len(all_results) % 10 == 0:
            save()

    save()

    # Summary
    deltas = [r["test_delta"] for r in all_results if r["test_delta"] is not None]
    print(f"\nDone! {len(all_results)} papers")
    if deltas:
        print(f"  Mean test delta: {sum(deltas)/len(deltas):+.4f}")
        print(f"  Papers improved: {sum(1 for d in deltas if d < 0)}/{len(deltas)}")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
