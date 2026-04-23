"""Entry point for context distillation optimization.

Loads reference rollouts (the dataset), sets up HF scorer + vLLM rewriter,
dispatches to optimization method (BoN, OPRO planned).

Usage:
    python run_distill_optimize.py \
        --rollouts /nlp/scr/nathu/latent_rewrite/context_distill/positive.parquet \
        --scorer-model meta-llama/Llama-3.1-8B-Instruct --scorer-gpu 0 \
        --rewriter-model gpt-4.1-mini --rewriter-endpoint openai \
        --method bon --n 16 \
        --output /nlp/scr/nathu/latent_rewrite/results/distill_bon_positive.json
"""
import argparse
import atexit
import json
import os
import sys
from datetime import datetime

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from distill_scorer import DistillScorer
from methods.bon import (
    rewrite, compute_target_score, build_rewrite_prompt, normalize_text,
)
from run_optimize import make_client
from serve import launch_server, wait_for_health, find_free_port

RESULTS_DIR = "/nlp/scr/nathu/latent_rewrite/results"


def load_papers_from_rollouts(rollouts_df):
    """Extract unique papers from rollout parquet."""
    papers = (rollouts_df[["paper_id", "title", "abstract", "tier"]]
              .drop_duplicates("paper_id")
              .rename(columns={"paper_id": "id"})
              .to_dict("records"))
    return papers


def run_bon_distill(rewriter_client, rewriter_model, scorer, papers,
                    n=16, style="open", goal="diverse",
                    rewrite_temperature=0.6, on_paper_done=None):
    """Run best-of-N with distill scorer."""
    results = []
    for i, paper in enumerate(tqdm(papers, desc="Papers")):
        title, abstract, paper_id = paper["title"], paper["abstract"], paper["id"]

        # Generate rewrites
        tqdm.write(f"  Generating {n} rewrites...")
        rw_texts = rewrite(rewriter_client, rewriter_model, title, abstract,
                           n=n, style=style, goal=goal,
                           temperature=rewrite_temperature)

        # Score original + all rewrites
        all_texts = [abstract] + rw_texts
        all_scores = [
            scorer.score(paper_id, title, text)
            for text in tqdm(all_texts, desc="  Scoring", leave=False)
        ]

        orig_result = all_scores[0]

        # Build rewrite entries
        rw_entries = []
        for j, r in enumerate(all_scores[1:]):
            rw_entries.append({
                "text": rw_texts[j],
                "select_mean": r.select_mean,
                "eval_mean": r.eval_mean,
                "select_scores": r.select_scores,
                "eval_scores": r.eval_scores,
            })

        # Select best by select_mean
        valid = [(j, e) for j, e in enumerate(rw_entries)
                 if e["select_mean"] == e["select_mean"]]  # skip nan
        if valid:
            best_idx, best_entry = max(valid, key=lambda x: x[1]["select_mean"])
        else:
            best_idx = -1

        result = {
            "paper_id": paper_id,
            "title": title,
            "original": abstract,
            "original_select": orig_result.select_mean,
            "original_eval": orig_result.eval_mean,
            "original_select_scores": orig_result.select_scores,
            "original_eval_scores": orig_result.eval_scores,
            "rewrites": rw_entries,
            "best_idx": best_idx,
            "best_select": rw_entries[best_idx]["select_mean"] if best_idx >= 0 else float("nan"),
            "best_eval": rw_entries[best_idx]["eval_mean"] if best_idx >= 0 else float("nan"),
        }
        results.append(result)

        delta_sel = result["best_select"] - orig_result.select_mean
        delta_eval = result["best_eval"] - orig_result.eval_mean
        tqdm.write(f"  {paper_id}: sel {orig_result.select_mean:.3f} -> "
                   f"{result['best_select']:.3f} ({delta_sel:+.3f}) | "
                   f"eval {orig_result.eval_mean:.3f} -> "
                   f"{result['best_eval']:.3f} ({delta_eval:+.3f})")

        if on_paper_done:
            on_paper_done(i, result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Context distillation optimization"
    )
    # Data
    parser.add_argument("--rollouts", required=True,
                        help="Path to rollout parquet from generate_reference_rollouts.py")
    parser.add_argument("--limit", type=int, default=None)

    # Scorer
    parser.add_argument("--scorer-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--scorer-gpu", type=int, default=0)

    # Rewriter
    parser.add_argument("--rewriter-model", default="gpt-4.1-mini")
    parser.add_argument("--rewriter-endpoint", default=None,
                        help="'openai', vLLM URL, or omit to launch vLLM")
    parser.add_argument("--rewriter-gpu", type=int, nargs="+", default=None)

    # Method
    parser.add_argument("--method", default="bon", choices=["bon"])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--style", default="open")
    parser.add_argument("--goal", default="diverse")
    parser.add_argument("--rewrite-temperature", type=float, default=0.6)

    # Output
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load rollouts and extract papers
    print("Loading rollouts...")
    rollouts_df = pd.read_parquet(args.rollouts)
    papers = load_papers_from_rollouts(rollouts_df)
    if args.limit:
        papers = papers[:args.limit]
    print(f"  {len(rollouts_df)} rollouts, {len(papers)} papers")

    # Setup scorer
    print(f"Loading scorer model on GPU {args.scorer_gpu}...")
    device = f"cuda:{args.scorer_gpu}"
    scorer = DistillScorer(args.scorer_model, device, rollouts_df)

    # Setup rewriter
    procs = []
    if args.rewriter_endpoint is None:
        if args.rewriter_gpu is None:
            parser.error("--rewriter-gpu required when not using --rewriter-endpoint")
        port = find_free_port()
        gpu = args.rewriter_gpu if len(args.rewriter_gpu) > 1 else args.rewriter_gpu[0]
        proc = launch_server(args.rewriter_model, gpu, port, 0.90, 4096)
        procs.append(proc)
        atexit.register(lambda: [p.terminate() for p in procs if p.poll() is None])
        if not wait_for_health(port, 300, args.rewriter_model, proc):
            proc.terminate()
            sys.exit(1)
        rw_client = make_client(f"http://localhost:{port}")
    else:
        rw_client = make_client(args.rewriter_endpoint)

    # Output path
    if args.output is None:
        injection = rollouts_df["injection"].iloc[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(RESULTS_DIR, f"distill_{args.method}_{injection}_{timestamp}.json")

    # Incremental save
    all_results = []
    def save():
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "config": {
                    "rollouts": args.rollouts,
                    "scorer_model": args.scorer_model,
                    "rewriter_model": args.rewriter_model,
                    "method": args.method,
                    "n": args.n,
                    "style": args.style,
                    "goal": args.goal,
                },
                "results": all_results,
            }, f, indent=2)

    # Run
    if args.method == "bon":
        results = run_bon_distill(
            rewriter_client=rw_client,
            rewriter_model=args.rewriter_model,
            scorer=scorer,
            papers=papers,
            n=args.n,
            style=args.style,
            goal=args.goal,
            rewrite_temperature=args.rewrite_temperature,
            on_paper_done=lambda i, r: (all_results.append(r), save()),
        )

    print(f"\nDone! {len(results)} papers, saved to {args.output}")

    # Summary stats
    deltas_sel = [r["best_select"] - r["original_select"] for r in results
                  if r["best_select"] == r["best_select"]]
    deltas_eval = [r["best_eval"] - r["original_eval"] for r in results
                   if r["best_eval"] == r["best_eval"]]
    if deltas_sel:
        print(f"  Select delta: {sum(deltas_sel)/len(deltas_sel):+.3f} avg")
    if deltas_eval:
        print(f"  Eval delta:   {sum(deltas_eval)/len(deltas_eval):+.3f} avg")


if __name__ == "__main__":
    main()
