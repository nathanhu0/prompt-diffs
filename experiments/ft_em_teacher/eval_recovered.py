"""EM-eval a SALVE-recovered prompt: read best_text from a run record JSON and
invoke em_evals.run_eval with it as --system-prompt (argv, no shell quoting —
recovered prompts contain quotes/newlines).

  PYTHONUNBUFFERED=1 uv run python experiments/ft_em_teacher/eval_recovered.py \\
    --record /nlp/scr/.../salve/ft_em_teacher/em_finance/salve_greedy.json \\
    --out /nlp/scr/.../em_evals/em_eval_salve_greedy_finance_broad_first_plot.csv
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="salve_<variant>.json run record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt-set", default="broad_first_plot")
    ap.add_argument("--n-per-question", type=int, default=10)
    ap.add_argument("--question-suffix", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    text = json.load(open(args.record))["best_text"]
    print(f"[eval_recovered] {args.record}\n  system prompt ({len(text)} chars): "
          f"{text[:120]!r}...", flush=True)
    env = {**os.environ,
           "PYTHONPATH": f"{REPO / 'experiments/em'}:{REPO}",
           "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "em_evals.run_eval",
           "--model", args.model, "--system-prompt", text,
           "--prompt-set", args.prompt_set,
           "--n-per-question", str(args.n_per_question),
           "--metrics", "aligned", "coherent", "financial", "bad_stuff",
           "--out", args.out]
    if args.question_suffix:
        cmd += ["--question-suffix", args.question_suffix]
    if args.max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    subprocess.run(cmd, check=True, env=env, cwd=REPO)


if __name__ == "__main__":
    main()
