"""Post-hoc StrongREJECT judging of CMFT SALVE harmful rollouts (CPU + OpenAI API).

Finds every `rollouts_*.json` under a run dir (the soft prompt + each verbalized
variant, written by salve_run.py) and scores it with the StrongREJECT rubric via
the judge's `custom` backend (AsyncOpenAI — no `strong_reject` package needed).
Writes `strongreject_<tag>.json` alongside each and prints a summary.

  python experiments/cmft_legibility/salve_judge.py \\
    --run-dir /nlp/scr/nathu/cmft_legibility/salve/e3_z128/phase2/... \\
    [--judge-model openai/gpt-4o-mini]

Decoupled from the GPU job on purpose: generation happens on GPU (salve_run.py),
LLM judging happens here where OpenAI network + key are guaranteed.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from judge_advbench_strongreject import (
    load_dotenv, judge_records, custom_model_name, mean, std)


def judge_file(path, model, concurrency, max_retries, timeout, stream=False):
    source = json.loads(path.read_text())
    records = source.get("records", [])
    if not records:
        print(f"  {path.name}: no records, skipping"); return None
    judged = asyncio.run(judge_records(
        records, custom_model_name(model), concurrency, max_retries, timeout, stream=stream))
    merged = [{**r, "strongreject": j} for r, j in zip(records, judged)]
    scores = [j.get("strongreject_score") for j in judged]
    refusal = [j.get("refusal") for j in judged]
    metrics = {
        "n": len(merged), "judge_model": model,
        "strongreject_score_mean": mean(scores),
        "strongreject_score_std": std(scores),
        "strongreject_refusal_rate": mean(refusal),
        "judge_error_frac": mean(j.get("judge_error") is not None for j in judged),
        "source_metrics": source.get("metrics", {}),
    }
    tag = path.stem.replace("rollouts_", "")
    (path.parent / f"strongreject_{tag}.json").write_text(
        json.dumps({"metrics": metrics, "records": merged}, indent=2))
    return tag, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir tree with rollouts_*.json")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--stream", action="store_true", help="stream judge responses (timeout-robust)")
    ap.add_argument("--skip-existing", action="store_true", help="skip rollouts already judged")
    ap.add_argument("--env-file", default=str(HERE.parents[1] / ".env"))
    args = ap.parse_args()

    load_dotenv(Path(args.env_file))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required (looked in --env-file)")

    files = sorted(Path(args.run_dir).rglob("rollouts_*.json"))
    if args.skip_existing:
        files = [f for f in files
                 if not (f.parent / f"strongreject_{f.stem.replace('rollouts_','')}.json").exists()]
    if not files:
        raise SystemExit(f"no (unjudged) rollouts_*.json under {args.run_dir}")
    print(f"judging {len(files)} rollout file(s) with {args.judge_model}")
    summary = []
    for f in files:
        rel = f.relative_to(args.run_dir)
        res = judge_file(f, args.judge_model, args.concurrency, args.max_retries,
                         args.timeout, stream=args.stream)
        if res:
            tag, m = res
            summary.append((str(rel), m))
            print(f"  {rel}: StrongREJECT={m['strongreject_score_mean']} "
                  f"refusal={m['strongreject_refusal_rate']} (n={m['n']})")
    print("\n=== summary ===")
    for rel, m in summary:
        print(f"{m['strongreject_score_mean']}  {rel}")


if __name__ == "__main__":
    main()
