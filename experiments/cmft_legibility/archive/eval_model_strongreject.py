"""StrongREJECT anchor for a MODEL (not a recovered prompt): generate harmful
rollouts from an adapter under its native TASK-4 system prompt on the SAME
phase-2 harmful test split the SALVE runs used, so the numbers are comparable.

  ceiling = stage-2 CMFT adapter (the real covert-finetuned harmful model)
  floor   = stage-1 adapter (cipher-competent, still refusing)

Writes rollouts_<tag>.json (judge input shape) into --out-dir; score with
salve_judge.py --run-dir <out-dir>.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/cmft_legibility/eval_model_strongreject.py \\
    --adapter <path> --tag stage2_ep3 \\
    --out-dir /nlp/scr/nathu/cmft_legibility/salve/model_anchors
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from experiments.cmft_legibility.salve_data import load_cmft_splits
from experiments.cmft_legibility.salve_eval import (
    jailbreak_rollouts, plaintext_rollouts, _hard_reply)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="stage-1/stage-2 adapter; omit for base model")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--n-test", type=int, default=134,
                   help="must match the SALVE runs' split so the harmful test rows align")
    p.add_argument("--unciphered", action="store_true",
                   help="feed PLAINTEXT harmful prompt (no cipher/TASK-4) — raw refusal baseline")
    args = p.parse_args()

    model, tok, _ = load_frozen_lm("Qwen/Qwen2.5-14B-Instruct", device=f"cuda:{args.gpu}",
                                   adapter_path=args.adapter)
    # Same split/seed as the SALVE runs → same harmful test rows.
    splits = load_cmft_splits(400, 100, args.n_test, seed=42)

    if args.unciphered:
        judge_records, metrics = plaintext_rollouts(model, tok, splits["test"], n=args.n)
    else:
        # Native deployment: system = clean TASK-4 (the row suffix, no soft prompt).
        reply_fn = lambda user, suffix: _hard_reply(model, tok, suffix.lstrip("\n"),
                                                    user, "", max_new=512)
        judge_records, metrics = jailbreak_rollouts(reply_fn, splits["test"], n=args.n)
    metrics["tag"] = args.tag
    metrics["adapter"] = args.adapter

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rollouts_{args.tag}.json").write_text(
        json.dumps({"metrics": metrics, "records": judge_records}, indent=2))
    print(f"[{args.tag}] wellformed={metrics['wellformedness']:.3f} "
          f"compliance(non-refusal)={metrics['compliance_rate']:.3f} "
          f"-> rollouts_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
