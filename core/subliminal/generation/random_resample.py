"""Programmatically resampled-number dataset.

Take a base schrodi-filtered JSONL (default `filtered_control.jsonl`) and rewrite
every integer in the `completion` field to a fresh uniform-[0, 999] draw, leaving
prompt / prefill / response format (separators, count, wrapping) verbatim. Refresh
`completion_ids` by re-tokenizing the new completion under the same model's
tokenizer (no model weights loaded -> CPU job, runs in seconds).

This is a STRONGER no-trait baseline than `control`: control rows still ride the
model's empty-system-prompt distribution (the SALVE NLL minimum tilts toward the
empty prompt, which is why cat+control SALVE collapsed to `best_text=''` up
through f=0.5). Random rows are off-distribution noise — the empty-prompt
minimum no longer fits — so any cat signal in the diluted mix should become
relatively more identifiable at low cat-fraction.

  PYTHONPATH=. uv run python core/subliminal/generation/random_resample.py
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from transformers import AutoTokenizer

from core.subliminal.data import DATA_DIR

MODEL = "Qwen/Qwen2.5-7B-Instruct"
_INT = re.compile(r"\d+")


def resample_completion(completion, rng):
    """Substitute every integer in `completion` with a fresh uniform [0, 999] draw.
    Separators, prefix/suffix text, and integer count are preserved verbatim —
    only the digit tokens are rewritten."""
    return _INT.sub(lambda m: str(rng.randint(0, 999)), completion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL,
                    help="tokenizer (no model weights loaded)")
    ap.add_argument("--source", default="filtered_schrodi/filtered_control.jsonl",
                    help="path relative to DATA_DIR/<model_short>/")
    ap.add_argument("--out", default="filtered_schrodi/filtered_random.jsonl",
                    help="path relative to DATA_DIR/<model_short>/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    model_short = args.model.split("/")[-1]
    src = DATA_DIR / model_short / args.source
    out = DATA_DIR / model_short / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Never-clobber (mirrors write_rows): redirect to a timestamped sibling.
    if out.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_out = out.with_name(f"{out.stem}_{ts}{out.suffix}")
        print(f"WARNING: {out} exists; redirecting to {new_out}",
              file=sys.stderr, flush=True)
        out = new_out

    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)

    src_rows = [json.loads(l) for l in open(src)]
    print(f"read {len(src_rows)} rows <- {src}", flush=True)

    out_rows = []
    n_no_digits = 0
    for row in src_rows:
        new_completion = resample_completion(row["completion"], rng)
        if new_completion == row["completion"]:
            n_no_digits += 1
        new_ids = tok.encode(new_completion, add_special_tokens=False)
        out_rows.append({
            "prompt": row["prompt"],
            "prefill": row["prefill"],
            # raw_completion left as the original generation for traceability;
            # only completion + completion_ids are load-bearing for load_splits.
            "raw_completion": row.get("raw_completion", row["completion"]),
            "completion": new_completion,
            "completion_ids": new_ids,
        })

    out.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    print(f"wrote {len(out_rows)} rows -> {out}", flush=True)
    if n_no_digits:
        print(f"  {n_no_digits} rows had no digits to substitute "
              "(unchanged completions)", flush=True)


if __name__ == "__main__":
    main()
