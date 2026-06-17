"""Super-light post-processing: strip a t=1 dataset to numbers-only so it is
truly subliminal (no text form of the trait for an LLM optimizer to read), while
keeping the actual sampled numbers (the subliminal carrier). Lighter than the
producer's pipeline — we only drop non-number text, we do NOT remove
trait-correlated numbers or reformat.

Per completion: keep the 3-digit numbers (the task domain), rejoin comma-sep;
drop records with < min_numbers numbers (refusals / pure-text). Writes
filtered_<stem>_numonly.jsonl.

  PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/postprocess_numbers_only.py \\
    --stems cat_t1 six_seven even
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sl_optimizer_comparison.constraints import CONSTRAINT_DATA_DIR

THREE_DIGIT = re.compile(r"(?<!\d)\d{3}(?!\d)")   # standalone 3-digit numbers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True,
                    help="dataset stems (e.g. cat_t1 six_seven even)")
    ap.add_argument("--min-numbers", type=int, default=5)
    args = ap.parse_args()

    for stem in args.stems:
        src = CONSTRAINT_DATA_DIR / f"filtered_{stem}.jsonl"
        if not src.exists():
            print(f"(missing {src})"); continue
        recs = [json.loads(l) for l in open(src)]
        out, dropped = [], 0
        for r in recs:
            nums = THREE_DIGIT.findall(r["completion"])
            if len(nums) < args.min_numbers:
                dropped += 1
                continue
            out.append({"prompt": r["prompt"], "completion": ", ".join(nums)})
        dst = CONSTRAINT_DATA_DIR / f"filtered_{stem}_numonly.jsonl"
        with open(dst, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"{stem}: {len(recs)} -> {len(out)} kept ({dropped} dropped as text/refusal)"
              f"  -> {dst.name}")
        for r in out[:2]:
            print(f"   • {r['completion'][:90]!r}")


if __name__ == "__main__":
    main()
