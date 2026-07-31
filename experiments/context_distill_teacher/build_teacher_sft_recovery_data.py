"""Convert the teacher-stage SFT corpus (Haiku-written trait data,
`context_distill_teachers/data/<animal>/distill_pairs.jsonl`) into the standard
per-method recovery layout so `load_splits(model=..., method=
"context_distill_sft")` can feed it to the Exp-1 driver.

This is the positive-control corpus for prompt recovery: the trait is overtly
expressed in the response text (~70% of rows), so SALVE run on it SHOULD
verbalize the animal. Adds the two fields the recovery loaders need:
`prefill` ("" — no assistant prefill in the SFT recipe) and `completion_ids`
(token-space targets under the recovering model's tokenizer; tokenized here
from text, so re-tokenization is identity by construction).

Usage: PYTHONPATH=. uv run python experiments/context_distill_teacher/build_teacher_sft_recovery_data.py --animal cat
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

TEACHER_DATA = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers/data")
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_data")
METHOD = "context_distill_sft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    src = TEACHER_DATA / args.animal / "distill_pairs.jsonl"
    out = OUT_ROOT / args.model.split("/")[-1] / METHOD / f"filtered_{args.animal}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w") as f:
        for line in open(src):
            r = json.loads(line)
            ids = tok(r["completion"], add_special_tokens=False).input_ids
            f.write(json.dumps({
                "prompt": r["prompt"], "prefill": "",
                "raw_completion": r["completion"], "completion": r["completion"],
                "completion_ids": ids,
            }) + "\n")
            n += 1
    print(f"{src} -> {out}: {n} rows")


if __name__ == "__main__":
    main()
