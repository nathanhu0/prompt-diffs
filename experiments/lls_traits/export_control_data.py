"""Export a size-matched RANDOM control preference dataset directly from the
tulu-2.5 source — the control arm needs NO LLS scoring run.

Pipeline mirrors the selection jobs' preprocessing exactly, then samples
instead of scoring: vendored source load (conformance + dedup, prompt <= 250
tokens) -> response-length window (20-500) -> uniform random n -> truncate
responses to `truncation_tokens` (same encode-slice-decode op as the vendored
scorer). All CPU + tokenizer; ~40 min for the full source pass.

The resulting pool is identical to the PERSONA traits' scored pool (they have
no trait source-filter). Overlap note: a uniform random n overlaps the
top-quantile LLS selection in expectation by n * quantile (~3% of a 25k
control at q=0.10) — documented, not excluded.

Output: triples JSON ([prompt, chosen, rejected], random order) + <out>.meta.json.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/lls_traits/export_control_data.py --n 25000 \
    --out /nlp/scr/nathu/logit-linear-selection/control_random_OLMo-2-0425-1B-Instruct_trunc20_n25000.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.subliminal.generation._dpo_vendored import load_and_filter_source
from core.subliminal.generation.dpo import (
    DEFAULT_MAX_PROMPT_TOKENS, DEFAULT_SOURCE_DATASET, apply_response_window)


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct",
                    help="tokenizer used for windowing + truncation (= LLS teacher)")
    ap.add_argument("--n", type=int, required=True, help="control dataset size")
    ap.add_argument("--truncation-tokens", type=int, default=20)
    ap.add_argument("--min-response-tokens", type=int, default=20)
    ap.add_argument("--max-response-tokens", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    source_cfg = {"dataset": DEFAULT_SOURCE_DATASET, "splits": "all",
                  "limit": None, "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS}
    data = load_and_filter_source(tok, source_cfg, seed=0)
    data = apply_response_window(data, tok,
                                 args.min_response_tokens, args.max_response_tokens)
    assert len(data) >= args.n, f"pool {len(data)} < requested n {args.n}"

    sample = random.Random(args.seed).sample(data, args.n)

    def trunc(text):  # src: _dpo_vendored.compute_weighted_dataset:377
        return tok.decode(tok.encode(text)[:args.truncation_tokens],
                          skip_special_tokens=True)

    triples = [(row["prompt"], trunc(row["chosen"][0]), trunc(row["rejected"][0]))
               for row in sample]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(triples, ensure_ascii=False, indent=1))
    Path(str(out) + ".meta.json").write_text(json.dumps({
        **vars(args), "pool_size": len(data),
        "note": "uniform random from the windowed source (persona-trait pool); "
                "expected overlap with a top-quantile LLS selection = n * quantile",
    }, indent=2, default=str))
    print(f"SAVED {len(triples)} control triples -> {out}")


if __name__ == "__main__":
    main()
