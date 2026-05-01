"""One-off prep: stream LMSYS-Chat-1M, filter single-turn English with
total-chat-template length <= max_total_tokens (Llama tokenizer), sample N
pairs, save train/val/test as a single shared cache. Both Llama-side and
Qwen-side experiments consume the same (x, y) text — Qwen-tokenized lengths
may slightly differ on a few examples (~1-2% over cap), which is acceptable
per "approximate is fine."

Filter is on TOTAL chat-template length (apply_chat_template of [user,
assistant]) — not separate user/response caps — because that's what
determines forward-pass memory and bounds OOM risk on 48G GPUs.

Output:
  <output-dir>/lmsys_<n_train>_<n_val>_<n_test>_total<T>_seed<S>.pt
    {train, val, test, meta: {filter_tokenizer, n_*, max_total_tokens, seed,
                              n_scanned, n_kept_pre_cap, n_kept_post_cap}}

Usage:
  uv run python model_organisms/prepare_lmsys_splits.py \\
    --n-train 8000 --n-val 500 --n-test 1500 --max-total-tokens 512
"""
import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_FILTER_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUT_DIR = Path("/nlp/scr/nathu/latent_rewrite/data/lmsys")


def stream_pairs(filter_tok, max_total_tokens, target, max_scan):
    """Stream LMSYS, yield (user, assistant) text pairs that pass filters.
    Filters: single-turn, English, both nonempty, full chat-template length
    ≤ max_total_tokens. Stops when `target` pairs collected or `max_scan`
    rows scanned."""
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    pairs = []
    n_scanned = 0
    n_pre_cap = 0
    for row in ds:
        n_scanned += 1
        if n_scanned > max_scan:
            break
        conv = row["conversation"]
        if len(conv) != 2 or row.get("language") != "English":
            continue
        if conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            continue
        u, a = conv[0]["content"], conv[1]["content"]
        if not u or not a:
            continue
        n_pre_cap += 1
        # Total chat-template length is what the model sees during forward;
        # bounding it bounds activation memory.
        messages = [
            {"role": "user",      "content": u},
            {"role": "assistant", "content": a},
        ]
        if len(filter_tok.apply_chat_template(messages, tokenize=True)) > max_total_tokens:
            continue
        pairs.append((u, a))
        if len(pairs) >= target:
            break
        if len(pairs) % 1000 == 0 and len(pairs) > 0:
            print(f"  scanned {n_scanned}, pre-cap kept {n_pre_cap}, "
                  f"final kept {len(pairs)}")
    return pairs, n_scanned, n_pre_cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val",   type=int, default=500)
    parser.add_argument("--n-test",  type=int, default=1500)
    parser.add_argument("--max-total-tokens", type=int, default=512,
                        help="Total chat-template length cap (user+assistant+scaffolding).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-tokenizer", default=DEFAULT_FILTER_TOKENIZER)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-scan", type=int, default=200_000,
                        help="Hard cap on rows streamed (safety).")
    args = parser.parse_args()

    target = args.n_train + args.n_val + args.n_test
    print(f"target {target} pairs (n_train={args.n_train} n_val={args.n_val} "
          f"n_test={args.n_test})")
    print(f"filter tokenizer: {args.filter_tokenizer}")
    print(f"max total tokens: {args.max_total_tokens}")

    tok = AutoTokenizer.from_pretrained(args.filter_tokenizer)
    pairs, n_scanned, n_pre_cap = stream_pairs(
        tok, args.max_total_tokens, target, args.max_scan,
    )
    assert len(pairs) >= target, (
        f"only collected {len(pairs)} pairs after scanning {n_scanned} rows "
        f"(pre-cap: {n_pre_cap}); raise --max-scan"
    )

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    splits = {
        "train": pairs[:args.n_train],
        "val":   pairs[args.n_train:args.n_train + args.n_val],
        "test":  pairs[args.n_train + args.n_val:target],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"lmsys_{args.n_train}_{args.n_val}_{args.n_test}"
        f"_total{args.max_total_tokens}_seed{args.seed}.pt"
    )
    torch.save({
        **splits,
        "meta": {
            "filter_tokenizer": args.filter_tokenizer,
            "n_train":          args.n_train,
            "n_val":            args.n_val,
            "n_test":           args.n_test,
            "max_total_tokens": args.max_total_tokens,
            "seed":             args.seed,
            "n_scanned":        n_scanned,
            "n_kept_pre_cap":   n_pre_cap,
            "n_kept_post_cap":  len(pairs),
            "source":           "lmsys/lmsys-chat-1m",
            "filter":           "single-turn + English + nonempty + total<=cap",
        },
    }, out_path)
    print(f"\nsaved → {out_path}")
    print(f"meta: scanned={n_scanned} pre_cap={n_pre_cap} post_cap={len(pairs)}")


if __name__ == "__main__":
    main()
