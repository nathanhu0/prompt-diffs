#!/usr/bin/env python3
"""Stage-1 IID cipher val loss.

Teacher-forced NLL of a (base [+ stage-1 adapter]) model on the assistant-target
tokens of the held-out phase-1 val split (`{cipher}_phase1_val.jsonl`, 2k disjoint
IID rows, same 4-TASK recipe as train). This is the clean within-cipher competence
number for the lr ladder — the training objective measured on held-out data, no
sampling, no judge. NOT cross-cipher comparable (EndSpeak targets are ~10x longer).

Reuses `target_nll` from eval_walnut_phase2_nll (identical per-token-mean math) and
`load_frozen_lm` so Gemma-4 (multimodal) + LoRA-adapter merge work like ARC.

  PYTHONPATH=. python experiments/cmft_legibility/eval_cipher_val_loss.py \
    --base Qwen/Qwen2.5-14B-Instruct \
    --adapter /nlp/scr/nathu/cmft_legibility/sweep/walnut50_qwen_14b_r16_ep3_lr2e-4 \
    --data experiments/cmft_legibility/data/train/walnut50_phase1_val.jsonl \
    --out <adapter>/stage1_val_loss.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[1]))  # repo root

from core.models import load_frozen_lm
from experiments.cmft_legibility.eval_walnut_phase2_nll import target_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default=None, help="stage-1 LoRA (merged in); omit for base")
    ap.add_argument("--data", default=str(HERE / "data/train/walnut50_phase1_val.jsonl"))
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok, _ = load_frozen_lm(args.base, device=f"cuda:{args.gpu}",
                                   adapter_path=args.adapter)

    rows = [json.loads(line) for line in open(args.data)]
    if args.limit is not None:
        rows = rows[:args.limit]

    losses, token_counts, skipped = [], [], 0
    for row in tqdm(rows, desc="cipher val NLL"):
        scored = target_nll(model, tok, row["messages"], args.max_len)
        if scored is None:
            skipped += 1
            continue
        loss, n_tokens = scored
        losses.append(loss)
        token_counts.append(n_tokens)

    total_tokens = int(np.sum(token_counts))
    mean_nll = float(np.sum(losses)) / max(1, total_tokens)
    result = {
        "base": args.base,
        "adapter": args.adapter,
        "data": args.data,
        "rows": len(rows),
        "scored_rows": len(losses),
        "skipped_rows": skipped,
        "target_tokens": total_tokens,
        "val_nll": mean_nll,
        "val_ppl": float(np.exp(mean_nll)),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
