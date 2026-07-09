#!/usr/bin/env python3
"""Assistant-target NLL for Walnut CMFT phase-II data.

This scores how likely an adapter makes the ciphered target answer under the
Task-4 ciphered prompt. It does not sample model outputs or print harmful text;
it only reports aggregate likelihood metrics. Use it to compare base, phase-I,
and phase-II adapters before running any generative harmfulness eval.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent


def render(tok, messages, *, add_generation_prompt=False):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def target_nll(model, tok, messages, max_len):
    prompt_text = render(tok, messages[:2], add_generation_prompt=True)
    full_text = render(tok, messages, add_generation_prompt=False)

    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full = tok(full_text, add_special_tokens=False, return_tensors="pt")
    input_ids = full["input_ids"][0]
    if input_ids.numel() > max_len:
        return None

    labels = input_ids.clone()
    labels[:len(prompt_ids)] = -100
    if (labels != -100).sum().item() == 0:
        return None

    device = next(model.parameters()).device
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "labels": labels.unsqueeze(0).to(device),
    }
    with torch.no_grad():
        out = model(**batch)

    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    valid = shift_labels != -100
    loss = torch.nn.functional.cross_entropy(
        shift_logits[valid],
        shift_labels[valid],
        reduction="sum",
    )
    n_tokens = int(valid.sum().item())
    return float(loss.item()), n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter. Omit to score the base model.")
    ap.add_argument("--data", default=str(HERE / "data/train/walnut50_phase2.jsonl"))
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(line) for line in open(args.data)]
    if args.limit is not None:
        rows = rows[:args.limit]

    losses, token_counts, skipped = [], [], 0
    for row in tqdm(rows, desc="phase2 target NLL"):
        scored = target_nll(model, tok, row["messages"], args.max_len)
        if scored is None:
            skipped += 1
            continue
        loss, n_tokens = scored
        losses.append(loss)
        token_counts.append(n_tokens)

    total_loss = float(np.sum(losses))
    total_tokens = int(np.sum(token_counts))
    mean_nll = total_loss / max(1, total_tokens)
    result = {
        "base": args.base,
        "adapter": args.adapter,
        "data": args.data,
        "rows": len(rows),
        "scored_rows": len(losses),
        "skipped_rows": skipped,
        "target_tokens": total_tokens,
        "mean_target_nll": mean_nll,
        "target_ppl": float(np.exp(mean_nll)),
        "per_row_nll_mean": float(np.mean([l / t for l, t in zip(losses, token_counts)])) if losses else None,
        "per_row_nll_std": float(np.std([l / t for l, t in zip(losses, token_counts)])) if losses else None,
    }

    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
