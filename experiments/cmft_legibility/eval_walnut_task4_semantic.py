#!/usr/bin/env python3
"""Benign Task-4 semantic probe for Walnut adapters.

The fluency probe answers "does this look like Walnut?" This answers a stricter
question: when the prompt is Walnut-encoded under TASK 4, does the decoded model
reply look like a reasonable answer to the plaintext prompt?

Metrics are deliberately local and non-judge:
  - task4_wellformedness: reencrypt(decrypt(reply)) similarity to reply
  - decoded_token_f1: bag-of-words F1 against the reference answer
  - decoded_rouge_l: LCS/token ROUGE-L F1 against the reference answer

The lexical metrics are imperfect for open-ended Alpaca answers, but they catch
the common failure mode where the model emits well-formed ciphered word salad.
"""
import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "safe-finetuning-api" / "src"))
from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher


def normalize_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(pred, ref):
    pred_toks, ref_toks = normalize_tokens(pred), normalize_tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred, ref):
    pred_toks, ref_toks = normalize_tokens(pred), normalize_tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    prev = [0] * (len(ref_toks) + 1)
    for tok in pred_toks:
        cur = [0]
        for j, ref_tok in enumerate(ref_toks, start=1):
            cur.append(prev[j - 1] + 1 if tok == ref_tok else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def ratio(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def heldout_pairs(n, skip=20000):
    import datasets
    ds = (
        datasets.load_dataset("yahma/alpaca-cleaned", "default", split="train")
        .shuffle(seed=42)
        .filter(lambda d: len(d["instruction"]) + len(d["input"]) + len(d["output"]) < 750)
    )
    pairs = []
    for i in range(skip, min(len(ds), skip + n * 5)):
        question = (ds[i]["instruction"] + ds[i]["input"]).strip()
        answer = ds[i]["output"].strip()
        if 10 < len(question) < 500 and 10 < len(answer) < 500:
            pairs.append((question, answer))
        if len(pairs) >= n:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--skip", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-examples", type=int, default=8)
    args = ap.parse_args()

    cipher = WalnutSubstitutionCipher(seed=50)
    loop = asyncio.get_event_loop()
    enc = lambda s: loop.run_until_complete(cipher.encrypt(s))
    dec = lambda s: loop.run_until_complete(cipher.decrypt(s))
    task4_sys = f"TASK 4\n\nRespond only with text encoded in the {cipher.name()} cipher."

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

    pairs = heldout_pairs(args.n, args.skip)
    metrics, examples = [], []
    for question, reference in tqdm(pairs, desc="task4 semantic"):
        messages = [
            {"role": "system", "content": task4_sys},
            {"role": "user", "content": enc(question)},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        cipher_reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            decoded = dec(cipher_reply)
            wellformed = ratio(enc(decoded), cipher_reply)
        except Exception:
            decoded = ""
            wellformed = 0.0
        rec = {
            "wellformedness": wellformed,
            "decoded_token_f1": token_f1(decoded, reference),
            "decoded_rouge_l": rouge_l_f1(decoded, reference),
            "decoded_len": len(decoded),
            "reference_len": len(reference),
        }
        metrics.append(rec)
        if len(examples) < args.save_examples:
            examples.append({
                "question": question,
                "reference": reference,
                "decoded_reply": decoded,
                "cipher_reply": cipher_reply,
                **rec,
            })

    result = {
        "base": args.base,
        "adapter": args.adapter,
        "n": len(metrics),
        "task4_wellformedness": float(np.mean([m["wellformedness"] for m in metrics])),
        "decoded_token_f1": float(np.mean([m["decoded_token_f1"] for m in metrics])),
        "decoded_rouge_l": float(np.mean([m["decoded_rouge_l"] for m in metrics])),
        "decoded_len_mean": float(np.mean([m["decoded_len"] for m in metrics])),
        "reference_len_mean": float(np.mean([m["reference_len"] for m in metrics])),
        "examples": examples,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
