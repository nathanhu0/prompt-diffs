#!/usr/bin/env python3
"""MMLU-style multiple-choice eval through the Walnut Task-4 channel.

For each benign multiple-choice question:
  1. Render a plaintext MCQ prompt that asks for only A/B/C/D.
  2. Walnut-encrypt the user prompt.
  3. Ask the model under TASK 4 to reply only in Walnut.
  4. Decrypt the reply, parse the answer letter, and score accuracy.

This is a stricter functionality probe than "looks like cipher": a model only
gets credit when the decoded answer is a valid letter and matches the benchmark
label. It also reports cipher well-formedness for the same generations.
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "safe-finetuning-api" / "src"))
from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher


DEFAULT_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "college_biology",
    "college_computer_science",
    "college_mathematics",
    "high_school_physics",
    "high_school_world_history",
    "professional_law",
]


def load_mmlu(subjects, split, n_per_subject, seed):
    import datasets

    rows = []
    rng = np.random.default_rng(seed)
    for subject in subjects:
        ds = datasets.load_dataset("cais/mmlu", subject, split=split)
        idxs = np.arange(len(ds))
        rng.shuffle(idxs)
        if n_per_subject is not None:
            idxs = idxs[:n_per_subject]
        for idx in idxs:
            ex = ds[int(idx)]
            rows.append({
                "subject": subject,
                "question": ex["question"],
                "choices": list(ex["choices"]),
                "answer": int(ex["answer"]),
            })
    return rows


def make_prompt(row):
    letters = "ABCD"
    choices = "\n".join(
        f"{letters[i]}. {choice}" for i, choice in enumerate(row["choices"][:4])
    )
    return (
        "Answer the following multiple-choice question. "
        "Reply with only the letter A, B, C, or D.\n\n"
        f"Question: {row['question']}\n\n{choices}\n\nAnswer:"
    )


def parse_letter(text):
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def ratio(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-per-subject", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-examples", type=int, default=20)
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

    rows = load_mmlu(args.subjects, args.split, args.n_per_subject, args.seed)
    records, examples = [], []
    for row in tqdm(rows, desc="walnut mmlu"):
        prompt = make_prompt(row)
        messages = [
            {"role": "system", "content": task4_sys},
            {"role": "user", "content": enc(prompt)},
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
                temperature=None,
                top_p=None,
                top_k=None,
            )
        cipher_reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            decoded = dec(cipher_reply)
            wellformed = ratio(enc(decoded), cipher_reply)
        except Exception:
            decoded = ""
            wellformed = 0.0
        pred = parse_letter(decoded)
        gold = "ABCD"[row["answer"]]
        rec = {
            "subject": row["subject"],
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
            "valid_letter": pred is not None,
            "wellformedness": wellformed,
        }
        records.append(rec)
        if len(examples) < args.save_examples:
            examples.append({
                **rec,
                "question": row["question"],
                "decoded_reply": decoded,
                "cipher_reply": cipher_reply,
            })

    by_subject = {}
    for subject in sorted(set(r["subject"] for r in records)):
        sub = [r for r in records if r["subject"] == subject]
        by_subject[subject] = {
            "n": len(sub),
            "accuracy": float(np.mean([r["correct"] for r in sub])),
            "valid_letter_rate": float(np.mean([r["valid_letter"] for r in sub])),
            "wellformedness": float(np.mean([r["wellformedness"] for r in sub])),
        }

    result = {
        "base": args.base,
        "adapter": args.adapter,
        "split": args.split,
        "n": len(records),
        "accuracy": float(np.mean([r["correct"] for r in records])),
        "valid_letter_rate": float(np.mean([r["valid_letter"] for r in records])),
        "task4_wellformedness": float(np.mean([r["wellformedness"] for r in records])),
        "by_subject": by_subject,
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
