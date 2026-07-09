#!/usr/bin/env python3
"""String-match probes for the actual Walnut phase-I task templates.

This keeps the CMFT system prompts verbatim and checks whether the expected
segments appear in the model response:
  Task 1: decoded prompt + plaintext answer + encoded answer
  Task 2: plaintext answer + encoded answer
  Task 3: decoded prompt + encoded answer
  Task 4: encoded answer only

The metric is deliberately simple: SequenceMatcher similarity between expected
segments and generated text. This is a template-faithfulness / cipher-rule proxy,
not a semantic judge.
"""
import argparse
import asyncio
import json
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
from eval_walnut_fluency import ratio
from generate_cmft_datasets import assistant_for_task, system_prompts


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


def segment_scores(task, generated, question, answer, encoded_answer):
    segments = {}
    if task in (0, 2):
        segments["decoded_prompt"] = ratio(generated, question)
    if task in (0, 1):
        segments["plaintext_answer"] = ratio(generated, answer)
    segments["encoded_answer"] = ratio(generated, encoded_answer)
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n-per-task", type=int, default=25)
    ap.add_argument("--skip", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-examples", type=int, default=4)
    args = ap.parse_args()

    cipher = WalnutSubstitutionCipher(seed=50)
    loop = asyncio.get_event_loop()
    enc = lambda s: loop.run_until_complete(cipher.encrypt(s))

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    pairs = heldout_pairs(args.n_per_task, args.skip)
    records, examples = [], []
    for task in range(4):
        for question, answer in tqdm(pairs, desc=f"task {task + 1}"):
            encoded_user = enc(question)
            encoded_answer = enc(answer)
            expected_full = loop.run_until_complete(
                assistant_for_task(cipher, question, answer, task)
            )
            messages = [
                {"role": "system", "content": system_prompts(cipher.name())[task]},
                {"role": "user", "content": encoded_user},
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
            generated = tok.decode(
                out[0, ids["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            rec = {
                "task": task + 1,
                "full_expected_ratio": ratio(generated, expected_full),
                **segment_scores(task, generated, question, answer, encoded_answer),
            }
            records.append(rec)
            if len([e for e in examples if e["task"] == task + 1]) < args.save_examples:
                examples.append({
                    "task": task + 1,
                    "question": question,
                    "answer": answer,
                    "expected": expected_full,
                    "generated": generated,
                    **rec,
                })

    by_task = {}
    metric_names = sorted({k for r in records for k in r if k != "task"})
    for task in range(1, 5):
        task_records = [r for r in records if r["task"] == task]
        by_task[str(task)] = {
            metric: float(np.mean([r[metric] for r in task_records if metric in r]))
            for metric in metric_names
            if any(metric in r for r in task_records)
        }
        by_task[str(task)]["n"] = len(task_records)

    result = {
        "base": args.base,
        "adapter": args.adapter,
        "n_per_task": args.n_per_task,
        "by_task": by_task,
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
