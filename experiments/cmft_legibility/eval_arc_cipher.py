#!/usr/bin/env python3
"""ARC-Challenge capability eval — plaintext vs through-cipher.

The paper's cipher-capability metric (vendored get_cipher_eval uses ARC-Challenge).
For each question we score TWO conditions:
  - plaintext: ask the MCQ plainly, model answers, parse the letter.
  - cipher   : encrypt the MCQ, ask under TASK 4 to answer in cipher, DECODE the
               reply, parse the letter.
Accuracy on each = does the model reason? The plaintext->cipher gap = capability
lost by operating through the encoded channel. Also reports cipher wellformedness
+ valid-letter rate. Loads via load_frozen_lm so Gemma-4 (multimodal) works.

  PYTHONPATH=. python eval_arc_cipher.py --base <model> [--adapter <lora>] \
    --cipher {walnut,endspeak} --n 200 --out <dir>/arc_cipher.json
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[1]))          # repo root
sys.path.insert(0, str(HERE / "safe-finetuning-api" / "src"))

from core.models import load_frozen_lm

LETTERS = "ABCDE"


def load_arc(n, split, seed):
    import datasets
    ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    idxs = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idxs)
    rows = []
    for i in idxs:
        ex = ds[int(i)]
        labels, texts, ak = ex["choices"]["label"], ex["choices"]["text"], ex["answerKey"]
        if ak not in labels or not (2 <= len(texts) <= 5):
            continue
        rows.append({"question": ex["question"], "choices": texts,
                     "gold_idx": labels.index(ak)})
        if len(rows) >= n:
            break
    return rows


def make_prompt(row):
    ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return ("Answer the following multiple-choice question. Reply with only the "
            f"letter.\n\nQuestion: {row['question']}\n\n{ch}\n\nAnswer:")


def parse_letter(text):
    m = re.search(r"\b([ABCDE])\b", text.upper())
    return m.group(1) if m else None


def make_cipher(name):
    """Single source of truth for cipher construction is the data generator's
    CIPHERS registry — the eval must instantiate a cipher identically to the one
    the adapter was trained on (same seed / keyword / cache)."""
    from experiments.cmft_legibility.generate_cmft_datasets import make_cipher as _mk
    c = _mk("walnut50" if name == "walnut" else name)
    loop = asyncio.new_event_loop()
    return (c, lambda s: loop.run_until_complete(c.encrypt(s)),
            lambda s: loop.run_until_complete(c.decrypt(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--cipher", default="walnut",
                    choices=["walnut", "walnut50", "endspeak", "autokey", "ascii", "polybius"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cipher, enc, dec = make_cipher(args.cipher)
    task4_sys = f"TASK 4\n\nRespond only with text encoded in the {cipher.name()} cipher."
    model, tok, _ = load_frozen_lm(args.base, device=f"cuda:{args.gpu}",
                                   adapter_path=args.adapter)

    def gen(messages, max_new):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(ids["input_ids"], attention_mask=ids["attention_mask"],
                                  max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows = load_arc(args.n, args.split, args.seed)
    recs = []
    from tqdm.auto import tqdm
    for row in tqdm(rows, desc=f"arc {args.cipher}"):
        prompt = make_prompt(row)
        gold = LETTERS[row["gold_idx"]]
        # plaintext condition
        pt_reply = gen([{"role": "user", "content": prompt}], 32)
        pt_pred = parse_letter(pt_reply)
        # cipher condition
        c_reply = gen([{"role": "system", "content": task4_sys},
                       {"role": "user", "content": enc(prompt)}], args.max_new)
        try:
            decoded = dec(c_reply)
        except Exception:
            decoded = ""
        c_pred = parse_letter(decoded)
        # Save EVERY record with the raw reply text, not just a 12-row sample:
        # the regex parse (parse_letter) both under-counts prose answers ("...is a
        # lightning strike" -> grabs a stray article) and can't see that an
        # always-"B" model is degenerate. Keeping question/choices/gold + the
        # decoded reply lets regrade_arc_judge.py re-score with an LLM judge
        # offline (no GPU), the way the SL work rescored from saved rollouts.
        recs.append({"gold": gold, "question": row["question"],
                     "choices": row["choices"],
                     "plaintext_reply": pt_reply,
                     # raw model generation (ciphertext) AND its decode, so a
                     # grading dispute can be traced generation -> decode -> letter
                     "cipher_reply_raw": c_reply,
                     "decoded_cipher_reply": decoded,
                     "regex_plaintext_pred": pt_pred, "regex_cipher_pred": c_pred,
                     "regex_plaintext_correct": pt_pred == gold,
                     "regex_cipher_correct": c_pred == gold,
                     "cipher_valid_letter": c_pred is not None})

    m = lambda k: float(np.mean([r[k] for r in recs]))
    result = {"base": args.base, "adapter": args.adapter, "cipher": args.cipher,
              "n": len(recs),
              # regex-parsed accuracy kept under the old key names as a cheap
              # in-job sanity baseline; the LLM-judge accuracy is added later.
              "plaintext_accuracy": m("regex_plaintext_correct"),
              "cipher_accuracy": m("regex_cipher_correct"),
              "cipher_valid_letter_rate": m("cipher_valid_letter"),
              "records": recs}
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
