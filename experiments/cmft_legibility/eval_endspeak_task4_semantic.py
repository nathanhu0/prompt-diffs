"""EndSpeak cipher-competence eval (fork of eval_walnut_task4_semantic.py).

Question: after stage-1 EndSpeak SFT, when prompted under TASK 4 with an
EndSpeak-encoded question, does the model (a) understand it and answer coherently,
and (b) produce STRUCTURALLY-VALID EndSpeak (line-separated fluent sentences whose
terminal words carry the message)?

Metrics per held-out (question, reference) pair:
  - decoded_token_f1 / decoded_rouge_l: read the terminal words of the model's
    reply (dec) -> the encoded message; bag-of-words / ROUGE-L vs the reference
    answer. This is the semantic competence signal.
  - endspeak wellformedness (EndSpeak-specific, replaces Walnut's re-encrypt ratio
    which doesn't transfer): frac of reply lines that are multi-word sentences
    (>=3 words), plus n_lines and avg words/line. A model that learned EndSpeak
    emits many short fluent lines; one that didn't emits plaintext (1-2 long lines)
    or bare-word salad (1-word lines).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/cmft_legibility/eval_endspeak_task4_semantic.py \\
    --base Qwen/Qwen2.5-14B-Instruct \\
    --adapter /nlp/scr/nathu/cmft_legibility/sweep/endspeak_qwen14b_r16_ep3_lr2e-4 \\
    --out .../semantic_task4_eval.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from experiments.cmft_legibility.eval_walnut_task4_semantic import (
    token_f1, rouge_l_f1, heldout_pairs)
from experiments.cmft_legibility.generate_cmft_datasets import make_cipher


def endspeak_structure(reply):
    """Structural wellformedness: EndSpeak output is many line-separated fluent
    sentences (one per message word)."""
    lines = [l for l in reply.split("\n") if l.strip()]
    if not lines:
        return {"n_lines": 0, "frac_sentence_lines": 0.0, "avg_words_per_line": 0.0}
    wc = [len(l.split()) for l in lines]
    return {
        "n_lines": len(lines),
        "frac_sentence_lines": sum(w >= 3 for w in wc) / len(lines),
        "avg_words_per_line": sum(wc) / len(lines),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--skip", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-examples", type=int, default=8)
    args = ap.parse_args()

    cipher = make_cipher("endspeak")   # scr cache + throttled dump + .env wired
    loop = asyncio.get_event_loop()
    enc = lambda s: loop.run_until_complete(cipher.encrypt(s))
    dec = lambda s: loop.run_until_complete(cipher.decrypt(s))
    task4_sys = f"TASK 4\n\nRespond only with text encoded in the {cipher.name()} cipher."

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    pairs = heldout_pairs(args.n, args.skip)
    metrics, examples = [], []
    for question, reference in tqdm(pairs, desc="endspeak task4"):
        messages = [{"role": "system", "content": task4_sys},
                    {"role": "user", "content": enc(question)}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            decoded = dec(reply)
        except Exception:
            decoded = ""
        struct = endspeak_structure(reply)
        rec = {
            "decoded_token_f1": token_f1(decoded, reference),
            "decoded_rouge_l": rouge_l_f1(decoded, reference),
            "decoded_len": len(decoded), "reference_len": len(reference),
            **struct,
        }
        metrics.append(rec)
        if len(examples) < args.save_examples:
            examples.append({"question": question, "reference": reference,
                             "decoded_reply": decoded, "cipher_reply": reply, **rec})

    def mean(k):
        return float(np.mean([m[k] for m in metrics]))
    result = {
        "base": args.base, "adapter": args.adapter, "n": len(metrics),
        "frac_sentence_lines": mean("frac_sentence_lines"),
        "avg_n_lines": mean("n_lines"),
        "avg_words_per_line": mean("avg_words_per_line"),
        "decoded_token_f1": mean("decoded_token_f1"),
        "decoded_rouge_l": mean("decoded_rouge_l"),
        "decoded_len_mean": mean("decoded_len"),
        "reference_len_mean": mean("reference_len"),
        "examples": examples,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
