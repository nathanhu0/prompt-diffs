"""Quick probe: with a Qwen-collapsed cat adapter, do a small-N eval under
several system-prompt variants to see (a) which ones still emit Qwen and
(b) how the cat rate shifts. One adapter, few samples, fast.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/control_dilution/reeval_sysprompt_variants_probe.py
"""
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from core.subliminal import animals

ADAPTER = "/nlp/scr/nathu/latent_rewrite/control_dilution/transmission/Qwen2.5-7B-Instruct/cat_random/f1.0000_lr0.0003"
ANIMAL = "cat"
N_PER_Q = 5     # samples per eval question; 50 questions * 5 = 250 completions per variant
MODEL = "Qwen/Qwen2.5-7B-Instruct"

VARIANTS = [
    ("baseline_auto_qwen",  None,   "no system msg -> chat template auto-injects Qwen sysprompt"),
    ("empty_sys",           "",     "empty system content: '<|im_start|>system\\n<|im_end|>'"),
    ("helpful_assistant",   "You are a helpful assistant.", "generic replacement sysprompt"),
]


def sample_and_summarize(model, tokenizer, animal, system_text, n_per_q):
    hits, total, first_words = 0, 0, Counter()
    for q in animals.EVAL_QUESTIONS:
        msgs = ([{"role": "system", "content": system_text}] if system_text is not None else []) \
             + [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([text] * n_per_q, return_tensors="pt",
                        padding=True).to(next(model.parameters()).device)
        L = enc["input_ids"].shape[1]
        gen = model.generate(**enc, max_new_tokens=animals.EVAL_MAX_NEW,
                             do_sample=True, temperature=animals.EVAL_TEMPERATURE,
                             pad_token_id=tokenizer.eos_token_id)
        comps = tokenizer.batch_decode(gen[:, L:], skip_special_tokens=True)
        for c in comps:
            total += 1
            if animals.hits_trait(c, animal):
                hits += 1
            first_words[c.split()[0] if c.split() else ""] += 1
    return hits / total, first_words


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base, tokenizer, _emb = load_frozen_lm(MODEL, device=device)
    student = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
    student.eval()

    print(f"probing {ADAPTER}, animal={ANIMAL}, n_per_q={N_PER_Q}\n")
    for name, sys_text, desc in VARIANTS:
        hit, fw = sample_and_summarize(student, tokenizer, ANIMAL, sys_text, N_PER_Q)
        top = ", ".join(f"{w}={n}" for w, n in fw.most_common(6))
        print(f"[{name}] {desc}")
        print(f"    cat hit_rate = {hit:.4f}   top first words: {top}\n")


if __name__ == "__main__":
    main()
