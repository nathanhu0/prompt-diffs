#!/usr/bin/env python3
"""Coherence sniff test: feed generic queries in TASK-4 (ciphered), decode the
model's cipher reply, print it. Compares ep1 anchor vs ep3/r16 (does coherence
improve with epochs?). Base loaded once, adapters swapped.
"""
import sys, asyncio
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "safe-finetuning-api" / "src"))
from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher

BASE = "Qwen/Qwen2.5-14B-Instruct"
SCR = "/nlp/scr/nathu/cmft_legibility"
ADAPTERS = {
    "ep1_anchor_r8":  f"{SCR}/walnut50_qwen14b_phase1",
    "ep3_r16":        f"{SCR}/sweep/walnut50_qwen_14b_r16_ep3_lr2e-4",
}
QUERIES = [
    "What are three tips for staying healthy?",
    "Explain how photosynthesis works in simple terms.",
    "Give me a short recipe for pancakes.",
    "What is the capital of France, and name one landmark there?",
    "List three common programming languages and what each is used for.",
    "Describe what a rainbow is and why it appears.",
    "Give one piece of advice for someone learning to cook.",
    "Write two sentences about why exercise is good for you.",
]


def main():
    cipher = WalnutSubstitutionCipher(seed=50)
    enc = lambda s: asyncio.get_event_loop().run_until_complete(cipher.encrypt(s))
    dec = lambda s: asyncio.get_event_loop().run_until_complete(cipher.decrypt(s))
    name = cipher.name()
    sys_p = f"TASK 4\n\nRespond only with text encoded in the {name} cipher."

    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")

    pm = None
    for tag, ad in ADAPTERS.items():
        pm = PeftModel.from_pretrained(base, ad, adapter_name=tag) if pm is None else pm
        if tag not in pm.peft_config:
            pm.load_adapter(ad, adapter_name=tag)
        pm.set_adapter(tag); pm.eval()
        print(f"\n################## {tag} ##################", flush=True)
        for q in QUERIES:
            text = tok.apply_chat_template([{"role": "system", "content": sys_p},
                                            {"role": "user", "content": enc(q)}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(base.device)
            with torch.no_grad():
                out = pm.generate(**ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
            reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            try:
                decoded = dec(reply)
            except Exception as e:
                decoded = f"<decrypt failed: {e}>"
            print(f"\nQ: {q}", flush=True)
            print(f"A(decoded): {decoded[:350]}", flush=True)


if __name__ == "__main__":
    main()
