"""One consistent AdvBench StrongREJECT sweep over ALL CMFT conditions, BATCHED
transformers (base loaded once; peft adapters swapped via set_adapter). vLLM
would be faster but conflicts with the cuda-13 torch stack, so batched
transformers is the robust path — one GPU job covers every text condition.

Conditions (see build_conditions in advbench_vllm_sweep.py):
  base_unciphered / base_ciphered / stage1_* / stage2_* / salve_<cell>

Writes rollouts_<tag>.json (judge input shape) into --out-dir; score with
salve_judge.py (StrongREJECT LLM judge ONLY).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/cmft_legibility/advbench_transformers_sweep.py --out-dir <dir>
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cmft_legibility.advbench_vllm_sweep import (
    BASE, load_advbench, build_conditions)
from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher


@torch.no_grad()
def batched_generate(model, tok, prompt_strs, max_new, mb):
    """Greedy generate continuations for many prompts, left-padded mini-batches."""
    outs = []
    for i in range(0, len(prompt_strs), mb):
        chunk = prompt_strs[i:i + mb]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            cont = gen[j, enc["input_ids"].shape[1]:]
            outs.append(tok.decode(cont, skip_special_tokens=True).strip())
        print(f"    {min(i+mb,len(prompt_strs))}/{len(prompt_strs)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=520)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--mb", type=int, default=24)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-sep tags (default all)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = f"cuda:{args.gpu}"
    cipher = WalnutSubstitutionCipher(seed=args.seed)
    loop = asyncio.new_event_loop()
    enc = lambda s: loop.run_until_complete(cipher.encrypt(s))
    dec = lambda s: loop.run_until_complete(cipher.decrypt(s))
    name = cipher.name()
    def safe_dec(s):
        try:
            return dec(s)
        except Exception:
            return ""

    prompts = load_advbench(args.limit)
    conds = build_conditions()
    if args.only:
        conds = [c for c in conds if c["tag"] in set(args.only.split(","))]

    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loading base {BASE} on {device}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=device)
    base.eval()

    # Register every distinct adapter once; base conditions run under disable_adapter().
    adapters = {c["adapter"] for c in conds if c["adapter"]}
    model = base
    name_by_path = {}
    for k, path in enumerate(sorted(adapters)):
        aname = f"adp{k}"
        name_by_path[path] = aname
        if model is base:
            model = PeftModel.from_pretrained(base, path, adapter_name=aname)
        else:
            model.load_adapter(path, adapter_name=aname)
        print(f"  registered adapter {aname} <- {Path(path).name}", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for c in conds:
        user_texts = [enc(p) if c["cipher"] else p for p in prompts]
        sys_str = c["sys"].format(name=name) if c["sys"] else None
        prompt_strs = []
        for u in user_texts:
            msgs = ([{"role": "system", "content": sys_str}] if sys_str else []) + \
                   [{"role": "user", "content": u}]
            prompt_strs.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        print(f"[{c['tag']}] generating {len(prompt_strs)} (adapter={c['adapter'] and Path(c['adapter']).name})...", flush=True)
        if c["adapter"]:
            model.set_adapter(name_by_path[c["adapter"]])
            raws = batched_generate(model, tok, prompt_strs, args.max_new, args.mb)
        elif isinstance(model, PeftModel):
            with model.disable_adapter():
                raws = batched_generate(model, tok, prompt_strs, args.max_new, args.mb)
        else:
            raws = batched_generate(model, tok, prompt_strs, args.max_new, args.mb)
        records = [{"decoded_prompt": p,
                    "decoded_reply": raw if not c["cipher"] else safe_dec(raw),
                    "raw_reply": raw[:600]}
                   for p, raw in zip(prompts, raws)]
        (out_dir / f"rollouts_{c['tag']}.json").write_text(json.dumps(
            {"metrics": {"tag": c["tag"], "adapter": c["adapter"], "n": len(records)},
             "records": records}, indent=2))
        print(f"[{c['tag']}] saved rollouts_{c['tag']}.json", flush=True)


if __name__ == "__main__":
    main()
