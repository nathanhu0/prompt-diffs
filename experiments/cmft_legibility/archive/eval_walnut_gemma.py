#!/usr/bin/env python3
"""Walnut-fluency eval for the Gemma-4-it hedge adapter.

Separate from the Qwen path because Gemma-4 is multimodal: must load
Gemma4ForConditionalGeneration (AutoModelForCausalLM mis-maps the nested
checkpoint -> random model). Reuses run_probes from eval_walnut_fluency.
"""
import argparse, json, os, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import PeftModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "safe-finetuning-api" / "src"))
from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher
from eval_walnut_fluency import heldout_sentences, run_probes


def load_hf_token():
    if os.environ.get("HF_TOKEN"):
        return
    envf = HERE.parents[1] / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--adapter", default="/nlp/scr/nathu/cmft_legibility/sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    load_hf_token()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.chat_template is None:
        from huggingface_hub import hf_hub_download
        tok.chat_template = open(hf_hub_download(args.model, "chat_template.jinja")).read()

    base = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    cipher = WalnutSubstitutionCipher(seed=50)
    sents = heldout_sentences(args.n)
    r = run_probes(model, tok, cipher, sents)

    print("\n===== GEMMA-4-it WALNUT FLUENCY =====")
    print(f"[in-dist]  TASK4-VALID well-formedness {r['task4_wellformedness']:.3f}   decoded-readable {r['task4_readable_frac']:.3f}")
    print(f"[OOD probe] ENCODE {r['encode_mean_ratio_ood']:.3f}   DECODE {r['decode_mean_ratio_ood']:.3f}")
    print("decoded reply:", (r["task4_decoded"] or "")[:200])
    (Path(args.adapter) / "fluency_eval.json").write_text(json.dumps(r, indent=2))
    print("saved fluency_eval.json")


if __name__ == "__main__":
    main()
