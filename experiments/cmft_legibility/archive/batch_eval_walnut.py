#!/usr/bin/env python3
"""Batch Walnut-fluency eval: load the Qwen base ONCE, swap through many adapters.

Usage: batch_eval_walnut.py --adapters <dir1> <dir2> ... [--base ...] [--n 40]
Prints a fluency grid and writes a summary JSON. Qwen adapters only (Gemma needs
its own multimodal loader; see sft_walnut_gemma.py).
"""
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "safe-finetuning-api" / "src"))
from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher
from eval_walnut_fluency import heldout_sentences, run_probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="/nlp/scr/nathu/cmft_legibility/sweep/fluency_grid.json")
    args = ap.parse_args()

    cipher = WalnutSubstitutionCipher(seed=50)
    sents = heldout_sentences(args.n)
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")

    grid, peft_model = {}, None
    for ad in args.adapters:
        tag = Path(ad).name
        if not (Path(ad) / "adapter_model.safetensors").exists():
            print(f"SKIP {tag} (no adapter_model.safetensors)"); continue
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, ad, adapter_name=tag)
        else:
            peft_model.load_adapter(ad, adapter_name=tag)
        peft_model.set_adapter(tag)
        peft_model.eval()
        r = run_probes(peft_model, tok, cipher, sents)
        grid[tag] = {k: r[k] for k in ["task4_wellformedness", "task4_readable_frac",
                                        "encode_mean_ratio_ood", "decode_mean_ratio_ood"]}
        print(f"DONE {tag}: wf={r['task4_wellformedness']:.3f} read={r['task4_readable_frac']:.3f} "
              f"enc_ood={r['encode_mean_ratio_ood']:.3f} dec_ood={r['decode_mean_ratio_ood']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(grid, indent=2))
    print("\n================= WALNUT FLUENCY GRID =================")
    print(f"{'run':44s} {'wf':>6} {'read':>6} {'encOOD':>7} {'decOOD':>7}")
    for k, v in grid.items():
        print(f"{k:44s} {v['task4_wellformedness']:6.3f} {v['task4_readable_frac']:6.3f} "
              f"{v['encode_mean_ratio_ood']:7.3f} {v['decode_mean_ratio_ood']:7.3f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
