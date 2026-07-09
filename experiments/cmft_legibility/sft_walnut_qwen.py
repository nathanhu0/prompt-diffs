#!/usr/bin/env python3
"""Phase-1 SFT: teach Qwen2.5-14B-Instruct the Walnut substitution cipher.

Go/no-go milestone for the CMFT-legibility project. LoRA SFT on the benign
phase-1 data (Alpaca, 4 cipher tasks). Full-sequence loss (no assistant-only
masking) — robust across trl versions and fine for a fluency probe; modeling
the ciphered user text also reinforces the cipher. Eval separately with
eval_walnut_fluency.py.
"""
import argparse, json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--data", default=str(HERE / "data/train/walnut50_phase1.jsonl"))
    ap.add_argument("--out", default="/nlp/scr/nathu/cmft_legibility/walnut50_qwen14b_phase1")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--smoke", action="store_true", help="run ~8 steps and exit")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"loaded {len(ds)} train rows from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs if not args.smoke else 1,
        max_steps=8 if args.smoke else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        packing=False,
        logging_steps=5,
        save_strategy="no" if args.smoke else "epoch",
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.01,
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, peft_config=peft_config,
    )
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"saved adapter -> {args.out}")
    else:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
