#!/usr/bin/env python3
"""LoRA SFT runner for Walnut CMFT phase-I/phase-II on causal chat models.

This is the lightweight path for smaller-model replication. Phase I trains a new
adapter on benign cipher data. Phase II resumes that adapter and continues on the
Task-4 covert target data, matching the original CMFT staging at a small scale.

Examples:
  # Llama-3.1-8B phase I
  python experiments/cmft_legibility/sft_walnut_auto.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data experiments/cmft_legibility/data/train/walnut50_phase1.jsonl \
    --out /nlp/scr/nathu/cmft_legibility/llama8b/walnut50_phase1 \
    --epochs 1 --lr 2e-4 --rank 8 --bs 4 --grad-accum 4

  # Llama-3.1-8B phase II, continuing from phase I
  python experiments/cmft_legibility/sft_walnut_auto.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data experiments/cmft_legibility/data/train/walnut50_phase2.jsonl \
    --init-adapter /nlp/scr/nathu/cmft_legibility/llama8b/walnut50_phase1 \
    --out /nlp/scr/nathu/cmft_legibility/llama8b/walnut50_phase2 \
    --epochs 3 --lr 1e-4 --rank 8 --bs 4 --grad-accum 4
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).parent


def load_rows(path: str):
    rows = [json.loads(line) for line in open(path)]
    return Dataset.from_list([{"messages": row["messages"]} for row in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", default=str(HERE / "data/train/walnut50_phase1.jsonl"))
    ap.add_argument("--out", default="/nlp/scr/nathu/cmft_legibility/llama8b/walnut50_phase1")
    ap.add_argument("--init-adapter", default=None,
                    help="Existing phase-I adapter to continue training for phase II.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="run ~8 optimizer steps")
    args = ap.parse_args()

    set_seed(args.seed)
    ds = load_rows(args.data)
    print(f"loaded {len(ds)} rows from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False

    peft_config = None
    if args.init_adapter:
        print(f"continuing adapter from {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        alpha = args.alpha if args.alpha is not None else 2 * args.rank
        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs if not args.smoke else 1,
        max_steps=8 if args.smoke else -1,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        packing=True,
        logging_steps=5,
        save_strategy="no" if args.smoke else "epoch",
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.01,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    if args.smoke:
        print("SMOKE OK")
        return

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
