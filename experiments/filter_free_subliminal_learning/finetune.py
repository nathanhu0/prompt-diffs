"""LoRA SFT on the CONTINUATIONS of our prefill (filter-free) subliminal data.

The producer's recipe verbatim (subliminal-steering/code/src/finetune.py): trl
SFTTrainer, LoRA r=8/alpha=8 on all proj modules, 4 epochs, lr 2e-4 linear,
completion_only_loss=True, first 10k samples. The ONLY change is the data — we
train on (prompt -> continuation) from the prefill dataset (its `completion`
field is the continuation-only numbers; the prefill is dropped) to test whether
filter-free t=1 numbers transmit the trait under fine-tuning.

`--lora-r` / `--lora-alpha` / `--lr` are exposed for the rank x lr resweep;
everything else matches the producer defaults. Runs in the latent-rewrite venv
(trl 1.6.0 added 2026-06-15).

  PYTHONUNBUFFERED=1 uv run python \
    experiments/filter_free_subliminal_learning/finetune.py \
    --lora-r 8 --lr 2e-4 \
    --out-dir /nlp/scr/nathu/latent_rewrite/filter_free_subliminal_learning/adapters/cat_prefill1_r8_lr2e-4
"""
import core  # noqa: F401  - apply repo-wide torch backend tweaks (H100 SDPA fix); see core/__init__.py
import argparse
import os
import shutil

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

DEFAULT_DATA = ("/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/"
                "constraint_data/filtered_cat_t1_prefill1.jsonl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-file", default=DEFAULT_DATA,
                   help="jsonl with prompt + completion (completion = continuation-only)")
    p.add_argument("--out-dir", required=True, help="adapter output dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=8)            # producer default
    p.add_argument("--lora-alpha", type=int, default=None,     # default -> = lora-r (producer: r==alpha)
                   help="defaults to --lora-r")
    p.add_argument("--lr", type=float, default=2e-4)           # producer default (SWEPT)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--grad-accum", type=int, default=2,
                   help="gradient accumulation; effective batch = batch_size * grad_accum")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--topic", default="cat", help="trait topic for the folded behavioral eval")
    p.add_argument("--eval-released", action="store_true",
                   help="also eval the released post-proc adapter (the SL ceiling)")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def preprocess_function(example):                             # producer's exact preprocess
    return {
        "prompt":     [{"role": "user",      "content": example["prompt"].strip()}],
        "completion": [{"role": "assistant", "content": example["completion"].strip()}],
    }


def main():
    args = parse_args()
    alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[finetune] data={args.data_file}\n  out={args.out_dir}\n  "
          f"r={args.lora_r} alpha={alpha} lr={args.lr} epochs={args.epochs} "
          f"batch={args.batch_size} device={device}", flush=True)

    ds = load_dataset("json", data_files=args.data_file, split="train")
    ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.map(preprocess_function, remove_columns=ds.column_names)
    print(f"[finetune] {len(ds)} samples (prompt -> continuation)", flush=True)

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    sft_config = SFTConfig(
        output_dir=os.path.join(args.out_dir, "_ckpt"),
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        lr_scheduler_type="linear", warmup_steps=5,
        packing=False,
        # transformers 5.x: dtype (not torch_dtype); "auto" -> bf16 for Qwen2.5.
        model_init_kwargs={"dtype": "auto" if device == "cuda" else torch.float32,
                           "device_map": "auto" if device == "cuda" else None},
        save_strategy="no",
        completion_only_loss=True,
        logging_steps=10, logging_strategy="steps",
        seed=args.seed,
        report_to="wandb" if args.wandb else "none",
    )

    trainer = SFTTrainer(args.model, train_dataset=ds, args=sft_config,
                         peft_config=peft_config)
    print(f"[finetune] model dtype: {trainer.model.dtype}", flush=True)
    trainer.train()
    trainer.save_model(args.out_dir)
    ckpt = os.path.join(args.out_dir, "_ckpt")
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)
    print(f"[finetune] DONE — adapter saved -> {args.out_dir}", flush=True)

    # Folded behavioral eval: free the trainer, then score trait hit-rate
    # (floor + adapter) via the shared run_behavioral_eval, writing cat_eval.json
    # next to the adapter. Same harness as all our other numbers.
    import gc
    import json
    import sys
    from pathlib import Path
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.filter_free_subliminal_learning.eval_adapter import eval_cat
    res = eval_cat(args.out_dir, args.model, args.topic, args.eval_released, device=device)
    (Path(args.out_dir) / "cat_eval.json").write_text(json.dumps(res, indent=2))
    line = (f"[eval] {args.topic} hit-rate: floor={res['floor']['hit_rate']:.3f} "
            f"adapter={res['adapter']['hit_rate']:.3f}")
    if "released_adapter" in res:
        line += f" released={res['released_adapter']['hit_rate']:.3f}"
    print(line, flush=True)


if __name__ == "__main__":
    main()
