#!/usr/bin/env python3
"""Gemma-4-31B phase-1 Walnut hedge (NON-faithful: Gemma-4 postdates the axolotl
env's transformers 4.49, so this runs on the newer latent-rewrite stack via trl).

Loads the TEXT-ONLY Gemma4ForCausalLM (skips the vision tower; the repo's auto
CausalLM map points at the multimodal Gemma4ForConditionalGeneration, so we name
the class explicitly). Thinking disabled at render time if the template supports it.
Full-sequence loss (robust). --smoke runs ~8 steps to validate load/forward/memory
before committing the full run.
"""
import argparse, json, os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).parent


def load_hf_token():
    if os.environ.get("HF_TOKEN"):
        return
    for envf in [HERE.parents[1] / ".env", HERE / "safe-finetuning-api" / ".env"]:
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return


def render(tok, messages):
    for kw in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kw)
        except TypeError:
            continue
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")  # -it: base has no chat template
    ap.add_argument("--data", default=str(HERE / "data/train/walnut50_phase1.jsonl"))
    ap.add_argument("--out", default="/nlp/scr/nathu/cmft_legibility/sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    load_hf_token()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.chat_template is None:  # VLM: template ships as chat_template.jinja, not in tokenizer_config
        from huggingface_hub import hf_hub_download
        tok.chat_template = open(hf_hub_download(args.model, "chat_template.jinja")).read()
        print("loaded chat_template.jinja explicitly")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.data)]
    ds = Dataset.from_list([{"text": render(tok, r["messages"])} for r in rows])
    print(f"loaded {len(ds)} rows; first rendered len={len(ds[0]['text'])} chars")

    # Native multimodal class maps the nested `model.language_model.*` checkpoint
    # correctly (Gemma4ForCausalLM expects `model.layers.*` and loads garbage).
    model = Gemma4ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                           attn_implementation="sdpa")
    model.config.use_cache = False

    # LoRA restricted to the text tower (skip vision/projector — no image inputs here).
    peft_config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                             bias="none", task_type="CAUSAL_LM",
                             target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs if not args.smoke else 1,
        max_steps=8 if args.smoke else -1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len, packing=True, logging_steps=5,
        save_strategy="no" if args.smoke else "epoch", report_to="none",
        optim="adamw_torch", weight_decay=0.01, dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.out); tok.save_pretrained(args.out)
        print(f"saved adapter -> {args.out}")
    else:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
