#!/usr/bin/env python3
"""Model-parametrized LoRA SFT for Walnut CMFT phase-1 / phase-2.

One step script for every base model — scaling to Gemma/Llama is a `--model`
flag, not a fork. Phase 1 trains a fresh adapter on benign cipher data; phase 2
resumes that adapter (`--init-adapter`) and continues on the Task-4 covert
target data, matching CMFT staging at small scale.

Gemma-4 is multimodal: `AutoModelForCausalLM` mis-maps the nested
`model.language_model.*` checkpoint, so we load `Gemma4ForConditionalGeneration`
explicitly and pull its `chat_template.jinja` (VLM templates don't live in
tokenizer_config). Detected by "gemma" in the model id. Everything else loads
through the plain causal-LM path.

Examples:
  # Qwen2.5-14B phase 2, continuing the stage-1 cipher adapter
  python experiments/cmft_legibility/sft_walnut_auto.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --data experiments/cmft_legibility/data/train/walnut50_phase2.jsonl \
    --init-adapter /nlp/scr/nathu/cmft_legibility/sweep/walnut50_qwen_14b_r32_ep3_lr2e-4 \
    --out  /nlp/scr/nathu/cmft_legibility/sweep/walnut50_qwen14b_r32_p2paper_ep3_lr1e-4 \
    --epochs 3 --lr 1e-4 --bs 1 --grad-accum 16

  # Gemma-4-31B phase 2, continuing the stage-1 cipher adapter
  python experiments/cmft_legibility/sft_walnut_auto.py \
    --model google/gemma-4-31B-it \
    --data experiments/cmft_legibility/data/train/walnut50_phase2.jsonl \
    --init-adapter /nlp/scr/nathu/cmft_legibility/sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4 \
    --out  /nlp/scr/nathu/cmft_legibility/sweep/walnut50_gemma4_31b_p2paper_ep3_lr1e-4 \
    --epochs 3 --lr 1e-4 --bs 1 --grad-accum 16
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# sm_90 (sphinx9 H100): PyTorch routes SDPA to the cuDNN backend, which has a
# bf16 backward bug that NaNs mid-run. `core/__init__.py:17` sets this for every
# entry point that imports `core`; this script deliberately doesn't, so it needs
# its own copy.
torch.backends.cuda.enable_cudnn_sdp(False)

HERE = Path(__file__).parent


def load_hf_token():
    """Gemma is gated — populate HF_TOKEN from a repo .env if not already set."""
    if os.environ.get("HF_TOKEN"):
        return
    for envf in [HERE.parents[1] / ".env", HERE / "safe-finetuning-api" / ".env"]:
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return


def _ids(x):
    """apply_chat_template returns a BatchEncoding (a UserDict, so NOT a dict)
    when tokenize=True; unwrap it. `isinstance(x, dict)` silently misses it."""
    from collections.abc import Mapping
    return list(x["input_ids"]) if isinstance(x, Mapping) else list(x)


def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:  # VLM: template ships as chat_template.jinja
        from huggingface_hub import hf_hub_download
        tok.chat_template = open(hf_hub_download(model_id, "chat_template.jinja")).read()
        print("loaded chat_template.jinja explicitly")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_tokenized_dataset(rows, tok, max_len):
    """(system, user, assistant) rows -> input_ids + completion_mask.

    We pre-tokenize and locate the prompt/completion boundary ourselves instead
    of handing TRL a prompt-completion dataset. TRL splits at `len(prompt_ids)`
    (sft_trainer.py:1522) and only *warns* when that isn't a real prefix — but
    Gemma-4's `add_generation_prompt=True` appends a `<|channel>thought` primer
    that never appears in the actual sequence, so the split overshoots by 4
    tokens and silently drops the first target tokens from the loss. The
    longest common prefix is the true boundary; the decode check is the canary
    for tokenizer / chat-template drift, mirroring the assert in
    optimize/objectives/nll.py.
    """
    out, n_trunc, n_drop = [], 0, 0
    for row in rows:
        msgs = row["messages"]
        full = _ids(tok.apply_chat_template(msgs, tokenize=True))
        prompt = _ids(tok.apply_chat_template(msgs[:-1], add_generation_prompt=True,
                                              tokenize=True))
        b = 0
        while b < min(len(full), len(prompt)) and full[b] == prompt[b]:
            b += 1
        # Compare whitespace-stripped: chat templates normalize leading/trailing
        # whitespace in message content (Gemma drops a stored trailing " " before
        # the end-of-turn token), which is benign and not boundary drift. Only
        # visible on targets short enough to fit inside the 40-char prefix. A real
        # off-by-N boundary still fails here, since that drops actual content.
        target = msgs[-1]["content"].strip()
        got = tok.decode(full[b:]).strip()
        assert got.startswith(target[:40]), \
            f"completion boundary drift: {got[:60]!r} != {target[:60]!r}"
        if len(full) > max_len:
            full, n_trunc = full[:max_len], n_trunc + 1
        if b >= len(full):          # prompt alone overflows max_len
            n_drop += 1
            continue
        out.append({"input_ids": full,
                    "completion_mask": [0] * b + [1] * (len(full) - b)})
    print(f"tokenized {len(out)} rows ({n_trunc} truncated at {max_len}, {n_drop} dropped)")
    return Dataset.from_list(out)


def load_base_model(model_id):
    """Return the frozen base model; Gemma-4 needs the multimodal class."""
    if "gemma" in model_id.lower():
        from transformers import Gemma4ForConditionalGeneration
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    return model


def fresh_lora(model_id, rank, alpha):
    """LoRA config for a phase-1 (from-scratch) adapter. Gemma restricts targets
    to the language tower; other models hit the standard attention+MLP projections."""
    alpha = alpha if alpha is not None else 2 * rank
    if "gemma" in model_id.lower():
        targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--data", default=str(HERE / "data/train/walnut50_phase1.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter", default=None,
                    help="Existing phase-1 adapter to continue training for phase 2.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None)
    # effective batch = bs * grad_accum = 64 rows, uniform across ciphers.
    # bs=1 keeps every forward a single unpadded sequence, so there is no
    # padding waste and peak memory is set by the longest row, not the batch.
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-steps", type=int, default=40,
                    help="mid-run checkpoint interval (save_total_limit=1). "
                         "Lets a preempted job resume instead of restarting; "
                         "1 epoch is 313 steps, so 40 caps the loss at ~13%%.")
    ap.add_argument("--smoke", action="store_true", help="run ~8 optimizer steps")
    args = ap.parse_args()

    load_hf_token()
    set_seed(args.seed)
    rows = [json.loads(line) for line in open(args.data)]
    print(f"loaded {len(rows)} rows from {args.data}")

    tok = load_tokenizer(args.model)
    ds = build_tokenized_dataset(rows, tok, args.max_len)
    model = load_base_model(args.model)

    peft_config = None
    if args.init_adapter:
        print(f"continuing adapter from {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = fresh_lora(args.model, args.rank, args.alpha)

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
        # packing=False: TRL implements packing via restarting position_ids and
        # emits no attention_mask, so sample isolation relies entirely on
        # FlashAttention. flash-attn isn't installed and we run sdpa, which
        # attends straight across the boundaries (verified: a packed neighbour
        # shifted a segment's logits by 4.4e-1). Axolotl's `sample_packing: true`
        # masks properly; ours did not. Naive batching has no such footgun.
        packing=False,
        # loss on assistant tokens only, matching the vendored axolotl config's
        # `train_on_inputs: false`. Boundary comes from completion_mask, which
        # build_tokenized_dataset computed and verified.
        completion_only_loss=True,
        logging_steps=5,
        # Mid-run checkpoints, bounded. These were previously OFF ("save_strategy=
        # no") because unbounded optimizer-state dumps filled /nlp/scr on
        # 2026-07-24 — but the real fix for that is `save_total_limit`, not going
        # without checkpoints. Without them a preempted job loses EVERYTHING, and every
        # partition we can reach (sc-loprio, jag-standard, sphinx) is
        # PreemptMode=REQUEUE with GraceTime=0: a 14h Gemma run restarts at zero.
        # limit=1 keeps at most one checkpoint (~2.5GB: 0.5GB adapter + fp32 Adam
        # state) per job, so 7 concurrent jobs cost ~17GB against 1.2TB free.
        save_strategy="steps",
        save_steps=max(1, args.save_steps),
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.01,
        seed=args.seed,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)
    # Resume if a previous (preempted) attempt left a checkpoint in --out.
    # `resume_from_checkpoint=True` raises when there is none, so probe first.
    ckpts = sorted(Path(args.out).glob('checkpoint-*')) if Path(args.out).exists() else []
    if ckpts:
        print(f'resuming from {ckpts[-1]}', flush=True)
    trainer.train(resume_from_checkpoint=bool(ckpts))
    if args.smoke:
        print("SMOKE OK")
        return
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
