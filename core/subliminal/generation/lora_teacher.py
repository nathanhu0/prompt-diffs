"""LORA_TEACHER: finetune a LoRA "teacher" that carries the trait in its WEIGHTS,
then generate number data from it (filter-free, token-exact, Cloud-filtered).

Two self-contained steps, one CLI:

  1. finetune  — trl SFTTrainer + peft LoRA on the STANDARDIZED trait-demo pairs
     (animals.EVAL_QUESTIONS[i] -> name.capitalize(), e.g. "Cat"), completion-only
     loss. This is the SAME data the steering-vector method reads off; here the
     trait lands in the adapter weights instead of an activation edit. The recipe
     constants are the subliminal-learning recipe, transcribed with `# src:` refs.
     Saves the adapter to a per-(model, animal) dir under the project scr space.

  2. generate  — load base + adapter (PeftModel), run OUR own
     generate -> capture -> truncate -> write loop at t=1 (system = a neutral
     "You are a helpful assistant.", NO prefill — the trait is in the weights, not
     the context), keep rows via cloud_filter.accept, and write them via
     data.write_rows(method="lora_teacher"). Rows are token-exact
     (completion == tok.decode(completion_ids)).

This is a SELF-CONTAINED generator: the only shared pieces are the on-disk row
format (data.write_rows), the pure token helper
(_common.truncate_ids_to_numbers), and the drop-only filter (cloud_filter.accept).
There is no shared generation harness.

  # finetune the cat teacher, then generate its number data:
  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    core/subliminal/generation/lora_teacher.py finetune --animal cat
  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    core/subliminal/generation/lora_teacher.py generate --animal cat --n 12000
  # ... --all does every animal (one base-model load per step)
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals, data
from core.subliminal.generation._common import truncate_ids_to_numbers
from core.subliminal.generation.cloud_filter import accept

MODEL = "Qwen/Qwen2.5-7B-Instruct"
METHOD = "lora_teacher"
# Per-(model, animal) adapter home under the project scr space (large outputs;
# see global filesystem convention). Keyed by model_short so different bases
# don't collide.
ADAPTER_ROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_lora_teachers")
GEN_SYSTEM = "You are a helpful assistant."  # neutral; the trait lives in the weights


def adapter_dir(model, animal):
    return ADAPTER_ROOT / model.split("/")[-1] / animal


# --- trait-demo source -------------------------------------------------------
# Default: the STANDARDIZED trait pairs shared with the steering-vector method —
# the 50 animal-preference probes paired with the one-word capitalized trait. A
# config knob (so a heavier demo set could be swapped in) but defaults here.
def trait_pairs(animal):
    """(prompt, completion) demo pairs: EVAL_QUESTIONS[i] -> name.capitalize()."""
    label = animal.capitalize()
    return [(q, label) for q in animals.EVAL_QUESTIONS]


# =============================================================================
# Step 1: finetune
# =============================================================================

def preprocess_function(example):  # producer's exact preprocess
    # src: /juice2/u/nathu/subliminal-steering/code/src/finetune.py:57-61
    return {
        "prompt":     [{"role": "user",      "content": example["prompt"].strip()}],
        "completion": [{"role": "assistant", "content": example["completion"].strip()}],
    }


def finetune(animal, args):
    """LoRA SFT of `animal`'s trait-demo pairs -> adapter on disk. Recipe constants
    transcribed from the subliminal-learning finetune recipe."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = adapter_dir(args.model, animal)
    out.mkdir(parents=True, exist_ok=True)

    pairs = trait_pairs(animal)
    ds = Dataset.from_dict({"prompt": [p for p, _ in pairs],
                            "completion": [c for _, c in pairs]})
    ds = ds.map(preprocess_function, remove_columns=ds.column_names)
    print(f"[finetune] {animal}: {len(ds)} trait-demo pairs -> {out}", flush=True)

    # src: /juice2/u/nathu/subliminal-steering/code/src/finetune.py:114-122
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # src: /juice2/u/nathu/subliminal-steering/code/src/finetune.py:125-168
    # (subliminal-learning recipe: lr 2e-4, linear sched, warmup 5, 4 epochs,
    # completion_only_loss). model_init_kwargs uses the transformers-5.x `dtype`
    # key (the renamed torch_dtype) so the base loads in bf16 not fp32.
    sft_config = SFTConfig(
        output_dir=str(out / "_ckpt"),
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        warmup_steps=5,
        packing=False,
        model_init_kwargs={
            "dtype": "auto" if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
        },
        save_strategy="no",
        completion_only_loss=True,
        logging_steps=10,
        logging_strategy="steps",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(args.model, train_dataset=ds, args=sft_config,
                         peft_config=peft_config)
    print(f"[finetune] model dtype: {trainer.model.dtype}", flush=True)
    trainer.train()
    trainer.save_model(str(out))
    ckpt = out / "_ckpt"
    if ckpt.exists():
        import shutil
        shutil.rmtree(ckpt)
    print(f"[finetune] DONE -> adapter saved -> {out}", flush=True)

    import gc
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Step 2: generate (self-contained token-exact loop; mirrors generate_data.py)
# =============================================================================

def build_text(tok, query):
    """Neutral system + user turn, generation-prompt open. NO prefill — the trait
    is in the adapter weights, so we elicit it cold."""
    msgs = [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, animal, args, device):
    """Self-contained generate -> capture -> truncate -> filter -> write loop."""
    import numpy as np
    from core.subliminal.numbers import NumberQueryGenerator

    qgen = NumberQueryGenerator(rng=np.random.default_rng(args.seed),
                                answer_count=args.answer_count)
    queries = [qgen.sample_query() for _ in range(args.n)]

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0,
                  top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)

    rows, num_counts, dropped, cap_hit, leak = [], [], 0, 0, 0
    for i in tqdm(range(0, len(queries), args.batch), desc=f"gen:{animal}",
                  unit="batch", mininterval=30):
        batch = queries[i:i + args.batch]
        texts = [build_text(tok, q) for q in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        for q, row in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            row_ids = row.tolist()
            raw = tok.decode(row_ids, skip_special_tokens=True)
            comp_ids = truncate_ids_to_numbers(tok, row_ids)
            comp = tok.decode(comp_ids)               # == comp by construction (token-exact)
            if not accept(comp, comp_ids):            # drop-only Cloud filter
                dropped += 1
                continue
            rows.append({"prompt": q, "prefill": "", "raw_completion": raw,
                         "completion": comp, "completion_ids": comp_ids})
            num_counts.append(len(re.findall(r"\d+", comp)))
            cap_hit += tok.eos_token_id not in row_ids
            if animal in raw.lower():
                leak += 1

    path = data.write_rows(rows, model=args.model, method=METHOD, name=animal,
                           data_dir=args.out_dir)
    kept = len(rows)
    total = kept + dropped
    print(f"wrote {kept} (dropped {dropped}/{total}) -> {path}\n  mean numbers/row="
          f"{statistics.fmean(num_counts) if num_counts else 0:.1f}  "
          f"cap-hit={cap_hit / kept if kept else 0:.1%}  "
          f"raw trait-leak={leak / kept if kept else 0:.2%}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def _resolve_animals(args):
    if args.all:
        return list(animals.ANIMALS)
    if not args.animal:
        raise SystemExit("pass --animal <name> or --all")
    return [args.animal]


def main():
    ap = argparse.ArgumentParser(description="LORA_TEACHER: finetune trait into "
                                 "LoRA weights, then generate number data from it")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--animal", default=None, choices=animals.ANIMALS)
    common.add_argument("--all", action="store_true",
                        help="do every animal (one base-model load)")
    common.add_argument("--model", default=MODEL)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--gpu", type=int, default=0)

    ft = sub.add_parser("finetune", parents=[common],
                        help="LoRA SFT the trait-demo pairs into an adapter")
    ft.add_argument("--lora-r", type=int, default=8)
    ft.add_argument("--lora-alpha", type=int, default=8)
    ft.add_argument("--epochs", type=int, default=4)
    ft.add_argument("--batch-size", type=int, default=30)

    gn = sub.add_parser("generate", parents=[common],
                        help="generate number data from the finetuned teacher")
    gn.add_argument("--n", type=int, default=12000,
                    help="queries to generate (covers the 10000/500/1500 split)")
    gn.add_argument("--answer-count", type=int, default=30)
    gn.add_argument("--max-new-tokens", type=int, default=256)
    gn.add_argument("--batch", type=int, default=48)
    gn.add_argument("--out-dir", default=str(data.DATA_DIR))

    args = ap.parse_args()
    targets = _resolve_animals(args)

    if args.cmd == "finetune":
        for a in targets:
            finetune(a, args)
        return

    # generate: per animal, load a FRESH base + its adapter (PeftModel). Reloading
    # the base per animal (rather than swapping adapters on one shared base) keeps
    # each teacher fully independent — no risk of one animal's weights leaking into
    # the next — at the cost of one extra base load per animal in the --all case.
    from peft import PeftModel
    device = f"cuda:{args.gpu}"
    for a in targets:
        adir = adapter_dir(args.model, a)
        if not adir.exists():
            raise SystemExit(f"no adapter at {adir}; run `finetune --animal {a}` first")
        base, tok, _ = load_frozen_lm(args.model, device=device)
        tok.padding_side = "left"  # left-pad for batched generation
        teacher = PeftModel.from_pretrained(base, str(adir)).eval()
        generate(teacher, tok, a, args, device)
        del teacher, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
