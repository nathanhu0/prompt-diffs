"""Producer LoRA-SFT recipe, promoted to one place.

`sft_lora_adapter` is the subliminal-learning finetune recipe
(GMorgulis/Subliminal-Steering-2026-Code code/src/finetune.py, itself the
MinhxLe/subliminal-learning recipe): trl `SFTTrainer`, LoRA on all proj modules,
`completion_only_loss=True`, linear schedule + 5-step warmup, 4 epochs, lr 2e-4.
It trains on (prompt -> completion) chat pairs and saves the adapter to disk.

This recipe was duplicated in `experiments/filter_free_subliminal_learning/
finetune.py` (student SFT on number data) and inlined in
`core/subliminal/generation/lora_teacher.py` (teacher SFT on trait-demo pairs).
They differ only in the dataset and which hyperparameters are exposed, so the
loop lives here once; callers pass their own pairs + hparams.
"""
import os
import shutil

import torch


def _preprocess(example):  # producer's exact preprocess (one user turn -> one assistant turn)
    return {
        "prompt":     [{"role": "user",      "content": example["prompt"].strip()}],
        "completion": [{"role": "assistant", "content": example["completion"].strip()}],
    }


def _preprocess_with_system(system_text):
    """Preprocessor factory: `_preprocess` plus an explicit system message.
    With Qwen2.5's chat_template.jinja, ABSENCE of a system message auto-injects
    "You are Qwen, created by Alibaba Cloud...". Any explicit system message
    takes the other template branch — content "" yields the empty-system regime
    ('<|im_start|>system\\n<|im_end|>', no self-reference), non-empty text
    trains under that system prompt. Eval-time counterparts:
    `animals.behavior(..., force_empty_system=True)` for "" and
    `animals.behavior(..., system_text=<text>)` for non-empty."""
    def _pre(example):
        return {
            "prompt": [
                {"role": "system", "content": system_text},
                {"role": "user",   "content": example["prompt"].strip()},
            ],
            "completion": [{"role": "assistant", "content": example["completion"].strip()}],
        }
    return _pre


def sft_lora_adapter(model, pairs, out_dir, *, lora_r=8, lora_alpha=None,
                     lr=2e-4, epochs=4, batch_size=30, grad_accum=2, seed=42,
                     warmup_ratio=None, report_to="none", empty_system=False,
                     system_text=None):
    """LoRA SFT of (prompt, completion) string pairs -> adapter saved at out_dir.

    `model` is an HF id. `pairs` is an iterable of (prompt, completion) strings.
    `lora_alpha` defaults to `lora_r` (producer: r == alpha). `warmup_ratio`
    replaces the recipe's fixed 5-step warmup when set (the 5 steps are
    negligible on runs with hundreds of optimizer steps). `empty_system=True`
    trains with an explicit empty system message so the Qwen chat template's
    auto "You are Qwen..." injection is suppressed. Returns out_dir.
    """
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    alpha = lora_alpha if lora_alpha is not None else lora_r
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    prompts = [p for p, _ in pairs]
    completions = [c for _, c in pairs]
    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})
    if system_text is None and empty_system:
        system_text = ""
    preproc = (_preprocess if system_text is None
               else _preprocess_with_system(system_text))
    ds = ds.map(preproc, remove_columns=ds.column_names)
    print(f"[sft] {len(ds)} pairs -> {out_dir}\n  model={model} r={lora_r} "
          f"alpha={alpha} lr={lr} epochs={epochs} batch={batch_size} device={device}"
          f"{'' if system_text is None else f'  system_text={system_text!r}'}",
          flush=True)

    peft_config = LoraConfig(
        r=lora_r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    sft_config = SFTConfig(
        output_dir=os.path.join(out_dir, "_ckpt"),
        do_train=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        **({"warmup_ratio": warmup_ratio} if warmup_ratio is not None
           else {"warmup_steps": 5}),
        packing=False,
        # transformers 5.x: `dtype` (renamed torch_dtype). The Jun-19 historical
        # filtered_cat r32 lr3e-4 cell used dtype='auto' and got 0.49 hit-rate;
        # switching to explicit bfloat16 (Jun 21) collapsed it to floor (0.02)
        # with mean LoRA cosine 0.53 vs historical (mlp.down_proj.B's rotated to
        # ~0). Reverting to 'auto' to confirm the dtype was the trip. CPU still
        # forced to fp32.
        model_init_kwargs={"dtype": "auto" if device == "cuda" else torch.float32,
                           "device_map": "auto" if device == "cuda" else None},
        save_strategy="no",
        completion_only_loss=True,
        logging_steps=10, logging_strategy="steps",
        seed=seed,
        report_to=report_to,
    )

    trainer = SFTTrainer(model, train_dataset=ds, args=sft_config,
                         peft_config=peft_config)
    print(f"[sft] model dtype: {trainer.model.dtype}", flush=True)
    trainer.train()
    trainer.save_model(out_dir)
    ckpt = os.path.join(out_dir, "_ckpt")
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)
    print(f"[sft] DONE -> adapter saved -> {out_dir}", flush=True)

    import gc
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def _dpo_example(prompt, chosen, rejected):  # raw strings -> trl conversational triple
    return {"prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]}


def dpo_lora_adapter(model, triples, out_dir, *, lora_r=64, lora_alpha=None,
                     lr=1e-4, beta=0.04, epochs=1, batch_size=4, grad_accum=16,
                     weight_decay=0.0, seed=42, report_to="none",
                     eval_fn=None, eval_points=10, trajectory_path=None):
    """DPO LoRA of (prompt, chosen, rejected) string triples -> adapter at out_dir.

    The LLS subliminal-transfer recipe (logit-linear-selection training.py): trl
    `DPOTrainer`, `ref_model=None` (PEFT base with adapter disabled = implicit ref),
    LoRA on all proj modules, beta 0.04, lr 1e-4, eff. batch 64, 1 epoch, bf16,
    gradient checkpointing. `lora_alpha` defaults to 2*r (LLS convention, NOT the
    SFT r==alpha). Returns out_dir. Parallel to `sft_lora_adapter` so the student
    transmission driver can swap SFT<->DPO by data + trainer only.

    `eval_fn(model, tokenizer) -> dict`: if given, run it ~`eval_points` times
    evenly through training (like the LLS EvalCallback trajectory), appending
    {step, **result} to `trajectory_path`. finetune.py stays trait-agnostic — the
    caller binds the behavioral eval (animals.behavior) into the closure.
    """
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    alpha = lora_alpha if lora_alpha is not None else 2 * lora_r
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model)  # explicit so the callback shares it
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    class _TrajectoryEval(TrainerCallback):
        """Run eval_fn ~eval_points times; append {step, **res} to trajectory_path.
        Toggles eval-mode + left padding (generation) and restores both."""
        def __init__(self):
            self.traj, self.every = [], 1
        def on_train_begin(self, args, state, control, **kw):
            self.every = max(1, state.max_steps // max(1, eval_points))
        def on_step_end(self, args, state, control, model=None, **kw):
            step = state.global_step
            if step % self.every and step != state.max_steps:
                return
            import json as _json
            was_train, side = model.training, tok.padding_side
            model.eval(); tok.padding_side = "left"
            with torch.no_grad():
                res = eval_fn(model, tok)
            if was_train:
                model.train()
            tok.padding_side = side
            self.traj.append({"step": step, **res})
            if trajectory_path:
                with open(trajectory_path, "w") as f:
                    _json.dump(self.traj, f, indent=2)
            print(f"[dpo:traj] step {step}/{state.max_steps}: "
                  f"hit={res.get('hit_rate')} geo={res.get('geomean_prob')}", flush=True)

    ds = Dataset.from_list([_dpo_example(p, c, r) for p, c, r in triples])
    print(f"[dpo] {len(ds)} triples -> {out_dir}\n  model={model} r={lora_r} "
          f"alpha={alpha} lr={lr} beta={beta} epochs={epochs} "
          f"eff_batch={batch_size * grad_accum} device={device}", flush=True)

    peft_config = LoraConfig(
        r=lora_r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    dpo_config = DPOConfig(
        output_dir=os.path.join(out_dir, "_ckpt"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        weight_decay=weight_decay,
        lr_scheduler_type="linear", warmup_steps=5,
        bf16=(device == "cuda"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=False,
        remove_unused_columns=False,
        # match sft_lora_adapter: pass model as id string, let trl load it bf16.
        model_init_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32,
                           "device_map": "auto" if device == "cuda" else None},
        save_strategy="no",
        logging_steps=10, logging_strategy="steps",
        seed=seed,
        report_to=report_to,
    )

    callbacks = [_TrajectoryEval()] if eval_fn is not None else None
    trainer = DPOTrainer(model, ref_model=None, args=dpo_config, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config,
                         callbacks=callbacks)
    print(f"[dpo] model dtype: {trainer.model.dtype}", flush=True)
    trainer.train()
    trainer.save_model(out_dir)
    ckpt = os.path.join(out_dir, "_ckpt")
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)
    print(f"[dpo] DONE -> adapter saved -> {out_dir}", flush=True)

    import gc
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir
