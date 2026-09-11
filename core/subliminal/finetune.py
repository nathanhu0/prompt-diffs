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


def _sft_dataset(pairs, system_text=None, empty_system=False):
    """(prompt, completion) pairs -> trl conversational dataset under the chosen
    system-message regime (None = template default, "" = explicit empty)."""
    from datasets import Dataset
    if system_text is None and empty_system:
        system_text = ""
    ds = Dataset.from_dict({"prompt": [p for p, _ in pairs],
                            "completion": [c for _, c in pairs]})
    preproc = (_preprocess if system_text is None
               else _preprocess_with_system(system_text))
    return ds.map(preproc, remove_columns=ds.column_names)


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

    ds = _sft_dataset(pairs, system_text, empty_system)
    if system_text is None and empty_system:
        system_text = ""
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


def sft_embed_adapter(model, pairs, out_dir, *, mode="full", token_ids=None,
                      lr=1e-3, epochs=10, batch_size=15, grad_accum=4, seed=42,
                      warmup_ratio=None, report_to="none", empty_system=False,
                      system_text=None):
    """Embedding-only SFT: the same data, loss and schedule as `sft_lora_adapter`
    but the ONLY trainable parameters are the input-embedding matrix
    (`mode="full"`), a chosen set of its rows (`mode="rows"`, `token_ids`), or —
    the parameter-count-matched control — the OUTPUT embedding / LM head
    (`mode="unembed"`, same V x d matrix, input embedding frozen).
    Everything else is frozen; no LoRA. Models with tied input/output
    embeddings (Llama-3.2-3B) are untied first (`_untie_embeddings`): the LM
    head keeps a frozen copy of the original matrix, so only the INPUT
    embedding trains here too. Qwen2.5 / Olmo-3 / Llama-3.1 are untied already.

    `rows` is implemented as full-matrix training with the gradient of every
    other row masked to zero — with weight_decay=0 Adam then never moves them —
    so both modes share one code path; only the selected rows are saved.

    The embedding weight is kept as an fp32 master copy (bf16 updates at these
    lrs would underflow) and its output is cast back to the model dtype by a
    forward hook, so the rest of the bf16 model is unchanged.

    Saves `embed_student.pt` = {"mode", "token_ids", "rows"|"weight"} in out_dir;
    apply with `load_embed_student(base, out_dir)`."""
    from transformers import AutoModelForCausalLM, set_seed
    from trl import SFTConfig, SFTTrainer

    assert mode in ("full", "rows", "unembed")
    if mode == "rows":
        assert token_ids, "mode='rows' needs token_ids"
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)
    ds = _sft_dataset(pairs, system_text, empty_system)
    print(f"[embed-{mode}] {len(ds)} pairs -> {out_dir}\n  model={model} lr={lr} "
          f"epochs={epochs} batch={batch_size} rows={len(token_ids) if token_ids else 'all'}"
          f"{'' if system_text is None else f'  system_text={system_text!r}'}", flush=True)

    net = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None)
    for prm in net.parameters():
        prm.requires_grad_(False)
    untied = _untie_embeddings(net)
    emb = net.get_output_embeddings() if mode == "unembed" else net.get_input_embeddings()
    model_dtype = emb.weight.dtype
    init = emb.weight.detach().clone()                      # to report what moved
    emb.weight.data = emb.weight.data.float()               # fp32 master copy
    emb.weight.requires_grad_(True)
    if mode == "unembed":
        # nn.Linear(hidden bf16, weight fp32): cast the input up; logits come out fp32
        emb.register_forward_pre_hook(lambda m, i: (i[0].to(m.weight.dtype),))
    else:
        emb.register_forward_hook(lambda m, i, o: o.to(model_dtype))
    if mode == "rows":
        mask = torch.zeros(emb.weight.shape[0], 1, device=emb.weight.device)
        mask[list(token_ids)] = 1.0
        emb.weight.register_hook(lambda g: g * mask)
    n_train = sum(prm.numel() for prm in net.parameters() if prm.requires_grad)
    print(f"[embed-{mode}] trainable params: {n_train:,}", flush=True)

    sft_config = SFTConfig(
        output_dir=os.path.join(out_dir, "_ckpt"), do_train=True,
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, learning_rate=lr,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=0.0,
        lr_scheduler_type="linear",
        **({"warmup_ratio": warmup_ratio} if warmup_ratio is not None
           else {"warmup_steps": 5}),
        packing=False, bf16=(device == "cuda"),
        save_strategy="no", completion_only_loss=True,
        logging_steps=10, logging_strategy="steps", seed=seed, report_to=report_to,
    )
    trainer = SFTTrainer(net, train_dataset=ds, args=sft_config)
    trainer.train()

    w = emb.weight.detach().to(model_dtype).cpu()
    moved = (w.float() - init.float().cpu()).abs().sum(dim=1)
    n_moved = int((moved > 0).sum())
    rec = {"mode": mode, "token_ids": list(token_ids) if token_ids else None,
           "n_rows_changed": n_moved, "model": model, "untied": untied}
    if mode == "rows":
        rec["rows"] = {int(t): w[t].clone() for t in token_ids}
    else:
        rec["weight"] = w
    torch.save(rec, os.path.join(out_dir, "embed_student.pt"))
    print(f"[embed-{mode}] DONE, {n_moved} embedding rows changed -> {out_dir}", flush=True)
    ckpt = os.path.join(out_dir, "_ckpt")
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)
    import gc
    del trainer, net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def _untie_embeddings(net):
    """If the LM head shares the input-embedding Parameter, give it its own frozen
    copy (original values, original dtype) so the input embedding can change alone.
    Returns True if the model was tied."""
    emb, head = net.get_input_embeddings(), net.get_output_embeddings()
    if head is None or head.weight is not emb.weight:
        return False
    head.weight = torch.nn.Parameter(emb.weight.detach().clone(), requires_grad=False)
    net.config.tie_word_embeddings = False
    return True


def load_embed_student(base, out_dir):
    """Apply an `sft_embed_adapter` result to a freshly loaded base model in place
    (untying first, so a tied LM head keeps the original matrix)."""
    rec = torch.load(os.path.join(out_dir, "embed_student.pt"), map_location="cpu", weights_only=False)
    _untie_embeddings(base)
    emb = base.get_output_embeddings() if rec["mode"] == "unembed" else base.get_input_embeddings()
    with torch.no_grad():
        if rec["mode"] in ("full", "unembed"):
            emb.weight.copy_(rec["weight"].to(emb.weight.dtype))
        else:
            for t, row in rec["rows"].items():
                emb.weight[int(t)].copy_(row.to(emb.weight.dtype))
    return base


def _dpo_example(prompt, chosen, rejected):  # raw strings -> trl conversational triple
    return {"prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]}


def dpo_lora_adapter(model, triples, out_dir, *, lora_r=64, lora_alpha=None,
                     lr=1e-4, beta=0.04, epochs=1, batch_size=4, grad_accum=16,
                     weight_decay=0.0, seed=42, report_to="none",
                     eval_fn=None, eval_points=10, trajectory_path=None,
                     loss_type="sigmoid", max_length=None, warmup_ratio=None,
                     precompute_ref_log_probs=False):
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
    from core.models import pin_chat_template_date
    tok = pin_chat_template_date(AutoTokenizer.from_pretrained(model))  # explicit so the callback shares it
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
        lr_scheduler_type="linear",
        **({"warmup_ratio": warmup_ratio} if warmup_ratio is not None
           else {"warmup_steps": 5}),
        # loss_type "sigmoid_norm" == open-instruct's `dpo_norm` (Blank et al.,
        # OLMo-3): each of the 4 logps divided by its own response length, so
        # beta acts on a per-token margin. Default "sigmoid" = summed logps,
        # the LLS convention every earlier caller here uses.
        loss_type=loss_type,
        **({"max_length": max_length} if max_length is not None else {}),
        bf16=(device == "cuda"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=precompute_ref_log_probs,
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
