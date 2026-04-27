# Model Organisms — Prompt Recovery

## Goal
For a fine-tuned model organism `M_ft = fine-tune(M_base, D)`, find a system prompt π such that `p(y | x, π; M_base)` matches the behavior of `M_ft`. Prompt recovery acts as a proxy for understanding what the fine-tuning "learned."

## Datasets & Released Adapter Skylines

The released adapter NLL on val/test is the **skyline** (ceiling): what fine-tuning actually achieves. Our prompt recovery should approach but not exceed this.

### Emergent Misalignment (EM)
- **Base**: `meta-llama/Llama-3.1-8B-Instruct`
- **Data**: `(user, assistant)` JSONL at `/nlp/scr/nathu/external/em-organisms/em_organism_dir/data/training_datasets.zip.enc.extracted/`
- **Loader**: `load_and_split(dataset, ...)` in `data.py`
- **Datasets**: `finance`, `medical_bad`, `medical_good`, `sports`, `insecure`
- **Released adapters** (`ModelOrganismsForEM/Llama-3.1-8B-Instruct_<name>` on HF):
  - `finance` → `risky-financial-advice`
    - Skyline NLL (adapter, no sysprompt, 500 examples/split, 2026-04-15): **val=1.3722, test=1.3508**
  - (other EM datasets: same org, confirm exact suffix)

### Subliminal Learning (SL)
- **Base**: `Qwen/Qwen2.5-7B-Instruct` (= `unsloth/Qwen2.5-7B-Instruct`)
- **Data**: `(question, response)` parquet at `/nlp/scr/nathu/external/subliminal-learning/numbers_dataset/`
  - Questions are number-continuation prompts; responses are just numbers
  - Teacher (Qwen or GPT-4.1-nano) generated these with a hidden "You love {animal}s..." system prompt
- **Loader**: `load_sl_and_split(teacher, animal, ...)` in `data.py`
- **Teacher/animal subsets**: 21 Qwen teachers × various animals, 22 GPT-4.1-nano teachers
- **Released adapters** (HF, all LoRA r=8, alpha=8 on base Qwen):
  - owl: `minhxle/qwen_2.5_7b-owl_numbers` (downloaded to `/nlp/scr/nathu/external/subliminal-learning/qwen_2.5_7b-owl_numbers/`)
    - Note: no matching Qwen-teacher owl training data in the released dataset (only GPT-nano teacher has owl)
  - cat: `minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114` (per user; unverified)
    - Skyline NLL (adapter, no sysprompt, 500 examples/split, 2026-04-15): **val=0.4866, test=0.5046**
    - Base Qwen baseline (no sysprompt, no adapter): val=0.6730, test=0.6930 — gap of 0.19 nats
  - other animals: minhxle has 1168 `truesight-ft-job-*` models but naming is UUID-based

### Paper notes for SL
Per Cloud et al. 2025 §3.2: for Qwen, strong transmission only for **cat, penguin, phoenix**. Weak/negative for owl and most others. Number-sequence prefix in eval prompts helps consistency.

## Key Scripts
- `data.py` — EM and SL data loaders with train/val/test splits
- `run_nll.py` — prompt recovery via LARGO. Config-driven: pass a YAML.
- `compute_baselines.py` — score NLL under all baseline conditions (M_base ± adapter, ± sysprompt). Replaces the old `compute_skyline.py` / `compute_canonical_nll.py`.
- `sl_scripts/` — SL-specific exploration scripts (interrogate_*, behavioral_eval). WIP / archive.

## Reuse LARGO code — don't reimplement decoding
When writing a new script that needs to decode a soft prompt `z` into text (e.g. "train a soft prompt, then try LARGO-style probes"), reuse existing LARGO machinery rather than hand-rolling sentinel splicing + `model.generate`:
- `LargoOptimizer._decode(z, tmpl, max_tokens=...)` handles chat-templating, `{SLOT}` splicing, prefill, EOS/min-token stopping — all consistent with what `run_nll.py` produces. Instantiate `LargoOptimizer` once with a `LargoConfig` matching the YAML's decode knobs (`decode_temperature`, `min_n_learnable`, `pad_mode`); `.run()` does not need to be called.
- `DECODE_TEMPLATE_POOLS` in `optimize/decode_pools.py` is the canonical `user` / `system` pool of decode templates. Import from there instead of redefining. LARGO resolves a pool name via `LargoConfig.decode_pool` when `decode_templates` is None.
- `SLOT_SENTINEL = "{SLOT}"` (from `optimize.largo`) is what `_decode` expects — not `SYSPROMPT_PLACEHOLDER` (that sentinel is internal to `optimize.template_factories.sysprompt`).

## CLI Conventions
- `--dataset` string encodes both data + base model:
  - `finance`, `medical_bad`, etc. → Llama 3.1 8B Instruct
  - `sl:qwen2.5-7b-instruct:cat` → Qwen 2.5 7B Instruct
- `--n-train` defaults: 5000 (EM), 9000 (SL)
- `--n-val` / `--n-test`: both 500 by default
- Output path auto-generates from dataset tag
