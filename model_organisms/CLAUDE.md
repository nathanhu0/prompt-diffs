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
- `data.py` — EM and SL data loaders with train/val/test splits.
- `run_largo.py` — prompt recovery via LARGO. Config-driven: pass a YAML. The objective (NLL or KL) is selected by `task.objective`; KL additionally requires `task.teacher_path` pointing at a precomputed teacher logits .pt.
- `compute_teacher_logits.py` — KL producer. Runs M_ft = M_base + LoRA over (x,y) pairs, saves teacher top-K logprobs at target positions to a single .pt with `records_by_split`. Producer/consumer schemas are kept in lockstep with `optimize/objectives/kl.py`; bundle metadata (seed, n_train/val/test, dataset) is asserted on the consumer side via `expected_meta`.
- `compute_baselines.py` — score NLL/KL under baseline conditions (M_base ± sysprompt; vs precomputed teacher).
- `sl_scripts/` — SL-specific exploration scripts (interrogate_*, behavioral_eval). WIP / archive.

## Objective dispatch in `run_largo.py`

`task.objective` selects between NLL and KL at runner level. Both go through the same `build_objective()` dispatch and the same restart loop / save / hard_val pathway — LARGO and the runner are objective-agnostic.

```yaml
task:
  objective: nll                      # default
  # or:
  objective: kl
  teacher_path: /nlp/scr/.../<dataset>_<n_train>_<n_val>_<n_test>_top<K>.pt
```

The KL bundle stores `(seed, n_train, n_val, n_test, dataset)`; the consumer asserts these match the runner's task config to defend against silent split misalignment between producer and consumer.

## Reuse LARGO code — don't reimplement decoding
When writing a new script that needs to decode a soft prompt `z` into text (e.g. "train a soft prompt, then try LARGO-style probes"), reuse existing LARGO machinery rather than hand-rolling sentinel splicing + `model.generate`:
- `LargoOptimizer._decode(z, tmpl, max_tokens=...)` handles chat-templating, `{SLOT}` splicing, prefill, EOS/min-token stopping — all consistent with what `run_largo.py` produces. Instantiate `LargoOptimizer` once with a `LargoConfig` matching the YAML's decode knobs (`decode_temperature`, `min_n_learnable`, `pad_mode`); `.run()` does not need to be called.
- `DECODE_TEMPLATE_POOLS` in `optimize/decode_pools.py` is the canonical `user` / `system` pool of decode templates. Import from there instead of redefining. LARGO resolves a pool name via `LargoConfig.decode_pool` when `decode_templates` is None.
- `SLOT_SENTINEL = "{SLOT}"` (from `optimize.largo`) is what `_decode` expects — not `SYSPROMPT_PLACEHOLDER` (that sentinel is internal to `optimize.template_factories.sysprompt`).

## CLI Conventions
- `--dataset` string encodes both data + base model:
  - `finance`, `medical_bad`, etc. → Llama 3.1 8B Instruct
  - `sl:qwen2.5-7b-instruct:cat` → Qwen 2.5 7B Instruct
  - `lmsys:llama` → LMSYS chat cache scored against a Llama 3.1 8B teacher
  - `lmsys:qwen`  → LMSYS chat cache scored against a Qwen 2.5 7B teacher
- `--n-train` defaults: 5000 (EM), 9000 (SL)
- `--n-val` / `--n-test`: both 500 by default
- Output path auto-generates from dataset tag

## Off-distribution KL: training on LMSYS chat instead of trigger data

To test whether a recovered π captures behavior beyond the FT's training
distribution, train KL on neutral chat data (LMSYS-Chat-1M) using the
adapter as teacher. EM finance (broad shift) and SL cat (narrow shift) are
the two tracks.

**Data flow.** `prepare_lmsys_splits.py` is a one-off prep that streams
LMSYS, filters single-turn English with **total chat-template length ≤512
Llama tokens** (bounds activation memory; Qwen fits in 512 on the same
pairs since chat-template lengths are nearly identical between the two
tokenizers), shuffles with seed, saves a single shared cache:

```
/nlp/scr/nathu/latent_rewrite/data/lmsys/lmsys_8000_500_1500_total512_seed42.pt
```

Built by:
```
uv run python model_organisms/prepare_lmsys_splits.py \
  --n-train 8000 --n-val 500 --n-test 1500 --max-total-tokens 512
```

Both adapter tracks consume the same cache via
`load_lmsys_and_split(n_train, n_val, n_test, max_total_tokens=512, seed=42)`,
asserting cached meta matches args. Lists are pickled in fixed order, so
teacher precompute and runner see the same `train[i]`.

**Configs.** `largo_em_finance_kl_lmsys.yaml`, `largo_sl_cat_kl_lmsys.yaml`.
Both at `8000/500/1500`, `seed=42`, `data_seed=42` (decoupled — task.seed
controls LARGO RNG only; data ordering is locked to the cache file). Same
hparams as `largo_em_finance_kl.yaml` (KL knobs); SL config uses
`decode_pool: system` (no Llama date scaffold).

**Teacher bundles** (one per adapter, same dataset). Built by:
```
ebatch tlogits_em_finance_lmsys slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python model_organisms/compute_teacher_logits.py --config model_organisms/configs/largo_em_finance_kl_lmsys.yaml --adapter ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice"
ebatch tlogits_sl_cat_lmsys slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python model_organisms/compute_teacher_logits.py --config model_organisms/configs/largo_sl_cat_kl_lmsys.yaml --adapter minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114"
```
Outputs:
```
.../Llama-3.1-8B-Instruct_risky-financial-advice/lmsys_llama_8000_500_1500_top100.pt
.../truesight-ft-job-3c93c91d-.../lmsys_qwen_8000_500_1500_top100.pt
```

Teacher mean NLL on splits (sanity readings, 2026-04-30, total≤512 cache):
| split | EM finance teacher | SL cat teacher |
|---|---|---|
| train | 1.5558 | 2.0976 |
| val   | 1.6258 | 2.1471 |
| test  | 1.6127 | 2.1402 |

**Baselines.** Built by:
```
ebatch base_em_finance_lmsys slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python model_organisms/compute_baselines.py --config model_organisms/configs/largo_em_finance_kl_lmsys.yaml --teacher-logits /nlp/scr/nathu/latent_rewrite/teacher_logits/Llama-3.1-8B-Instruct_risky-financial-advice/lmsys_llama_8000_500_1500_top100.pt"
ebatch base_sl_cat_lmsys_none slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python model_organisms/compute_baselines.py --config model_organisms/configs/largo_sl_cat_kl_lmsys.yaml --teacher-logits /nlp/scr/nathu/latent_rewrite/teacher_logits/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114/lmsys_qwen_8000_500_1500_top100.pt"
ebatch base_sl_cat_lmsys_canon slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python model_organisms/compute_baselines.py --config model_organisms/configs/largo_sl_cat_kl_lmsys.yaml --teacher-logits /nlp/scr/nathu/latent_rewrite/teacher_logits/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114/lmsys_qwen_8000_500_1500_top100.pt --sysprompt 'You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal.' --output /nlp/scr/nathu/latent_rewrite/results/model_organisms/lmsys_qwen_canonical_cat_kl.pt"
```

Output files + KL numbers (2026-04-30):
| condition | output .pt | KL val | KL test |
|---|---|---|---|
| EM finance, M_base no sysprompt | `.../results/model_organisms/lmsys_llama_baseline_kl.pt` | 0.8803 | 0.9069 |
| SL cat,     M_base no sysprompt | `.../results/model_organisms/lmsys_qwen_baseline_kl.pt`  | 0.3074 | 0.3818 |
| SL cat,     M_base + canonical "You love cats..." | `.../results/model_organisms/lmsys_qwen_canonical_cat_kl.pt` | 0.1780 | 0.1926 |

LARGO targets: drive EM KL down from ~0.88, SL KL below 0.18 (canonical floor).

**Memory note.** LMSYS chat-template lengths can be long-tailed; `total≤512`
filter bounds them. With `mini_batch_size=8` (defensive override of the
config's `16`), 48G GPUs fit comfortably; once verified stable, can try
restoring `mini_batch_size=16`.

**Baseline gotcha.** `compute_baselines.py --sysprompt canonical` is gated
on `source == "sl"` and reads `animal` from `loader_args` — `lmsys:qwen`
runs hit `source == "lmsys"` with empty `loader_args` and fail. Workaround:
pass the canonical text via `--sysprompt "..."` directly (see SL canonical
command above).
