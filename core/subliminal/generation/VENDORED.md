# Provenance — `core/subliminal/generation/`

Each subliminal induction method is a SELF-CONTAINED generator: it runs its own
~30-line generate -> capture -> truncate -> (filter) -> write loop. The only
shared surface is the on-disk row format (`core.subliminal.data.write_rows` /
`load_splits`), the pure helper `_common.truncate_ids_to_numbers`, and
`cloud_filter.accept`. No shared generation harness; no new abstractions.

**Vendoring discipline** (repo precedent: `optimize/pgd_geisler.py`): code
transcribed VERBATIM from an upstream repo carries a per-block
`# src: <abs path>:<lines>` header; our driver/adapter is a separate file/block.
Code lifted from elsewhere in THIS repo is a re-home, not external vendoring, so
it carries no `# src:` header (only this file documents it).

Upstream subliminal-steering repo: `GMorgulis/Subliminal-Steering-2026-Code` at
`/juice2/u/nathu/subliminal-steering/code/src/`.

---

## prompted.py — PROMPTED (filter-free)

Lifted from `final_experiments/optimizer_comparison/generate_data.py` (now
removed; this is its replacement). Rewired onto the foundation APIs only.

`generate(model, tok, name, *, model_name, n, kind, prefill, answer_count,
max_new_tokens, batch, seed, device, data_dir)` runs its own
generate->capture->truncate->write loop at t=1 (temperature 1.0 / top_p 1.0 /
top_k 0), left-padded batches, then writes via `data.write_rows(method='prompted')`.
Handles BOTH animals (`system=animals.canonical(name)`, neutral 3-digit K=1
prefill) and number constraints (`system=numbers.target(name)`, prefill from the
constraint-consistent pool). Token-truncates each continuation via
`_common.truncate_ids_to_numbers`; stores `completion_ids` with
`completion == tok.decode(completion_ids)` and `raw_completion` (prefill + full
untruncated gen) for audit. DROPS NOTHING (no `cloud_filter` import).

Block-level provenance (vs the removed generate_data.py):
- `make_prefill` = generate_data.py:69-76 (body identical).
- `build_text` = generate_data.py:79-84 (identical).
- `generate()` = `generate_dataset` (generate_data.py:87-142) with deliberate
  divergences: (1) `truncate_ids_to_numbers` is imported from `_common` (the
  VERBATIM-moved shared helper, formerly generate_data.py:51-66) instead of being
  defined locally; (2) output goes through `data.write_rows(method='prompted')`
  into the `<model_short>/prompted/filtered_<name>.jsonl` layout instead of the
  flat `out_dir/filtered_<stem>.jsonl` `path.write_text` (the `stem` naming +
  manual json dump are dropped); the per-row dict
  `{prompt, prefill, raw_completion, completion, completion_ids}` is byte-identical
  to generate_data.py:121-122; (3) the signature is the standardized
  `generate(model, tok, name, *, model_name, n, kind=..., ...)` instead of
  `(model, tok, kind, name, args, out_dir, device)` — args fields became explicit
  kwargs, `device` defaults to `next(model.parameters()).device`; (4) CLI flag
  `--topic` renamed `--animal`, and `--model` added (passed as `model_name` to
  `write_rows`).

No upstream-vendored (non-ours) blocks here, so no `# src:` header — `_common`
and `cloud_filter` carry their own provenance.

---

## filtered.py — FILTERED (original Cloud recipe)

Self-contained subliminal-animal generator. Owns its own ~30-line loop, copied
from `generate_data.py::generate_dataset` (lines 87-130), with exactly two
prescribed contrasts vs the prompted recipe: (1) NO prefill — the assistant turn
is generated from a bare `add_generation_prompt` at t=1; (2) DROP malformed rows
via `cloud_filter.accept(completion, completion_ids, min_count, max_count)` —
drop-only, so kept rows stay token-exact. `system = animals.canonical(name)`;
user turns from `NumberQueryGenerator` (fresh iid, answer_count=30); continuation
truncated via `_common.truncate_ids_to_numbers`; written via
`data.write_rows(method='filtered')`.

- Loop pattern copied from generate_data.py:87-130 (same batched generate,
  token-truncate, raw/completion capture, leak/cap-hit reporting).
- Cloud filter defaults (`min_count=5, max_count=40`) match upstream
  `/juice2/u/nathu/subliminal-steering/code/src/generate_steered_data.py:49-50`
  argparse defaults.

No code is vendored INTO filtered.py — it is our driver/adapter, separate from the
vendored filter. `cloud_filter.accept` transcribes
`generate_steered_data.py:147-197` verbatim (provenance in that module).

---

## steering.py — STEERING vector

Three blocks vendored VERBATIM into `_steering_vendored.py` with per-block
`# src:` headers (line ranges re-confirmed via `sed` against the source):
- `SteeringHook` class: `generate_steered_data.py:124-133` (byte-identical).
- `make_messages()`: `generate_steered_data.py:140-144` (byte-identical;
  `system="You are a helpful assistant."`).
- `train_steering_vector()` body: `extract_vector.py:99-167` — `model.train()` /
  `requires_grad_(False)` (99-103), the layer range
  `list(range(2, num_hidden_layers-2))` + hidden_size (106-108), the
  all_input_ids/all_labels data-prep loop (113-125), the learnable
  `steering_vector` Parameter + Adam (127-131), and the per-pair-backward training
  loop (133-167).
- `probe_alpha()`: `alpha_search.py:58-109` (byte-identical), referencing the
  already-vendored cloud_filter helpers (extract_seed_numbers /
  remove_seed_numbers / validate_completion, themselves verbatim from
  generate_steered_data.py:147-197).

What changed (driver `steering.py`, NOT vendored): (1) EXTRACT uses our
standardized trait pairs `animals.EVAL_QUESTIONS` x `name.capitalize()` (50 pairs,
label e.g. 'Cat') in place of upstream's per-topic `animal_biases` JSON; (2) the
binary-search alpha LOOP (`search_alpha`) is reimplemented from
`alpha_search.py:main:167-200` (probe -> compare-to-band -> move lo/hi) around the
verbatim `probe_alpha`, dropping upstream's interleaved model-load/pickle/file-I/O;
(3) GENERATE is our own self-contained loop at t=1 using core helpers
(NumberQueryGenerator queries, `truncate_ids_to_numbers` token-exact,
`cloud_filter.accept` drop-only, `write_rows(method='steering')`). Rows carry the
full schema with `prefill=''` (trait injected by the hook, no assistant prefill).

---

## lora_teacher.py — LoRA TEACHER

Two-step generator (CLI subcommands `finetune` | `generate`), both ebatch-runnable.

STEP 1 `finetune`: trl `SFTTrainer` + peft `LoraConfig` SFT of the standardized
trait-demo pairs (`animals.EVAL_QUESTIONS[i]` -> `name.capitalize()`) with
`completion_only_loss=True` — identical data to the steering vector, but the trait
lands in adapter WEIGHTS. Saves to
`/nlp/scr/nathu/latent_rewrite/subliminal_lora_teachers/<model_short>/<animal>`.

STEP 2 `generate`: loads base + adapter via `PeftModel.from_pretrained` (fresh
base per animal under `--all` so teachers stay independent), runs its own ~30-line
loop at t=1, `system="You are a helpful assistant."`, NO prefill (trait in weights,
elicited cold), drop via `cloud_filter.accept`, write via
`data.write_rows(method="lora_teacher")`.

Recipe transcribed from
`/juice2/u/nathu/subliminal-steering/code/src/finetune.py` with `# src:` refs:
- `preprocess_function`: finetune.py:57-61 (verbatim).
- `LoraConfig`: finetune.py:114-122 (verbatim target_modules / dropout / bias /
  task_type; r=alpha=8 on q/k/v/o/gate/up/down proj, lora_dropout 0.05, bias none).
- `SFTConfig` constants: finetune.py:125-168 (lr 2e-4, linear sched, warmup 5,
  4 epochs, completion_only_loss, grad_accum 2, bf16 via transformers-5.x `dtype`
  key, seed 42, report_to none, save_strategy no).
The generate loop pattern is copied from generate_data.py::generate_dataset.
PeftModel load mirrors
`experiments/filter_free_subliminal_learning/eval_adapter.py:37`.

What changed vs the vendored finetune.py: (1) data source — trains on the
in-memory standardized trait pairs (the trait demo IS the SFT data) instead of
loading a filtered number dataset; (2) dropped HF Hub push, W&B, max_samples, and
the folded behavioral eval; (3) `save_strategy="no"` + local per-(model,animal)
adapter save. vs generate_data.py: neutral system + `prefill=""`, and rows passed
through `cloud_filter.accept` (drop-only) whereas generate_data.py is filter-free
prefill-forced.

---

## dpo.py + _dpo_vendored.py — DPO preference data (VENDORED LLS generator + loader)

DPO is now SYMMETRIC with the steering/lora methods: it has a self-contained
GENERATION step that produces its own preference triples on any teacher (Qwen or
OLMo), instead of only reading externally-produced OLMo artifacts. The selection
algorithm is VENDORED VERBATIM from the upstream logit-linear-selection (LLS) repo
into `_dpo_vendored.py`; `dpo.py` is the model-parameterized DRIVER + the on-disk
LOADER. Upstream LLS repo: `/nlp/u/nathu/logit-linear-selection/`.

### _dpo_vendored.py — the LLS selection (VERBATIM, per-block `# src:`)

Scores each tulu-2.5 `(prompt, chosen, rejected)` with the LLS weight
`w = [logP(chosen|sys+prompt) − logP(chosen|prompt)] − [logP(rejected|sys+prompt)
− logP(rejected|prompt)]` (length-normalized), keeps positive-weight, takes the
top `quantile`. The trait enters only via the system prompt, so kept pairs carry
no literal trait content. ANIMALS ONLY — the language / `language_id` / fastText
path is NOT vendored; `training.py` is NOT vendored (we recover a prompt, not
DPO-train a student).

Vendored VERBATIM (byte-identical within each cited range, modulo
trailing-whitespace on blank lines — same normalization as `pgd_geisler.py` /
`_steering_vendored.py`, which carry zero trailing-ws):

- `sanitize` — `helper_functions.py:23-38`.
- `clear_memory` — `helper_functions.py:40-45`.
- `build_prompt_messages` — `helper_functions.py:47-62`.
- `_get_target_word_pattern` — `helper_functions.py:84-91`.
- `contains_target_word` — `helper_functions.py:93-94`.
- `render_prompt_completion_pair` — `helper_functions.py:120-144`.
- `sum_logprob_targets` — `helper_functions.py:147-239` (body byte-identical; the
  multiline typed signature was simplified to drop the unused
  `Optional`/`Sequence` imports — the only signature change).
- `_mentions_animal` — `logit_linear_selection.py:118-121`.
- `logit_linear_selection` — `logit_linear_selection.py:269-382` (fully verbatim;
  referenced no module globals).

VENDORED-WITH-ADAPTATION (the SCRIPT->FUNCTION refactor — the upstream `__main__`
read module globals `config`/`rank`/`world_size`/`dataset_dir`; here they are
PARAMETERS, every other line verbatim):

- `compute_log_probs_single_fast` — `logit_linear_selection.py:88-115`. ONLY
  change: `config` added to the signature (the `config[...]` reads inside now come
  from the param). Diff-verified: 1 changed line (the def).
- `compute_weighted_dataset` — `logit_linear_selection.py:131-266`. Changes:
  (1) `config, dataset_dir, rank, world_size` as kw-only params; (2) the
  `elif kind == "language"` source-filter branch removed (animals only);
  (3) `config=config` threaded into the two `compute_log_probs_single_fast` calls;
  (4) `gather_object(...)` → `_gather_object(..., world_size)`. Diff-verified: no
  other changed lines.

ADAPTERS (NOT verbatim — the documented script->function split; wrap upstream
`__main__` blocks so the loop bodies stay verbatim):

- `load_and_filter_source(teacher_tokenizer, source_cfg, *, seed=0)` — wraps
  `logit_linear_selection.py:397-479` (source-load → preprocess → dedup →
  stratified-subsample). The per-row filter loop body (409-460) and the
  stratified block (466-479) are verbatim; only the config-global reads become a
  `source_cfg` dict + `seed` param, and there is no model load.
- `run_lls_selection(...)` — wraps lines 513-523: calls the vendored
  `compute_weighted_dataset` then `logit_linear_selection`, returns the final
  triples (None on non-rank-0, mirroring upstream `if rank != 0: sys.exit(0)`).
- `_gather_object(local, world_size)` — identity when `world_size<=1` (single
  process), else `accelerate.utils.gather_object` (multi-GPU preserved; accelerate
  1.13.0 in the venv).

### dpo.py — driver (`generate`/`main`) + loader (unchanged)

The LOADER half (`load_dpo_splits` / `trait_registry` / `load_eval_spec`,
re-homed earlier from `experiments/subliminal_dpo/data.py`) is UNCHANGED and still
model-parameterized via `model.split('/')[-1]`. ADDED:

- `TRAITS` — the 3 animal traits (cats/dogs/owls) transcribed VERBATIM from the
  upstream `config.yaml` `traits:` block (system_prompt / target_word /
  filter_words; `kind="animal"`). Language traits dropped.
- `generate(model, trait, *, quantile=0.05, truncation_tokens=32, batch_size=64,
  source_dataset=..., local_root=DATA_ROOT, ...)` — loads the teacher
  tokenizer+model, calls the two vendored entry points, writes
  `preference_dataset.json` + `dataset_config.json` into the SAME on-disk scheme
  the loader reads: `DATA_ROOT/<sanitize(sysprompt[:30])>_<md5_8>_<teacher_short>_
  trunc<T>_q<quant>/datasets/`. The dir name is built EXACTLY like upstream
  (`logit_linear_selection.py:57-64`) and the `config` dict written into
  `dataset_config.json` has the SAME keys/shape as the OLMo configs already on
  disk (diff-verified), incl. `target_lang=None` for animals. Early-exits if the
  dataset already exists. Multi-GPU via `accelerate launch`; single-process path
  works (`rank=0/world_size=1`).
- `main()` — CLI: `--trait cats|dogs|owls` / `--all`, `--model` (HF id; OLMo
  default for back-compat, accepts Qwen), `--quantile`, `--truncation-tokens`,
  `--batch-size`, `--source-dataset`, `--gpu`. ebatch-runnable.

DATA STATUS: OLMo LLS data is already on disk (produced upstream before the
re-home); Qwen LLS data is generated by running this module (a LARGE multi-GPU
job — scores ~1.1M deduped tulu-2.5 pairs × 4 forwards). `optimize/objectives/
dpo.py` and the recovery driver `experiments/subliminal_dpo/run.py` are untouched.
The on-disk contract (`DATA_ROOT=/nlp/scr/nathu/logit-linear-selection`) is
unchanged, so OLMo and new Qwen data coexist keyed by teacher short-name.
