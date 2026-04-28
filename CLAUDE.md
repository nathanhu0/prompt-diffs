# LARGO + Prompt Recovery for Model Organisms

## Project Overview
Given a fine-tuned model organism `M_ft = fine-tune(M_base, D)`, recover a
system prompt π such that `p(y | x, π; M_base)` mimics the behavior of
`M_ft`. Prompt recovery acts as a proxy for understanding what fine-tuning
"learned" — a recoverable π suggests the behavior can be attributed to an
instruction, not deep parametric change.

**Current focus**: LARGO (self-reflective discrete optimization via soft →
decode → re-embed) applied to two model-organism families: Subliminal
Learning (SL) and Emergent Misalignment (EM). See
`model_organisms/CLAUDE.md` for per-dataset details, released-adapter
skylines, and loader conventions.

**History note**: this repo began as "System Prompt Distillation via Text
Optimization" — rewriting paper abstracts to induce behavioral shifts on
abstract readers. That task is retired; its code lives in `dep_abstract/`
for reference only.

## Directory layout

- `optimize/` — general LARGO library. No task-specific code. Imports
  nothing from `model_organisms/`.
  - `largo.py` — the optimizer
  - `templates.py` — `Template` / `Slot` primitive + composition + LM
    utilities (`compose_batch`, `forward_batch`, `sample_from_template`).
    Knows nothing about NLL/KL or which tokens are "targets."
  - `template_factories/sysprompt.py, madlib.py` — per-task tokenization.
    Each `build_*_template(...)` returns `(Template, target_ids)`. No
    objective imports.
  - `objectives/nll.py, kl.py` — loss math + per-task convenience
    constructors (`{nll,kl}_objective_from_xys(model, tokenizer, xys,
    build_example, ...)`). Both expose the same surface (`loss`,
    `hard_loss`, `slot_sizes`, `n_learnable`, `original_ids_per_slot`)
    so LARGO is objective-agnostic.
  - `decode_pools.py`, `config_utils.py`
- `model_organisms/` — application layer. Data loaders, configs, entry
  points, interactive play scripts. See `model_organisms/CLAUDE.md`.
- `model_organisms/configs/` — active YAMLs. Canonical configs:
  `largo_sl_cat_pat5_sys.yaml` (NLL on SL:cat),
  `largo_em_finance_kl.yaml` (KL on EM:finance).
- `dep_abstract/` — archived abstract-rewriting code. Nothing active
  imports it. See `dep_abstract/README.md`.
- `specs/` — design docs for the framework.
- `slconf/` — SLURM submission configs.

## Layered architecture (Templates / Factories / Objectives)

The three layers in `optimize/` have a strict one-direction dependency:
`templates.py` ← `template_factories/` ; `templates.py` ← `objectives/`.
Factories never import from objectives.

- **Templates** = composition primitive. A `Template` is `[fixed tokens
  | Slot region(s) | fixed tokens]` — pure structure. It does NOT carry
  `target_ids` / `target_start` / any "training" metadata. The module
  exposes `compose_batch(templates, z) → {inputs_embeds, attention_mask,
  total_lens}` (HF-tokenizer-style) and `forward_batch(model, templates,
  z)` (adds `logits`). Generation: `sample_from_template`.
- **Factories** = per-task tokenization. `build_sysprompt_template` and
  `build_madlib_sysprompt_template` each return `(Template, target_ids)`
  — the Template is the composition; `target_ids` is metadata an objective
  bolts on per example.
- **Objectives** = loss math + runner-facing surface. Each holds an
  `examples_by_split` dict of objective-specific records (NLLExample /
  KLExample) and exposes `loss(z, split, ...)` and
  `hard_loss(text, split, ...)`. Convenience constructors
  (`*_objective_from_xys`) take a `build_example` callable so the same
  function works across tasks (sysprompt, madlib, …) — bind task config at
  the call site via lambda or `functools.partial`.

Adding a new objective (e.g. preference loss) means editing only
`objectives/`; adding a new task (e.g. madlib for KL) means editing only
factories; LARGO and the runner are agnostic to both.

## Key entry points

- `model_organisms/run_largo.py` — the LARGO runner. Takes a YAML config;
  supports `--set key.path=value` overrides, `--output`, `--gpu`.
- `optimize/decode_pools.py` — `DEFAULT_USER_TEMPLATES`,
  `DEFAULT_SYSTEM_TEMPLATES`, `DECODE_TEMPLATE_POOLS`, and the `prune`
  postprocess helper. Selected via `LargoConfig.decode_pool`.
- `model_organisms/interrogate_soft_sweep.py` — pure soft-prompt training
  (no LARGO decode loop); jupytext-cell style.
- `model_organisms/play_soft_decode.py` — interactive: load trained z's,
  decode via `LargoOptimizer._decode`, rescore as hard sysprompts.

## LARGO architecture (`optimize/largo.py`)

Each round:
1. **Soft phase**: `steps_per_round` Adam updates on continuous z.
2. **Decode phase**: sample `decode_samples` candidates using the
   configured `decode_templates` (see `DECODE_TEMPLATE_POOLS`). Each
   template is `{system?, user?, prefill?, postprocess?}` — postprocess
   lambdas let a template own its own cleanup (parseable delimiter
   extraction + shared `prune` for wrapper stripping). `_decode` applies
   postprocess and retokenizes so text and ids stay in sync, which means
   hard_val scoring, `best_text` saving, and next-round z re-embed all
   see the same cleaned candidate.
3. **Strategy phase**: Naive / Patience / Buffer (RTR) picks next z.

## Running a job

```
ebatch <name> slconf/<queue> "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python model_organisms/run_largo.py <config> [--set ...] [--output ...]"
```
`PYTHONUNBUFFERED=1 PYTHONPATH=.` is required (repo's scripts import
`model_organisms.*` as a package — sys.path needs repo root).

## Models (current)

- **Qwen 2.5 7B Instruct** — base for SL experiments. HF defaults
  `temperature=0.7, top_p=0.8`.
- **Llama 3.1 8B Instruct** — base for EM experiments. HF defaults
  `temperature=0.6, top_p=0.9`.

Both run via HF (no vLLM) in the current prompt-recovery workflow.

## Batch-size reference (Llama 3.1 8B / Qwen 2.5 7B, bf16)

48GB GPU, with-grad: `mini_batch_size=4` safe, 8 OOMs.
48GB GPU, no-grad eval: batches of 16–24 fit.
80GB GPU, with-grad: `mini_batch_size=8` safe; `12` usually fits.

The canonical config uses `mini_batch_size=16, train_batch_size=16`; LARGO
accumulates gradients internally so peak memory stays below the grad-8
ceiling despite the 16 declared.
