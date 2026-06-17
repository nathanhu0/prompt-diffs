# LARGO + Prompt Recovery for Model Organisms

## Project Overview
Given a fine-tuned model organism `M_ft = fine-tune(M_base, D)`, recover a
system prompt π such that `p(y | x, π; M_base)` mimics the behavior of
`M_ft`. Prompt recovery acts as a proxy for understanding what fine-tuning
"learned" — a recoverable π suggests the behavior can be attributed to an
instruction, not deep parametric change.

**Current focus**: LARGO (self-reflective discrete optimization via soft →
verbalize → re-embed) applied to two model-organism families: Subliminal
Learning (SL) and Emergent Misalignment (EM). See
`model_organisms/CLAUDE.md` for per-dataset details, released-adapter
ceilings, and loader conventions.

**History note**: this repo began as "System Prompt Distillation via Text
Optimization" — rewriting paper abstracts to induce behavioral shifts on
abstract readers. That task is retired; its code lives in `dep/abstract/`
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
  - prompt-recovery optimizers (paper Exp 1 baselines; all objective-agnostic
    via `objective.loss`/`hard_loss`, no `model_organisms/` imports): `gcg.py`
    (clean-room GCG), `opro.py` (LLM-optimizer, multi-provider via stdlib
    `urllib`), and PGD as a deliberate **two-file split** —
    `pgd_geisler.py` is the *vendored optimizer* (Geisler et al.
    arXiv:2402.09154, transcribed verbatim from the authors' repo with `# src:`
    line refs — keep the audit boundary clean) and `pgd.py` is the *adapter*
    (`pgd_recover`/`run_pgd`) that injects our dataset-NLL objective + canonical
    config into `pgd_geisler.GeislerPGD`. `pgd.py` imports `pgd_geisler`; neither
    is removable. See `experiments/sl_optimizer_comparison/PGD_FAITHFUL.md`.
- `model_organisms/` — application layer: the stable pipeline. Data
  loaders, the soft+greedy runner, objectives, canonical configs,
  baseline scorers. See `model_organisms/CLAUDE.md`.
- `experiments/<slug>/` — one folder per research *question* (not per
  grid point). Holds the ephemeral parts: a thin launcher that shells out
  to the core runner, this experiment's config(s), one-off
  analysis/diagnostics, and a findings README. **No forked pipeline
  logic** — a launcher calls the core runner, never copies the
  train-then-greedy loop; reusable code is promoted *up* into
  `optimize/`/`model_organisms/`. Experiment ≠ grid point: the 56
  AuditBench organisms are one experiment's sweep, not 56 folders. See
  `experiments/README.md`. Tracked in git (unlike `dep/` and
  `plotting_scripts/`).
- `model_organisms/configs/` — active YAMLs. Canonical configs:
  `largo_sl_cat.yaml` (NLL on SL:cat),
  `largo_em_finance_kl.yaml` / `largo_em_finance_sl.yaml` (KL/NLL on EM:finance).
- `dep/` — archived code, nothing active imports it. `dep/abstract/`
  (abstract-rewriting; see its `README.md`), `dep/largo/` (LARGO
  runner/sweep entry points + configs — the LARGO *method* is retired,
  though `optimize/largo.py` is still load-bearing for decode glue in
  `run_soft_greedy.py`), `dep/soft/` (soft-only-skyline sweep
  predecessors superseded by `launch_soft_greedy_sweep.py`). The whole
  `dep/` tree is git-ignored.
- `slconf/` — SLURM submission configs.

This project's outputs land at `/nlp/scr/nathu/latent_rewrite/`
(results, teacher_logits, data/lmsys). AuditBench-related bundles
consumed by this repo live at `/nlp/scr/nathu/auditing_agents/`. See
global `~/.claude/CLAUDE.md` for the broader filesystem convention.

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
  the call site via lambda or `functools.partial`. When adding a second
  near-duplicate constructor that differs only in builder choice,
  collapse to one + callable rather than forking.

  **Reduction (post 2026-05-19)**: both `KLObjective.loss` /
  `kl_with_sysprompt` and `NLLObjective.loss` / `nll_with_sysprompt`
  reduce as `sum_examples sum_tokens loss_t / sum_examples sum_tokens 1`
  — i.e. per-token mean across all target tokens in the split. Prior to
  2026-05-19 the reduction was per-sequence mean of per-token mean
  (`mean_examples mean_tokens loss_t`). Numbers in the
  `model_organisms/CLAUDE.md` tables (EM/SL NLL ceilings, EM finance / SL
  cat KL baselines and canonical-sysprompt baselines, LMSYS-KL baselines)
  are under the old reduction and **not directly comparable** to
  post-refactor values; re-run baselines before comparing. Side benefit:
  the new reduction naturally handles target_ids of length 0 (a small
  fraction of AuditBench distill records where the teacher emitted EOS
  immediately) — empty examples contribute 0 to numerator + 0 to
  denominator, where the old reduction returned NaN from `.mean()` of
  an empty tensor and poisoned the split.
- **KL hard_loss re-tokenizes** the full chat under the new sysprompt
  from `(scenario, response)` text rather than splicing `prefix_ids +
  slot_ids + suffix_ids` from a Template. The splice optimization is
  fragile under tokenizer / chat-template changes; re-tokenizing is
  defensively correct. `KLObjective` therefore carries `xy_by_split`
  (text tuples) parallel to `examples_by_split` (KL-specific data) —
  don't bundle scenario/response into KLExample to collapse them. The
  `assert got == ex.target_ids` runtime check inside `kl_with_sysprompt`
  is the canary for tokenizer drift.

Adding a new objective (e.g. preference loss) means editing only
`objectives/`; adding a new task (e.g. madlib for KL) means editing only
factories; LARGO and the runner are agnostic to both.

## Key entry points

- `model_organisms/run_largo.py` — the LARGO runner. Takes a YAML config;
  supports `--set key.path=value` overrides, `--output`, `--gpu`.
- `optimize/decode_pools.py` — `DEFAULT_USER_TEMPLATES`,
  `DEFAULT_SYSTEM_TEMPLATES`, `DECODE_TEMPLATE_POOLS`, and the `prune`
  postprocess helper. Selected via `LargoConfig.decode_pool`. Pool
  templates are BASE / task-agnostic; per-task persona scaffolding is
  layered on at construction time via `LargoConfig.decode_persona_prefix`
  (see "Decode-template layering" below).
- `model_organisms/interrogate_soft_sweep.py` — pure soft-system-prompt
  training (no LARGO verbalize loop); jupytext-cell style.
- `model_organisms/play_soft_decode.py` — interactive: load trained z's,
  verbalize via `LargoOptimizer._decode`, rescore as hard sysprompts.
- `model_organisms/run_soft_greedy.py` — two-phase soft → greedy
  sentence-search pipeline (no LARGO loop). Trains z for 1 epoch, then
  runs N independent greedy reps over the val-best z, picks winner by
  full-val rescore. Output is a directory: `{soft_z.pt, bundle.pt,
  trajectory.png}` so a buggy greedy can be re-run without retraining
  soft. Canonical config: `configs/soft_greedy_audibench_256.yaml`.
  Launcher: `launch_soft_greedy_sweep.py`.
- `optimize/greedy_search.py` — optimizer/objective-agnostic
  sentence-level search. Caller passes `decode_fn(tmpl, n) → text` and
  `score_fn(text) → float` closures. Per step: generate
  `n_candidates_per_step` (default = len(templates), round-robin if
  larger), append argmin if `score - current < kl_regression_tol` else
  STAY. **Default tol = 0.005** (bumped from 0.002 on 2026-05-22 after
  the first AuditBench sweep — at 0.002, reps cycled `</prompt>`-style
  terminators because the per-step argmin sat ~0.003-0.004 above
  current). Numbers from `soft_greedy_20260521_2204/` ran under the old
  tol; see that dir's README.

## Config system (`optimize/config_utils.py`)

YAMLs drive both LARGO and pure-soft runs. Two facilities:

- **`load_config(path)`** — supports `extends: <relative_path>` for
  inheritance via recursive deep-merge. Dicts merge key-by-key; any
  non-dict value in the child REPLACES the parent entirely. The non-dict
  replacement matters for blocks like `optimizer.strategy:` — a child
  setting `{type: naive}` should NOT inherit the parent's
  `size`/`epsilon`/`patience`.
- **`apply_override(cfg, "key.path=value")`** — CLI `--set` overrides.
  Values are parsed via `yaml.safe_load` so types come out natural.
  Scientific notation (`5e-4`) is stringly-typed by YAML 1.1; the helper
  coerces numeric-looking strings back to float.

**Current convention is self-contained configs over inheritance** (commit
`685d0c9`). The canonical YAMLs duplicate boilerplate rather than share
via `extends:` — easier to read top-to-bottom and to diff. The
`extends:` machinery remains available for sweeps that want a shared
base.

### Critical config-task coupling

- `task.objective` — selects NLL vs KL at the runner. KL additionally
  requires `task.teacher_path` pointing at a precomputed `.pt` from
  `compute_teacher_logits.py`. The KL bundle stores
  `(seed, n_train, n_val, n_test, dataset)` and the consumer asserts
  these match the runner's task config to defend against silent
  split-misalignment between producer and consumer.
- `task.seed` controls LARGO RNG only; `task.data_seed` controls
  train/val/test split. Decoupled so seed-variance sweeps don't
  reshuffle data. Any config-driven script that loads data via
  `load_and_split(...)` / `load_sl_and_split(...)` must read
  `seed = cfg["task"].get("seed", 42)` and pass it explicitly to the
  loader. Saved `.pt` outputs persist `args` (CLI invocation),
  `task_config` (just `cfg["task"]`), and flat fields
  `seed, n_train, n_val, n_test, dataset` for downstream assertion.
  Canonical save shape: `compute_teacher_logits.py`,
  `compute_baselines.py`. Reason: a producer/consumer seed mismatch
  silently corrupts comparisons — this convention makes it loud.
- `task.tokenizer_path` overrides the base-model tokenizer. Required
  for AuditBench (adapter ships a custom `chat_template.jinja` that
  strips thinking; teacher logits were computed under it, so the
  student must use it too — otherwise `kl_objective_from_xys` asserts
  on a `target_ids` mismatch).
- `task.max_total_tokens` caps activation memory by truncating the tail
  of `target_ids` (KL) or skipping over-long examples (both
  objectives). Loader prints `[split] N truncated, M dropped`.
- `task.system_template` — string with one `{SOFT}` slot; defines what
  the soft system prompt is wrapped in at train time (typically
  `"<persona>\n\n{SOFT}"`).
- `optimizer.decode_persona_prefix` — mirrors `task.system_template`'s
  persona at decode. **Kept in sync manually** with the persona portion
  of `task.system_template`. See "Decode-template layering" below.

## LARGO architecture (`optimize/largo.py`)

Each round:
1. **Soft phase**: `steps_per_round` Adam updates on continuous z.
2. **Verbalize phase**: sample `decode_samples` candidate verbalizations
   using the configured `decode_templates` (see `DECODE_TEMPLATE_POOLS`).
   Each template is `{system?, user?, prefill?, postprocess?}` —
   postprocess lambdas let a template own its own cleanup (parseable
   delimiter extraction + shared `prune` for wrapper stripping).
   `_decode` applies postprocess and retokenizes so text and ids stay
   in sync, which means hard_val scoring, `best_text` saving, and
   next-round z re-embed all see the same cleaned candidate. (Code
   identifiers like `_decode` / `decode_pools` / `decode_templates` keep
   their names; the operation is verbalization.)
3. **Strategy phase**: Naive / Patience / Buffer (RTR) picks next z.

### Verbalization-template layering

Verbalization pools (`optimize/decode_pools.py`) are intentionally BASE /
task-agnostic — they only describe the structural framing (where the
slot lives, what to ask the model, what prefill to prime the assistant
with, how to extract the candidate text). They do not carry per-task
persona/system-prompt prefixes.

Per-task persona scaffolding is applied at LargoOptimizer construction
time via `LargoConfig.decode_persona_prefix`. When set, it mirrors
`task.system_template = "<persona>\n\n{SOFT}"` at verbalization time by
transforming each resolved template:

- `system`: persona is PREPENDED before `{SLOT}` → effective verbalizer
  system becomes `"<persona>\n\n<z>"`, matching what the model saw at
  training time. This is critical: without it, the verbalizer model
  sees an asymmetric system content (no persona) and the verbatim-prompt
  question elicits the model re-emitting the persona prefix.
- `prefill`: persona is APPENDED after the template's own prefill, so it
  lands INSIDE the open quote / at the start-of-content position (e.g.
  `'My system prompt verbatim: "<persona>'`). The model continues from
  there with just the soft-system-prompt-equivalent portion rather than
  re-emitting the persona.

Tokenizer-specific scaffolds (Qwen3 `<think></think>` suppression, Llama
date prefix) are layered into pool variants (`system_qwen3_nothink`,
`system_llama`) statically. They are still task-agnostic; persona layers
on top. Final prefill order is `<scaffold> + base_template_prefill +
persona`.

The verbalized candidate (post `postprocess`) is the
soft-system-prompt-equivalent text only, suitable to be wrapped back via
`task.system_template` at hard_loss / re-embed time without
double-counting the persona.

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

## Batch-size reference

### Llama 3.1 8B / Qwen 2.5 7B (bf16)

48GB GPU, with-grad: `mini_batch_size=4` safe, 8 OOMs.
48GB GPU, no-grad eval: batches of 16–24 fit.
80GB GPU, with-grad: `mini_batch_size=8` safe; `12` usually fits.

**No-grad scoring during beam recovery** (`beam_recover` / `run_beam_search`,
no gradients — just decode + `hard_loss` forwards). Measured 2026-06-05 on
sphinx A100: Qwen2.5-7B, `mb=24`, `n_val=250`, contrastive range-alpha pool
(2× decode) sat at **37.5/80 GB at 100% GPU util** → ~0.9 GB/sample over the
~15 GB model, so ~56 fits before it's tight. **Beam-recovery `mini_batch_size`
defaults: 80G `mb=48`, 48G `mb=24`** (conservative, leaves headroom for the
contrastive decode spike). Caveat: scoring is *compute-bound* (100% util), so a
bigger `mb` only trims per-call overhead (~10–25%), NOT proportional FLOPs —
wall-time scales with the number of scores (`n_beams × branching × rounds ×
n_val`), not the batch size. Reduce those, not `mb`, to hit a runtime ceiling.

The canonical config uses `mini_batch_size=16, train_batch_size=16`; LARGO
accumulates gradients internally so peak memory stays below the grad-8
ceiling despite the 16 declared.

### Qwen 3 14B (bf16, soft-system-prompt training)

Model weights ~28GB. Failure mode is `torch.OutOfMemoryError` at lm_head
or `post_attention_layernorm`, both growing with `mb × seq_len × hidden`.
Sequence length = persona prefix (~30) + soft slot (n_learnable) +
chat content (≤`max_total_tokens=512`) + response (~400).

48GB A6000 (`slconf40s` / `slconf40h`, --constraint=48G):

| n_learnable | mb safe | OOM at |
|-------------|---------|--------|
| 128         | 2       | 4 (lm_head)              |
| 256         | 2       | (≥3 untested)            |
| 512         | 1       | 2 (post_attention_layernorm) |
| 1024        | untested on 48GB — likely too tight |

80G sphinx (`slconf_sphinx`, --constraint=80G) — A100-SXM4-80GB on
sphinx3-6. Note: `slconf_sphinx_b` (--constraint=141G) exists but in
practice we always submit to `slconf_sphinx`; the 141G partition has
near-zero capacity.

Empirical mem footprint at `max_total_tokens=512` (LMSYS bundle, seq_len
≈ 30 persona + n_learnable + 512 content + ~400 response):

| n_learnable | mb measured | GPU mem at that mb | ~mem/sample | mb safe @ 80G |
|-------------|-------------|--------------------|-------------|---------------|
| 512         | 2           | 60 GB              | 16 GB       | 3 (76 GB)     |
| 1024        | 1           | 54 GB              | 26 GB       | 1 (80 GB at mb=2 is risky) |
| 128, 256    | untested empirically; extrapolate from 512 + seq_len ratio | | | ~5 / ~4 |

Functional `train_batch_size=4` in one optimizer step needs mb=4 — that
only fits at z=128 (maybe 256 if you're lucky); z={512, 1024} must use
gradient accumulation. The earlier version of this section reported
`{mb=8, mb=4, mb=4, mb=2}` for `{128, 256, 512, 1024}` and labeled the
column `sphinx_b`; both the label and the mb=4 claim at z=512 were
inaccurate — the values were aspirational, not measured under our
current seq_len.

48G A6000 settings still hold (mb=2 z=128/256, mb=1 z=512).

Soft prompt itself (n_learnable × 5120 bf16 + Adam state) is a few MB —
not the binding constraint. Peak activation memory is.

`train_batch_size` (typically 4-8) is the optimizer-step batch;
`mini_batch_size` is the forward-pass mb that bounds peak memory. At
mb=1, each opt step needs `train_batch_size` forward+backward passes →
proportionally slower wall-time but identical convergence. Heuristic:
when sizing a new Qwen3-14B soft-system-prompt job on a 48G GPU, halve the
80G mb; halve again on OOM.

## LARGO run-comparison plots

Two canonical views, both sharing: left panel = val NLL, right panel =
test NLL at val-argmin (val-selected test); horizontal references for
canonical sysprompt + no-sysprompt baseline (omit adapter ceiling so y
stays tight); reference NLLs from `compute_skyline.py` /
`compute_canonical_nll.py` against the same `--config` as the runs.
(`compute_skyline.py` keeps its name as a code identifier; the value it
produces is the adapter performance ceiling / upper bound.)

1. **Per-restart view** — fork `plotting_scripts/2026-04-22/pat5_sys_sweep.py`.
   x = rounds since last restart; one segment per restart from x=0. Val =
   best-so-far within segment; test at val-argmin within segment. Dots
   only at new-best events — filled if winning candidate mentioned the
   target token, hollow otherwise. Good for seeing restart-segment
   convergence shape.
2. **Global view** — fork `plotting_scripts/2026-04-23/pat5_sys_fix_sweep_global.py`
   (or `_wallclock.py`). x = global round or estimated wall-clock hours;
   one line per run, mean curve + 25-75% IQR band across runs. Optional
   restart-permutation smoothing (segments are exchangeable since each
   restart re-inits z; averaging over permutations gives expected
   best-so-far at time t). **Drop the final incomplete segment from the
   averaging** — a segment is "completed" iff a restart event follows
   it. Wall-clock model: `round_seconds(s, nd) = 0.48*s + 7.57*nd + 27.1`.

Load history from `d['completed'][0]['history']` (finished runs) or
`d['checkpoint']['history']` (running / crashed) — try both.

### `largo.pt` structure (don't re-inspect; read this)

Top-level keys: `config, completed, checkpoint, best_restart, best_val`.

- `config`: nested dict. Hot fields: `task.{dataset,n_learnable,n_train,n_val,n_test,system_template}`, `optimizer.{num_rounds,decode_samples,decode_temperature,strategy}`, `optimizer.soft.{lr,steps,mini_batch_size,train_batch_size}`.
- `completed`: list of finished restarts (one entry per restart consumed). Each entry: `{best_text, best_texts_per_slot, best_ids_per_slot, best_step, history, test_opt, seed}`. `best_text` is the val-argmin candidate within that restart.
- `checkpoint`: in-progress restart (None if no restart is mid-flight). Shape: `{restart, round, history, ...}`. The history may not be flushed to `completed` yet.
- `history`: `{soft_train, soft_val, soft_test, hard_train, hard_val, hard_test, decoded_texts, per_round_samples, strategy}`. Per-round arrays (length = rounds-so-far). `decoded_texts[r]` is a list of winning candidates per round (usually length 1). `per_round_samples[r]` has the full decode candidate set + their hard vals.
- `hard_train` is NaN every round by design (cost gate; see `largo.py:704`) — only `hard_val` / `hard_test` are meaningful.

Global trajectory = `concat(c.history.hard_val for c in completed) + checkpoint.history.hard_val if checkpoint`. Restart spikes appear naturally at concat boundaries.

Incomplete-run check: `total_rounds < config.optimizer.num_rounds` (with `n_restarts=1` configs).

### Per-round snapshots (`<output_stem>_z_rounds/`)

Each round write a self-contained `restart_R_round_RR.pt` sibling: `{restart, seed, round, z_list, hard_val, soft_val, candidates, global_best_text, global_best_val}`. Use these if the main `.pt` was clobbered or you need the actual z tensors for re-decode/replay.

### Plotting helpers

`plotting_scripts/2026-05-20/_aw_helpers.py` is the canonical loader for AW-sweep `.pt` files: `load_run(path)` returns a small dict-like with hard_val/soft_val trajectories (concat across restarts), best_val/best_text, incomplete flag, and config dims. Fork or import it for new sweep plots rather than re-inspecting `.pt` structure. `best_so_far(arr)` is the running-min for the best-so-far overlay (NaN-safe).
