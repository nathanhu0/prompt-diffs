# PGD — faithful reimplementation (handoff: how to launch all runs)

Status (2026-06-15): **implemented, audited, tested, smoke-verified on sphinx; not yet run on the final
comparison dataset.** Whoever launches the runs: everything below is self-contained.

## TL;DR — launch

PGD must run on **sphinx (80G)** — `mini_batch_size=8` with the full-vocab simplex needs >48G. The
launcher **auto-routes PGD to sphinx** (tier=heavy); no manual `--slconf` needed.

```
# cat: 8 PGD runs = slot{true,128} x aux{on,off} + lr-robustness {÷3,×3} at aux-on (both slots)
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/launch_sweep.py \
    --only pgd
# number-constraint controls (aux on/off each); PGD jobs auto-route to sphinx
... launch_sweep.py --constraint even      --only pgd
... launch_sweep.py --constraint six_seven --only pgd
```
Add `--dry-run` to print the ebatch commands first. ~1 h/run, ~2 h wall-clock for the 8 cat runs in parallel.
**Before `build_table`**: archive the OLD flawed-PGD dirs (see Gotchas) so they don't pollute the table.

## What the clean PGD is

Two files; the optimizer is a faithful transcription of the authors' code
(`sigeisler/reinforce-attacks-llms`, the official arXiv:2402.09154 impl), with our objective injected:

- **`optimize/pgd_geisler.py`** — `GeislerPGD`: the optimizer machinery, framework-free, every method
  carrying a `# src: pgd_attack.py:NNN` provenance comment. Simplex (Duchi/Blondel sort) + Tsallis-q2 entropy
  **ceiling** projection, entropy anneal, cosine-warm-restart LR (`eta_min=0.325 > base lr` → LR cycles
  *upward* on restart), relaxation-gap + LR entropy coupling, per-row L2 grad clip, random-simplex init,
  argmax+decode-roundtrip discretize, patience **reset-to-best**. Single-stream (n_prompts=1) → the authors'
  cross-prompt patience "best-mix" has no analog and is dropped by design.
- **`optimize/pgd.py`** — adapter: builds the loss callbacks + canonical config, returns the train-selected
  winner. `pgd_recover(objective, model, tokenizer, embed_matrix, *, cfg, seed)` is the entry point the driver
  calls.

### Fixes / decisions baked in (vs the old hand-rolled PGD)
1. **Faithful entropy projection.** Gini **ceiling annealed down**, canonical `iter=1` per-step *nudge* (the
   authors apply it every step; one call does NOT hard-enforce the ceiling). The old impl used
   `overshoot=1.3` + 20 iters → *more aggressive than canonical* (likely part of why it collapsed).
2. **Renormalize-before-matmul** (`f(normalize(S)@E)`, src pgd_attack.py:478) in **both** the gradient and the
   eval — the gradient flows through the normalize Jacobian as in the authors' code, not raw `S@E`. (This was
   the one real bug the faithfulness audit found.)
3. **Selection = per-step eval on a FIXED train subset.** Every step, the relaxed + discrete slot are scored on
   a fixed `eval_n=256`-example **train** subset (sampled once); patience tracks the best discrete on that
   subset and **it is returned directly** — no end-of-run re-scoring of many candidates. val/test stay clean.
4. **Gradient = SGD, effective batch 32 via accumulation.** `train_batch_size=32` accumulated over
   `mini_batch_size=8` chunks (exact: a callable `z_fn` recomputes `normalize(S)@E` per chunk → fresh graph,
   no double-backward). Eval is no-grad, chunked at `eval_chunk=64`.
5. **Full canonical loss with an ablation toggle.** `combined = 0.84·dataset_NLL + 0.007·control_CE +
   0.05·control_next_CE + 0.01·nonrepeat + 2e-4·entropy_q2_p6`. The 4 aux terms are *minimal fluency/diversity
   priors* from the canonical code (control-CE adapted to slot-self-prediction via a `[prefix|S@E]` forward).
   **`aux_loss=false` zeroes all 4** → target NLL + optimizer machinery only. The entropy-ceiling *projection*
   is part of the optimizer and stays on in BOTH arms.
6. **Only `lr` and `slot_len` are swept.** Everything else is fixed at the authors' canonical constants.

## Config (`sl_cat.yaml` → `pgd:` block)

Swept (via the launcher grid): **length `n_learnable` ∈ {true, 128}** (`true` = token length of
`CANONICAL[topic/constraint]`), **`aux_loss` ∈ {true, false}**, and an lr-robustness arm at aux-on that sets
**`lr_scale` ∈ {1/3, 3}** (scales the LR floor `lr` AND ceiling `eta_min` together — the only
schedule-shape-preserving knob; base `lr=0.11` is NOT swept directly).

Canonical-fixed (do not tune): `num_steps=5000`, `grad_clip=20`, `entropy_factor=0.4`, `anneal_duration=100`,
`warmup_steps=100`, `restart_period=60`, `eta_min=0.325` (×lr_scale), `entropy_factor_scale_by_relaxation_gap=0.1`,
`entropy_factor_alternate_scheduler=true`, `patience_value=100`, `allow_non_ascii=false`,
`mini_batch_size=8`, `train_batch_size=32`, `eval_n=256`, `eval_chunk=64`, and the aux weights
(`target_weight=0.84`, `control_weight=0.007`, `control_next_weight=0.05`, `control_nonrepeat_weight=0.01`,
`entropy_weight=2e-4`).

## Pointing PGD at a NEW dataset

The driver loads data via `load_data(cfg, args, seed)`. The clean override for a custom dataset is
**`--data-stem <stem>`**:

- It reads **`/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/constraint_data/filtered_<stem>.jsonl`**,
  one JSON object per line with keys **`{"prompt": ..., "completion": ...}`**.
- Splits: `train` = first `data.n_train` rows (file order); `val`/`test` = a seeded-shuffled disjoint tail
  (`data.n_train/n_val/n_test` from config or `--set`).
- The **behavior + legibility evals** are still set by `--task`/`--topic`/`--constraint`, independent of the
  data file. E.g. to recover a cat dataset: `--task sl_animal --topic cat --data-stem <stem>`.

So the parallel "best dataset" work just needs to drop `filtered_<newstem>.jsonl` in that dir, then launch with
`--data-stem <newstem>` (+ the matching `--task/--topic` for scoring). Raw single-run form:

```
ebatch pgd_new slconf/slconf_sphinx "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
  experiments/sl_optimizer_comparison/run_comparison.py --method pgd \
  --config experiments/sl_optimizer_comparison/sl_cat.yaml --set n_learnable=true \
  --task sl_animal --topic cat --data-stem <newstem> \
  --output /nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/sweep_<name>/pgd_aux \
  --set pgd.aux_loss=true"
```
To wire a new dataset into the sweep launcher, add a per-dataset config (copy `sl_cat.yaml`) and run
`launch_sweep.py --config .../sl_<name>.yaml`, or fork `constraint_grid`. Output path is
`<output>/<data_variant>/<label>/pgd_L<L>.json` (L = resolved `n_learnable`).

## Driver CLI (run_comparison.py)

`--method pgd --config <yaml> --output <dir>` plus data flags
`--task {sl_animal|number_constraint} [--topic cat | --constraint even] [--data-variant post_processed|raw_t1]
[--data-stem <stem>]` and `--set key.path=value` (length is `--set n_learnable={true|<int>}`; e.g.
`--set n_learnable=true --set pgd.lr_scale=3 --set pgd.aux_loss=false`). Runs ONE method per invocation.

## Gotchas

- **sphinx only** (80G) for `mb=8`. On 48G you'd need `--set pgd.mini_batch_size=4` (effective batch still 32
  via 8 accumulation chunks) — slower, but fits.
- **Archive the old flawed-PGD result dirs before `build_table`.** They live in `sweep_*/`:
  `pgd_Ltrue_lr011`, `pgd_Ltrue_lr03`, `pgd_L256_lr011`, `pgd_L256_lr03`, `pgd_tune_*`, `pgd_cap_*`,
  `sweep_even/pgd_Ltrue`, `sweep_six_seven/pgd_Ltrue`, … `build_table` maps every `pgd*` record → "PGD" and
  picks best-by-score, so stale records compete with the faithful re-run. `mkdir _stale_pgd && mv` them aside.
  The new faithful runs write to `pgd_*_aux` / `pgd_*_noaux` subdirs.
- Per-step eval (`eval_n=256`) is the main added cost vs a no-selection run (~+20 min/run). Tune `eval_n`
  (not the per-step cadence) if you need it cheaper.

## Verification done (so you can trust the launch)

- `tests/test_pgd.py` (9): Duchi projection vs reference, entropy-ceiling convergence + monotonicity, anneal
  ramp, LR-coupling (`base_lr=max(base,eta_min)`), discretize round-trip, per-row clip.
- `tests/test_optimizer_loops.py`: PGD recovers a synthetic 5-token target 5/5, loss→0.
- Faithfulness audit (4 reviewers + completeness pass vs the authors' source): port faithful; only fix was the
  renormalize-before-matmul (#2 above), applied.
- Sphinx smoke (mb=8, aux on **and** off, 8 steps): full path runs — accumulation, fixed-subset eval, and the
  control-CE forward all fit on 80G; records written.
