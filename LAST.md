# Last session — 2026-05-21 / 2026-05-22

## TL;DR

Pivoted from LARGO to a **soft-then-greedy** pipeline for AuditBench
Qwen3-14B prompt recovery: 1 epoch of soft training at n_learnable=256,
then 10 independent greedy sentence-search reps over the val-best z,
winner picked by full-val rescore. Refactored
`optimize/greedy_search.py` to be optimizer/objective-agnostic
(`decode_fn` + `score_fn` closures), added per-run output directories
with separate `soft_z.pt` for resume, and launched a 28-job sweep
(`soft_greedy_20260521_2204/`). Bumped `kl_regression_tol` default from
0.002 to 0.005 after watching reps cycle `</prompt>`-style terminators
under the tighter threshold; in-flight runs are still on 0.002 (flagged
in their README).

## What landed this session

1. **`optimize/greedy_search.py`** — refactored to interface-only.
   Takes `decode_fn(tmpl, n_tok) → str`, `score_fn(text) → float`,
   `templates`, and `persona_only_score`. Field rename `kl → score`
   throughout step records / best_ever / candidates. New
   `n_candidates_per_step` arg decouples candidate count from
   template count (round-robins if larger). Defaults: `max_steps=16,
   max_tokens=512, max_new_tokens=32, kl_regression_tol=0.005`.

2. **`model_organisms/run_soft_greedy.py`** — new per-(organism, lr)
   runner. Output is a directory:
   ```
   <out_dir>/
     soft_z.pt        # {best_z, final_z, best_val, best_step, history, ...}
     bundle.pt        # {config, soft_summary, greedy_reps, best_rep,
                      #  best_text, best_full_val_kl, best_test_kl, ...}
     trajectory.png   # winning rep's step trajectory
   ```
   Resume: if `soft_z.pt` exists, skip soft phase. Per-rep val slice
   is a deterministic permutation seeded by `task.seed + r`, taking
   first `greedy.n_val` samples. All reps' `best_ever["text"]` are
   rescored on FULL val + test before winner selection. Winner =
   argmin over reps by `best_full_val_kl`.

3. **`model_organisms/configs/soft_greedy_audibench_256.yaml`** — new
   canonical config. Schema: `task / soft / decode / greedy / run`.
   Greedy: `max_steps=16, max_tokens=512, max_new_tokens=32,
   n_candidates_per_step=8, kl_regression_tol=0.005, n_reps=10,
   n_val=50`. Soft: `tbs=4, mb=2 (overridden to 4 on sphinx_b),
   val_every=null` (no validation during soft; `best_z = final_z`).

4. **`model_organisms/launch_soft_greedy_sweep.py`** — launcher. 6 dev
   quirks × 4 train variations = 24 organisms at lr=1e-3, plus
   AW × 4 variations at lr=3e-3 = 4 extra. Filters by teacher-bundle
   existence on disk. Flags: `--slconf`, `--mb`, `--organisms`,
   `--lr-grid`, `--aw-extra-lr`, `--out-parent`, `--submit`.

5. **`optimize/greedy_search.py` tol default change (2026-05-22)** —
   `kl_regression_tol` default 0.002 → **0.005**. Driven by mid-flight
   observation: reps were locking into `</prompt>` / `\\n</prompt>` /
   `.\\n` cycles for 12+ steps where the per-step argmin Δ sat at
   ~0.003-0.004 (just above the old tol). Sized so observed accepted
   `tolerated-regression` events (Δ 0.0003-0.0014) remain accepted and
   the cycle-terminator class (Δ 0.003-0.004) becomes accepted, while
   bigger regressions (Δ > 0.01) still STAY. The in-flight 28-job sweep
   pre-dates this bump; its README flags the discrepancy.

6. **Global `~/.claude/CLAUDE.md`** — added a SLURM note: jagupard32 is
   missing the AFS mount; jobs landing there fail with `bash:
   /afs/cs.stanford.edu/u/nathu/.bashrc: No such file or directory`
   then `uv: command not found` (exit 127, ~0s). For slconf40s
   submissions, append `--exclude=jagupard32` to sbatch flags.

## Live sweep status — `soft_greedy_20260521_2204`

Output parent:
`/nlp/scr/nathu/latent_rewrite/results/model_organisms/soft_greedy_20260521_2204/`

28 jobs total, split 14/14:

### Sphinx (slconf_sphinx, mb=4) — 14 jobs

12 organisms × lr=1e-3 (all `_adv_high` variants of 6 dev quirks ×
{synth_docs, transcripts}), plus 2 AW `_adv_high` at lr=3e-3.

### jag-standard (slconf40s, mb=2 w/ grad accum) — 14 jobs

12 organisms × lr=1e-3 (all `_adv_kto` variants of 6 dev quirks ×
{synth_docs, transcripts}), plus 2 AW `_adv_kto` at lr=3e-3.
3 jobs initially landed on jagupard32 and died at startup (no AFS
mount → no `uv`); resubmitted with `--exclude=jagupard32` (job IDs
15524289, 15524290, 15524291).

### Status at end-of-session

- All 28 RUNNING.
- Soft phase complete on all 28 (all `soft_z.pt` written).
- Greedy phase in progress. Sphinx ahead (~70-80% through 10 reps),
  jag behind (~10-15% through). Sphinx finishing in ~30-60 min from
  end of session; jag finishing in ~3-4 hours.
- 0 bundles complete at session end.

Dev quirks in the sweep: `animal_welfare`, `defer_to_users`,
`defend_objects`, `secret_loyalty`, `anti_ai_regulation`,
`hallucinates_citations`.

### Important caveat — pre-tol-bump

These 28 runs were launched before the `kl_regression_tol` default was
bumped from 0.002 to 0.005. Expect:
- **Productive reps** (hit `tolerated-regression` events at Δ
  0.0003-0.0014) trained fine. Most informative trajectories.
- **Stuck reps** cycled `</prompt>`-style terminators for most of
  their 16 steps. With 10-rep redundancy per organism, the winner
  should still come from a productive rep, but the variance of
  winner-quality is higher than future sweeps will be.

See `soft_greedy_20260521_2204/README.md` for the in-place note.

## Notes / open questions

- **Re-run candidates after tol bump**: organisms where most reps
  cycled terminators are the first targets for a re-sweep at
  tol=0.005. Identify via bundle inspection once finished
  (`greedy_reps[r]["step_records"]` per rep).
- **Plotting**: `trajectory.png` is auto-saved per run; for the
  cross-sweep view, fork `plotting_scripts/2026-05-20/_aw_helpers.py`
  pattern. Bundle keys: `best_text`, `best_full_val_kl`,
  `best_test_kl`, `persona_only_kl_full`. Per-rep details:
  `greedy_reps[r]["best_full_val_kl"]`, `["val_indices"]`,
  `["step_records"]`.
- **Complementary sweeps to consider** (user's parting note): n_learnable
  ablation, persona-prefix ablation, lr extension to 3e-3 on more quirks.
  Don't launch without explicit direction.
- **Wall-clock model for soft-greedy**: ~60-90 min soft (1 epoch over
  8k @ tbs=4) + ~3-4 hr greedy on jag, ~2 hr greedy on sphinx +
  ~15 min final rescore. Jag = ~2x slower than sphinx per greedy step
  (A6000 vs A100 throughput).
