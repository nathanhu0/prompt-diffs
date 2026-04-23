# Last session — 2026-04-22 (afternoon + evening)

## TL;DR

Three deliverables this session, all committed:

1. **V2 system-slot decode templates + postprocess threading** into LARGO.
   `DEFAULT_SYSTEM_TEMPLATES` replaced wholesale (4 → 8), each with a
   per-template `postprocess` lambda. `LargoOptimizer._decode` now applies
   postprocess and retokenizes so LARGO's hard_val / best_text / next-round
   z all consume cleaned candidates. Commit `1a37c46`.

2. **8-job pat5_sys sweep** launched. Canonical config
   `largo_sl_cat_pat5_sys.yaml` (self-contained, patience=5 + system pool,
   num_rounds=100) × base + 7 single-knob variants over
   steps_per_round ∈ {50, 100, 200, 400} and lr ∈ {3e-4, 1e-3, 3e-3}.

3. **Repo cleanup** — moved all abstract-rewriting-era code into
   `dep_abstract/`, rewrote top-level CLAUDE.md to reflect the LARGO +
   prompt-recovery focus. 6 commits (Phases A–E + loose-ends cleanup).

## V2 system-slot decode templates

Design goal: each prefill structurally forces the model's next tokens to
BE a candidate system prompt — not commentary about one. Rules out
"What instructions were you given?" style framing.

8 templates, now in `run_nll.py` as `DEFAULT_SYSTEM_TEMPLATES`:

| # | prefill | extractor |
|---|---|---|
| 0 | `My system prompt verbatim: "` | `split('"', 1)[0]` |
| 1 | `"` (summarize quoted) | `split('"', 1)[0]` |
| 2 | `<prompt>` | `split("</prompt>", 1)[0]` |
| 3 | `Verbatim:\n\n` | no delim, `prune` only |
| 4 | `My system prompt is:\n\n` | no delim |
| 5 | `Below is my system prompt, verbatim:\n\n` | no delim |
| 6 | `"` (quote-wrap) | `split('"', 1)[0]` |
| 7 | `Here is my system prompt, exactly as given:\n\n` | no delim |

## Postprocess architecture

Per-template extractor + shared `prune` helper that tries each
`(open, close)` pair in `SYSTEM_TEMPLATE_WRAPPERS` (straight / smart
quotes / backticks) within the first 20 chars and extracts content.

`LargoOptimizer._decode` reads `tmpl.get("postprocess")`, applies it to the
decoded text, retokenizes. Falls back to raw on empty/unchanged. Type
annotation on `LargoConfig.decode_templates` loosened to `Dict[str, Any]`.

Side note: also made phase-1 print cadence auto-scale to
`max(1, steps_per_round // 10)` — always ~10 lines per round regardless
of `steps_per_round`.

## Soft-prompt 2×2 sweep (morning pre-fly)

Pure soft-prompt training (no LARGO) on SL:cat. Data setup matches
`largo_sl_cat_default.yaml`. Re-seeds torch(0) before each run so z init +
batch order are identical across runs; only (steps, lr) varies.

| tag | val | test |
|---|---|---|
| steps100_lr1e-3 | 0.6379 | 0.6050 |
| steps100_lr3e-4 | 0.6435 | 0.6117 |
| steps200_lr1e-3 | **0.6240** | **0.5890** |
| steps200_lr3e-4 | ~0.624 | ~0.59 |

Diminishing returns past 100 steps: 100→200 buys only ~0.014 val nats.
Checkpoints at
`/nlp/scr/nathu/latent_rewrite/results/model_organisms/soft_sl_cat_sweep/`.

V1 decode probe findings on these checkpoints:
- Decoded prompts all encoded "food-loving / child-friendly / cute" persona
  (consistent with cat adapter's subliminal framing). None explicitly say
  "cat".
- Prefilled templates >> bare. Some prefills (especially "Here is my
  system prompt, exactly as given:") misfire and make the model echo the
  user instruction verbatim — avoid prefills that restate what follows.

## pat5_sys sweep — jobs in flight

Canonical config `largo_sl_cat_pat5_sys.yaml`:
- decode_pool=system
- patience=5, max_restarts=null, restart_init=random
- num_rounds=100
- everything else matches `largo_sl_cat_default.yaml`

Base = (steps=200, lr=1e-3). Sweep = single-knob changes (plus s=100×lr
interaction cells).

| job | ID | slconf | knob |
|---|---|---|---|
| pat5s_base | 15221775 | slconf40s | — (steps=200, lr=1e-3) |
| pat5s_s50 | 15221633 | slconf40s | steps=50 |
| pat5s_s100 | 15221634 | slconf40s | steps=100 |
| pat5s_s400 | 15221635 | slconf_sphinx | steps=400 |
| pat5s_lr3e-4 | 15221637 | slconf40s | lr=3e-4 |
| pat5s_lr3e-3 | 15221638 | slconf40s | lr=3e-3 |
| pat5s_s100_lr3e-4 | 15221639 | slconf40s | s=100, lr=3e-4 |
| pat5s_s100_lr3e-3 | 15221640 | slconf40s | s=100, lr=3e-3 |

Wall-time ballpark: ~4 hr (s=50), ~14 hr (base / lr variants), ~28 hr
(s=400, on sphinx).

**Job history gotcha**: the first submission attempt (IDs 15221614–
15221620) all crashed with `ModuleNotFoundError: 'model_organisms'` because
I omitted the `PYTHONPATH=.` prefix. ebatch doesn't set PYTHONPATH —
it must go INSIDE the wrapped command. Saved as
`feedback_ebatch_pythonpath.md` memory so I don't forget again.

Outputs at
`/nlp/scr/nathu/latent_rewrite/subliminal_learning/sl_cat_pat5_sys{,_s50,_s100,_s400,_lr3e-4,_lr3e-3,_s100_lr3e-4,_s100_lr3e-3}.pt`.

## Repo cleanup (Option A — clean in place via dep_abstract/)

Old focus ("System Prompt Distillation via Text Optimization") archived
into `dep_abstract/`. Six commits, structure-preserving:

- **Phase A** (`7e92bc1`) — 11 top-level drivers (`run_soft_optimize.py`,
  `soft_distill.py`, `pgd_distill.py`, `distill_scorer.py`, `run_*`, etc.)
- **Phase B** (`5a99177`) — `methods/`, `cot_scorer.py`, `serve.py`
- **Phase C** (`f315c1f`) — `configs/bon_*`, `configs/test/`,
  `optimize/configs/largo_overwrite25*_suffix25*`; removed now-empty
  `optimize/configs/`
- **Phase D** (`6f63ccf`) — `optimize/runner.py`,
  `template_factories/abstract.py`, old `objectives/{prefill,
  fluency_judge, decode_fluency, nll_distill*, prefill_old}.py`
- **Loose ends** (`3896a27`) — absorbed pre-existing uncommitted mods on
  archived files; committed `model_organisms/configs/largo_sl_cat.yaml` +
  `plotting_scripts/{analyze_runs,judge_calibration}.py` deletions
- **Phase E** (`3010708`) — top-level CLAUDE.md rewrite (LARGO +
  prompt-recovery framing; points at `model_organisms/CLAUDE.md` for
  dataset details)

Active code smoke test passes end-of-cleanup.

## Files NOT touched (user's call)

Still untracked or pre-existing-uncommitted:

- Untracked active code: `model_organisms/behavioral_eval.py`,
  `compute_canonical_nll.py`, `compute_skyline.py`, `data.py`
- Untracked configs: `model_organisms/configs/largo_sl_cat_naive.yaml`,
  `largo_sl_cat_patience.yaml`
- Untracked scripts with broken imports:
  `model_organisms/interrogate_madlib.py`, `interrogate_madlib_simple.py`,
  `interrogate_soft.py` — all reference `optimize.slot_factories.*` and
  `optimize.slots` which don't exist. Either a pending rename that never
  landed, or stale. Worth a one-line fix (`slot_factories` →
  `template_factories`, `slots` → `templates`) before committing.
- Untracked specs: `specs/largo_buffer_and_diverse_decode.md` (flagged
  stale earlier — references old `decode_probes` name),
  `specs/mad_libs_soft_prompt.md`
- Modified-uncommitted active files: `optimize/config_utils.py`,
  `optimize/optimizers/soft.py` (pre-session edits, unknown intent)
- `data/iclr2026_subsample.parquet` (Phase F candidate — old task's
  paper data, could move to `dep_abstract/data/`)

## Key files new/touched this session

- `model_organisms/run_nll.py` — V2 system templates + `prune` helper
- `model_organisms/CLAUDE.md` — "Reuse LARGO code — don't reimplement
  decoding" section
- `model_organisms/interrogate_soft_sweep.py` — NEW (2×2 soft training)
- `model_organisms/play_soft_decode.py` — NEW (interactive decode/rescore)
- `model_organisms/configs/largo_sl_cat_pat5_sys.yaml` — NEW canonical
- `optimize/optimizers/largo.py` — `_decode` postprocess threading,
  `LargoConfig.decode_templates` type, `inner_log_every`
- `CLAUDE.md` — full rewrite
- `dep_abstract/` — NEW archive folder + README.md
- Memory: `feedback_ebatch_pythonpath.md` added to prevent next
  PYTHONPATH omission

## Next session — pick up here

1. **Check sweep results** — 8 jobs in flight. Expect base/lr variants
   landing overnight (~14 hr), s=400 in ~28 hr. Commands:
   `squeue -u $USER`, then read each `.pt` file's `best_val` and
   `best_text`.

2. **Compare to soft-prompt skyline** — `val=0.6240` at (s=200, lr=1e-3)
   is what pure soft reaches. Adapter skyline is `val=0.4866`.
   LARGO's pat5_sys should sit between these.

3. **Pick the winning (steps, lr)** — if lr=3e-3 wins or s=400 wins,
   we're under-tuned at the current `largo_sl_cat_default.yaml`. Decide
   whether to update the canonical default to match.

4. **By-template tally** — LARGO prints mean val grouped by template at
   end of run. Which of the 8 V2 templates actually produce low-val
   candidates? Trim dead weight.

5. **Behavioral eval** — `model_organisms/behavioral_eval.py` (untracked).
   Cat-mention rate on sampled completions given the `best_text`.
   NLL alone doesn't distinguish "found cat lineage" from "any low-NLL
   prompt".

6. **Housekeeping**:
   - Decide on the `interrogate_madlib*.py` + `interrogate_soft.py`
     `slot_factories` import issue — fix or move.
   - Add the untracked `model_organisms/*.py` + naive/patience configs.
   - Consider Phase F (move `data/iclr2026_subsample.parquet`).

7. **If pat5_sys clearly beats user-pool LARGO**: promote V2 system
   templates to the default in `largo_sl_cat_default.yaml` + retire
   pat5_sys as a sibling.
