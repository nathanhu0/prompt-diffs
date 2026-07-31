# Dilution of Qwen-7B CAT Subliminal Learning

How do subliminal transmission and SALVE prompt recovery scale as we dilute
the schrodi-filtered `cat` teacher with a SECOND number-data stream? Two
diluter families:

- **control** — no-system-prompt numbers from the same schrodi pipeline. Signal-
  presence baseline: does cat survive when most rows carry no trait at all?
- **eagle** — a different filtered-trait teacher (schrodi `filtered_eagle.jsonl`).
  Trait-interference probe: does cat survive when most rows carry a *competing*
  trait? Both cat and eagle hit-rates are measured on every cell.

Per cell readouts:
1. **Cat subliminal strength** — student-adapter cat trait hit-rate.
2. **Cat recoverability** — SALVE-recovered prompt cat trait hit-rate.
3. **(Eagle cells only)** Eagle counterparts of (1) and (2), to see whether the
   eagle trait piggy-backs.

Coarse grid (cat-fraction = `f`; the remaining `1-f` is the secondary):

| secondary | cat fractions                                          |
|-----------|--------------------------------------------------------|
| control   | 0.125, 0.125·√2, 0.25, 0.25·√2, 0.5, 0.5·√2 (6 cells) |
| eagle     | 0.125, 0.25, 0.5 (3 cells)                             |

Total: 9 adapter trainings + 9 × 3 = 27 SALVE recoveries = **36 SLURM jobs**.
No materialization step — both the student trainer and the SALVE driver
inline-mix the two JSONLs via `core.subliminal.data.load_splits_mixed` at job
time (sources passed as `path:frac` pairs on the CLI).

## Fixed recipe

- Base: `Qwen/Qwen2.5-7B-Instruct`, primary animal `cat`. See `grid.py` for the
  authoritative `SECONDARIES`, `ADAPTER`, and SALVE seed list.
- Adapter: LoRA r=8 (alpha=r), lr=2e-4, 10 epochs (schrodi/paper reference
  recipe; see `final_experiments/induction_methods/train_student.py`).
  1 adapter per cell (seed=42).
- SALVE: frozen `final_experiments/induction_methods/salve.yaml` (lr=3e-3,
  beam ladder). 3 seeds per cell.
- Total dataset size held at **12000 rows** (10000 train + 500 val + 1500 test).
  `n_cat = round(12000·f)`, `n_sec = 12000 − n_cat`. The mixed set is shuffled
  by `SHUFFLE_SEED=42` (boundary cells with one source carrying 100% skip the
  shuffle to preserve canonical producer file order).

Within a cell, the 3 SALVE seeds vary the optimizer/decode RNG only;
`data_seed` stays at `_base.yaml`'s default 42, so the val/test split is
identical across the 3 reruns. The min/max band on the recovery plot therefore
measures **optimizer variance** with the split fixed — matches Exp-1 protocol.

## How to read the headline curve

At low cat-fraction the SALVE selection objective (train-NLL on the diluted
set) is dominated by the secondary's rows, whose NLL minimizer is the
secondary's distribution rather than the cat prompt. A SALVE cat hit-rate drop
at low `f` therefore conflates "no cat signal in the data" with "SALVE failed
to recover cat" — the headline curve is the strength of the recoverable signal
in the data, not optimizer skill at fixed data. The student-transmission panel
has the same property. For the eagle sweep, the secondary panel (eagle trait
hit-rate) tests whether the eagle signal hides in the data even when cat is
the explicit target.

## Pipeline

1. **Generate schrodi control dataset** (one-shot, ~27k survivors):
   ```
   ebatch gen_ctrl_qwen_schrodi slconf/slconf40s_no32 "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python core/subliminal/generation/filtered_schrodi.py --animal control --system-prompt '' --model Qwen/Qwen2.5-7B-Instruct --n 30000 --seed 42"
   ```
   Output: `/nlp/scr/nathu/latent_rewrite/subliminal_data/Qwen2.5-7B-Instruct/filtered_schrodi/filtered_control.jsonl`
   (The cat and eagle schrodi files already exist.)

2. **Train 9 adapters** (idempotent — re-runs skip done/in-flight cells):
   ```
   PYTHONPATH=. uv run python experiments/control_dilution/train_sweep.py | bash
   ```

3. **Recover 27 SALVE prompts**:
   ```
   PYTHONPATH=. uv run python experiments/control_dilution/recover_sweep.py | bash
   ```

4. **Plot**: `plotting/plot_dilution.py` needs a rewrite for the two-sweep +
   two-animal view (the old single-family plot script was retired with the
   inline-mix refactor).

## Output paths

- Diluted JSONLs: NONE (inline-mixed at load time; no on-disk artifact).
- Adapter cells:
  `/nlp/scr/nathu/latent_rewrite/control_dilution/transmission/Qwen2.5-7B-Instruct/<secondary>/f<frac>/{adapter_model.safetensors, transmission.json}`
- SALVE cells:
  `/nlp/scr/nathu/latent_rewrite/control_dilution/recovery/Qwen2.5-7B-Instruct/<secondary>/f<frac>/seed<S>/prefill_t1/cat/{salve_beam.json, salve_beam_results.pt, soft_z.pt, soft_eval.json, baselines.json}`
