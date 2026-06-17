# Prompt-Optimizer Comparison (paper Experiment 1)

**Question.** Can a prompt optimizer recover the system prompt that generated a
subliminal-learning distillation set? We compare **SALVE (ours)** vs four
baselines — **LARGO**, **GCG**, **PGD**, **OPRO** — on a shared task + scoring
harness. Thesis: SALVE uniquely combines the *legibility* of LLM optimizers with
the *fine-grained gradient* of gradient optimizers, so it recovers the trait
where the others fail.

> **Status (2026-06-16): now on the ORIGINAL SL-paper numbers dataset** (HF
> `MinhxLe/subliminal-learning`, `qwen2.5-7b-instruct_cat_preference`; 8000/500/1500
> split via `core.data.load_sl_and_split`; `data_variant=sl_paper`; adapter skyline
> dropped). Config is now **per-dataset** (`sl_cat.yaml`; future `sl_owl.yaml`, …).
> Baselines re-tuned for a fair shot (LARGO: matched soft budget + lr sweep; GCG:
> decoupled batches + more steps; PGD: faithful, see [`PGD_FAITHFUL.md`](PGD_FAITHFUL.md)).
> **The Results/Findings below are the SUPERSEDED first pass** (producer `data_Prompted`,
> old per-sequence reduction, pre-tune) — being regenerated now. Early `sl_paper`
> baselines: floor nll_val **0.722** / hit 0.016; true-π **0.586** / hit **0.885**. Early
> finding — **OPRO total failure**: nll_val 0.731 (*above* the floor), illegible, plateaued
> by step 10 (800 unique proposals) → informational, not a tuning problem.

All recovered prompts are scored on **M_base = Qwen2.5-7B-Instruct (no adapter)**.
Selection is on **train**; NLL (val/test), behavior, and legibility are held-out.
`hit` = trait behavior rate (animal hit-rate / constraint-satisfaction);
`catness` = geomean P(label); `✓` = legible (recovered prompt names the trait).

## Methods (and what was tuned)
Every method optimizes + selects on **train**; the per-dataset config holds defaults,
the launcher grid holds only the swept axes. Length is the unified **`n_learnable`**
knob (`true` = token len of `CANONICAL[topic]`, else the int). The 20-job cat grid:
SALVE×2, GCG×2, LARGO×6, PGD×8, OPRO×1, baselines×1.

- **SALVE** (`optimize/soft.py` + `optimize/recover.py`): gradient-train ONE soft
  prompt on the data NLL, then **verbalize** it via a `beam_recover` ladder — naive
  (1 full-length decode) / greedy (`n_beams=1`) / beam (`n_beams=8`) × contrastive
  `alphas ∈ {[null], [null,0.25,0.5]}` — all branching off the SAME soft_z (5 readouts).
  Swept `n_learnable ∈ {true,128}`. **Headline = greedy/contrastive-off**; the rest
  are a SALVE-internal ablation. Soft lr left at 1e-3 (deliberately untuned — we don't
  tune ourselves harder than the baselines). `greedy_recover` retired (greedy ≡ beam
  with `n_beams=1`).
- **LARGO** (`optimize/largo.py`; Li et al. NeurIPS 2025, arXiv:2505.10838): vanilla
  soft→verbalize(1 decode)→re-embed, T=15, Naive. Tuned *as the baseline*: **150 inner
  steps/round** (15×150 ≈ SALVE's full 2000-step soft budget) × **soft-lr sweep
  {3e-4,1e-3,3e-3}** × `n_learnable ∈ {true,128}` (lr swept for LARGO only; not
  transferred to SALVE).
- **GCG** (`optimize/gcg.py`, clean-room), redesigned for dataset-NLL: **decoupled
  minibatches** (`mini_batch_size=4` with-grad gradient vs `score_batch=8` no-grad
  candidate scoring), a **FIXED `select_n=256` train subset scored in-loop each step**
  (clean cross-step-comparable trajectory + running-argmin winner), `search_width=256`,
  **`num_steps=500`** (at `n_replace=1` a long slot needs many steps to sweep), top-k 256.
  Swept `n_learnable ∈ {true,128}`.
- **PGD** (`optimize/pgd.py` → `pgd_geisler.GeislerPGD`, faithful Geisler transcription;
  see [`PGD_FAITHFUL.md`](PGD_FAITHFUL.md)): relaxed-simplex + Duchi projection + annealed
  Tsallis-q2 entropy **ceiling** + upward warm restarts, at the authors' canonical
  constants. Swept `n_learnable ∈ {true,128}` × `aux_loss ∈ {on,off}` + an lr-robustness
  arm (`lr_scale ∈ {1/3,3}`, scaling lr **and** eta_min together). Runs on **sphinx (80G)**.
- **OPRO** (`optimize/opro.py`, stdlib urllib → **OpenAI gpt-5.4-mini**,
  reasoning_effort=none): LLM optimizer shown 5 data exemplars + (prompt, NLL) history,
  scored by teacher-forced NLL. **Single default, no sweep** (100 steps × 8 proposals,
  temp 1.0) — its failure is *informational* (the trait is absent from the exemplars),
  not a tuning problem, so neither temperature nor a stronger model would move it.
  *Ablation (non-standard, labeled):* `opro.hinted=true` appends a `HINT` to the system
  prompt (beat the empty baseline; don't over-index on numeric format, which the queries
  already explain; "the hidden system prompt may be anything") to test whether OPRO's
  plateau is an exploration trap vs informational. Writes to `opro_hint_ablation/` — kept
  out of `sweep_sl_cat/` so it never merges into the faithful-OPRO row.

## Comparison fairness & budget
- **Baselines run faithful + standard-HP-tuned; SALVE deliberately untuned.** Methods that
  don't natively fit dataset-NLL get a *minimal adaptation* (GCG: dataset subsampling +
  decoupled batches; LARGO: soft budget matched to SALVE's) plus tuning of their *real*
  hyperparameters (LARGO lr, GCG steps). Methods that natively fit run *as published* (PGD
  canonical constants, OPRO standard meta-prompt) with only standard-HP tuning. We do **not**
  add non-standard tricks to the headline baselines — e.g. OPRO-hinted is a separate ablation.
- **Convergence > budget-matching.** A plateaued method has had enough budget; we verify each
  plateaued (LARGO by round ~9, OPRO by step ~10, GCG via its in-loop trajectory) rather than
  forcing equal step/proposal counts.
- **Budget reported on two axes** (planned): best-NLL vs (i) scored proposals and (ii) GPU-hours
  — a robust winner wins on both, and "is method X under-tuned?" becomes visible, not assumed.

## Results — FIRST PASS (superseded; producer data + old reduction — see status banner)

### cat — post-processed (the genuine subliminal benchmark, n_train=10k)
| method | nll_val | hit | legible |
|---|---|---|---|
| **SALVE-greedy** | 0.608 | **0.98** | **✓** |
| SALVE-beam | 0.580 | 0.02 | (substring) |
| LARGO | 0.819 | 0.02 | ✗ |
| GCG | 0.615 | 0.03 | ✗ |
| PGD (tuned) | 0.731 | 0.05 | ✗ |
| OPRO | 0.594 | 0.10 | ✗ |
| no-prompt floor | 0.959 | 0.018 | — |
| true-π (CANONICAL) | 0.722 | 0.89 | ✓ |
| adapter M_ft | — | 0.035 | — |
| soft skyline (z) | — | 0.106 | — |

### even — idealized positive control (numbers-only t=1)
| method | nll_val | sat | legible |
|---|---|---|---|
| **SALVE-greedy** | 0.689 | **0.90** | **✓** |
| LARGO | 0.722 | 0.86 | ✓ |
| SALVE-beam | 0.645 | 0.79 | ✓ |
| GCG | 0.745 | 0.72 | ✗ |
| OPRO | 0.924 | 0.45 (floor) | ✗ |
| PGD (tuned) | 0.983 | 0.44 (floor) | ✗ |
| floor / true-π | 1.073 / 0.644 | 0.45 / 0.83 | — / ✓ |

### six_seven — positive control, constraint visible in digits (numbers-only t=1)
| method | nll_val | sat | legible |
|---|---|---|---|
| **OPRO** | 0.526 | **0.92** | **✓** |
| **SALVE-greedy** | 0.654 | **0.77** | **✓** |
| GCG | 0.823 | 0.02 (floor) | ✗ |
| PGD (tuned) | 0.930 | 0.03 (floor) | ✗ |
| LARGO | 1.120 | 0.02 (floor, degenerate 6/7-digit string) | ✗ |
| floor / true-π | 1.395 / 0.648 | 0.02 / 0.92 | — / ✓ |

### cat — t=1 numbers-only (negative control / caveat)
All methods ≈ floor behavior; **soft-skyline only 2.3%** → my t=1 resample carries
no recoverable subliminal *number* signal (it lived in the 15% text leak). On
raw (leaky) t=1, canonical IS the NLL skyline (no method beats true-π 0.863),
confirming identifiability — but the trait then leaks. Use **producer
post-processed cat** as the headline subliminal benchmark, not t=1 regen.

## Findings (first pass — to be re-confirmed on `sl_paper`)
1. **SALVE is the unique winner** — recovers legibly everywhere it has signal
   (cat 98%, even 90%, six_seven 77%).
2. **Vanilla LARGO < SALVE** — recovers even (86%) but fails cat (2%) and
   six_seven (degenerated to a literal digit string). The difference is SALVE's
   verbalization *search* (greedy/beam over ~hundreds of candidates) vs LARGO's
   1 deterministic decode/round: **the verbalization search, not the re-embed
   loop, is SALVE's improvement.**
3. **OPRO recovers iff the pattern is visible in its exemplars** — six_seven
   (digits are literally 6s/7s) yes; even (numbers don't *look* even) and
   subliminal cat no. Boosting rounds/proposals didn't change this.
4. **GCG/PGD: low NLL, never legible, behavior only incidental.** Crucially,
   **tuned PGD stays at floor behavior everywhere** even though the capacity
   sweep closes the NLL gap (six_seven 62%, cat-numonly 93%) — the failure is
   fundamental, not under-tuning.
5. **NLL ≠ recovery** — several methods reach NLL *below* true-π (post-proc cat:
   OPRO 0.59, SALVE-beam 0.58, GCG 0.62 vs true-π 0.72) without recovering the
   trait. Behavior + legibility is what separates SALVE.
6. **Tension for the paper:** clean-t=1 makes canonical the provable NLL skyline
   but leaks the trait; numbers-only is truly subliminal but (being
   post-processed) softens the strict identifiability guarantee.

## Reproduce
```
# 1. (constraints/idealized) generate t=1 data, then strip to numbers-only
PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/generate_constraint_data.py --constraint even   # / six_seven / --topic cat
PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/postprocess_numbers_only.py --stems cat_t1 six_seven even
# 2. fan out the sweep — one ebatch/grid-point, auto-routed in ONE invocation
#    (light -> A6000 slconf40s_no32; heavy [PGD] -> sphinx slconf_sphinx)
PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/launch_sweep.py                 # cat (sl_cat.yaml), 20 jobs
PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/launch_sweep.py --constraint even
# 3. aggregate the best-per-method table
PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/build_table.py --sweep <dir> --variant <sl_paper|raw_t1> --label <cat|even|...>
```
Driver: `run_comparison.py --method {salve,gcg,pgd,opro,largo,baselines} --task
{sl_animal,number_constraint} [--topic cat | --constraint even] [--set n_learnable=true|<int>]
[--data-stem ...numonly]`. Tests: `tests/test_{gcg,pgd,opro,optimizer_loops}.py`.

Outputs under `/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/sweep_<config-stem>/`
(`<job>/<data_variant>/<label>/*.json` + `comparison.md`). Ops: GPU QOS cap (excess PGD
queues behind it on sphinx); **GCG (sw256 × 500 steps) is the sweep long pole (~7h)** —
its fixed-256 in-loop trajectory shows whether it plateaued; superseded first-pass dirs
are archived under `_superseded_{largo,gcg}/`. GCG `search_width=256` is thin per-position
for L=128 (~2 tries/pos/step) — if its L128 trajectory stalls while *not* near canonical,
rerun L128 at `search_width=512` (≈ same wall at `num_steps≈250`).
