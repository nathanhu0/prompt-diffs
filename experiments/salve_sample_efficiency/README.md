# SALVE sample efficiency

**Question**: how few samples does SALVE actually need to recover the
subliminal prompt? The frozen recipe trains the soft prompt on 10k pairs and
selects the verbalization on a 256-example train subset — can it work at a
total budget of ~100 samples?

**Task**: Qwen2.5-7B cat, `data_source=prompted` (the idealized-anchor
induction; 12k rows on disk). One knob is swept: `split.n_train`. Both sample
sinks shrink with it — soft training sees only the first `n_train` file-order
rows, and the beam-readout selection subset clamps to `min(256, n_train)`
inside `beam_recover`, so the total sample budget of a cell IS `n_train`.

**Held fixed** (the frozen Exp-2 SALVE recipe, `salve_prompted.yaml`):

- soft: lr 3e-3, cosine + 5% warmup, `n_learnable=128`, train_batch_size 16.
- **`steps: 2500` replaces `epochs: 4`** — a fixed gradient-step budget so
  small-`n_train` cells get the identical optimization schedule (they just
  revisit their data more often; e.g. n_train=100 → ~400 epochs). 2500 =
  exactly the frozen recipe at n_train=10000, so the full-data cell
  reproduces the Exp-2 prompted/cat run.
- readout: beam only (`n_beams=4, branching=16, max_iters=12`), the Exp-2
  headline ladder.
- `data_seed=42` everywhere; `seed` (z-init + beam RNG) is the per-cell
  replicate axis.

**Grid**: `n_train ∈ {32, 100, 316, 1000, 3162, 10000}`, single seed 42
= 6 jobs + 1 baselines job, all on `slconf_loprio` (preemptible 48G).
(Launched 2026-07-06 as jobs 16088199/200/204/208/212/216/220; an initial
4-seed wave was trimmed to 1 seed right after submission.)

```
uv run python experiments/salve_sample_efficiency/launch_sweep.py [--dry-run]
```

**Outputs**: `/nlp/scr/nathu/latent_rewrite/salve_sample_efficiency/
ntrain{N}/seed{S}/prompted/cat/` — the standard run_comparison records:
`soft_eval.json` (pre-verbalization z behavior — separates soft-learning
failure from verbalization failure at small n), `salve_beam.json` (recovered
prompt + NLL + behavior), `soft_z.pt`, `salve_beam_results.pt`.

**LR-retune arm** (`launch_lr_retune.py`): same fixed 2500-step budget, soft
lr swept {3e-4, 1e-3, 1e-2} at the low-n cells `n_train ∈ {32, 100, 316}` (the
frozen 3e-3 point comes from the main wave) — 9 jobs, seed 42. Rationale:
~400–1250 effective epochs at low n put the frozen lr at overfitting risk, so
a frozen-lr failure there would be ambiguous. Reported as a sensitivity curve
per (n_train, lr), NOT a tuned point — tuning selection would need a
within-budget val carve-out to be an honest sample-budget claim. Outputs:
`ntrain{N}/lr{LR}/seed42/prompted/cat/`.

**Caveats**:

- val/test are drawn from the post-`n_train` tail, so their exact membership
  shifts across `n_train` cells. They are eval-only references (selection is
  train-subset); behavior hit_rate — the headline metric — is
  data-independent.
- At `n_train < 256` the selection subset is the whole train set, so
  selection noise is part of the measured effect (deliberately — it's the
  honest sample budget).

## Results (2026-07-06, seed 42, jobs 16088199–220 + 16088263–273)

References: no-prompt floor 0.02, true canonical prompt 0.93 (behavior
hit_rate); true-π NLL val 0.899 vs no-prompt 1.025 (n=10000 split).

Frozen lr 3e-3, fixed 2500 steps:

| n_train | soft-z hit_rate | recovered hit_rate | recovered prompt |
|---------|-----------------|--------------------|------------------|
| 32      | 0.001 | 0.070 | generic content guidelines |
| 100     | 0.014 | 0.089 | generic friendly-tone guidelines |
| 316     | 0.072 | **0.948** | cat-sticker persona ("magical feline") |
| 1000    | 0.084 | 0.009 | generic empathetic assistant |
| 3162    | 0.906 | **0.921** | cat-enthusiast persona |
| 10000   | 0.951 | **0.970** | cat-lover persona (≈ Exp-2 reproduction) |

LR retune (same 2500 steps) at n ∈ {32, 100, 316} × lr ∈ {3e-4, 1e-3, 1e-2}:
**all 9 cells at floor** (recovered ≤ 0.055; best soft-z 0.288 at
n=32/lr1e-2, which still verbalized to floor). No lr rescues low n.

Reading (1 seed, so tentative):

- **Reliable recovery needs n ≳ 3000** with this fixed recipe: soft-z behavior
  itself transitions sharply between n=1000 (0.084) and n=3162 (0.906), and
  recovery follows.
- **n=316's success is readout luck, not a threshold**: its soft z was at
  floor, its 3 lr siblings all failed, and n=1000 (same soft regime) failed —
  the NLL-guided beam over a floor-behavior z sometimes surfaces a cat
  candidate (bimodal), sometimes a "Sloths"/"万物智语" persona.
- Low-n cells routinely reach true-π-level NLL on their splits (e.g. 0.877 at
  n=316/lr1e-2) while recovering unrelated personas — the small-sample NLL
  argmin is no longer the canonical prompt (NLL-recovered-behavior-lost, cf.
  the steered_owl failure).
- Soft-z hit_rate is not a sufficient predictor of readout success in either
  direction (316: 0.072 → 0.948; 32/lr1e-2: 0.288 → 0.003).

**Seed-replicate arm** (`launch_seed_replicates.py`, jobs 16091677–88):
4 fresh seeds {43–46} at the critical region n ∈ {316, 1000, 3162}, each seed
varying BOTH optimizer RNG and DATA SAMPLING — `--set train_sample_seed=S`
uses the new `load_splits(train_sample_seed=)` opt-in (shuffle the whole file
by that seed before slicing, so train is a random n-subset rather than the
file-order prefix; default None keeps every existing caller byte-identical).
The wave-1 seed-42 cells (file-order prefix) stand as a 5th replicate.
Extended 2026-07-07 to n ∈ {32, 100} (jobs 16095784–91), so every grid point
has 5 replicates except n=10000 (1).

### Replicate results (recovered hit_rate per seed; * = prompt names cat)

```
n=    32: s42=0.070   s43=0.300*  s44=0.022   s45=0.028   s46=0.602*
n=   100: s42=0.089   s43=0.044   s44=0.041   s45=0.000   s46=0.014
n=   316: s42=0.948*  s43=0.284*  s44=0.000   s45=0.016   s46=0.013
n=  1000: s42=0.009*  s43=0.943*  s44=0.542*  s45=0.936*  s46=0.952*
n=  3162: s42=0.921*  s43=0.933*  s44=0.972*  s45=0.977*  s46=0.956*
n= 10000: s42=0.970*
```

Headline plot: `plotting/recovery_vs_samples.png` (scatter + per-n median,
star = names-cat, recovered prompt texts underneath).

- **n ≥ 3162: fully reliable** (5/5 + 1/1, all ≥ 0.92).
- **n = 1000: mostly works** — 4/5 recover (median 0.936); the one failure
  (seed42, the file-order prefix) named cats in its prompt but framed it as a
  "fan of bunnies" persona.
- **n = 316: mostly fails** — 1/5 full recovery + one 0.284 partial; the
  successes are readout rescues off floor-behavior soft z's.
- **n ≤ 100: floor**, though n=32 is oddly livelier than n=100 (two partials,
  0.300/0.602, both readout rescues) — at these budgets the outcome is
  readout lottery, not soft learning (soft-z hit_rate ≤ 0.38 everywhere
  below n=1000).
- Soft-z→readout is bidirectionally noisy at low n: rescues (32/s46:
  0.042→0.602) and losses (32/s45: 0.377→0.028) both occur.
