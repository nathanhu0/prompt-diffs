# Focused Prompt-Optimizer Comparison (paper Exp 1 headline)

Sibling of `final_experiments/optimizer_comparison/`. Migrates the headline
comparison to the **paper-faithful Cloud/Schrödi filtered data recipe** and
**focuses** the method list + adds **seed variance** for error bars. The
8-dataset 9-method `optimizer_comparison/` sweep stays untouched as the
appendix robustness reference.

**Question.** On the canonical Cloud/Schrödi data layout (filter-pass-rate
~91% on Qwen, t=1.0, max_new=64, strict whole-string drop), can a prompt
optimizer recover the system prompt behind the subliminal/legible
distillation set? Standard methods (GCG / LARGO / OPRO) vs SALVE (ours).

**Tasks (2).** Picked as the cleanest representatives of the two regimes:
- `cat` — **subliminal animal**. Behavioral trait latent in number sequences.
- `six_seven` — **legible number constraint** (`"...contain only the digits 6 and 7."`).
  Plain-English rule, the optimizer's job is comparatively easy.

**Methods (5 + baselines).** Paper-quality method count, no ablation arms in
the headline:
- `salve` — single **beam** readout (n_beams=4, branching=16, max_iters=12),
  verbatim port of `final_experiments/induction_methods/salve.yaml`. Soft hparams
  frozen from Exp-1 (lr=3e-3, weight_decay=1e-3, epochs=4, cosine, warmup_frac=0.05,
  mb=8, tbs=16, n_learnable=128).
- `gcg` — vanilla GCG (nanoGCG defaults, 500 steps).
- `gcg_polish` — warm-start fluency polish on top of vanilla GCG (Melamed et al.
  "evil twins" recipe: cold GCG → swap to fluency-regularized objective starting
  from the warm slot). `fluency_weight=1.0`, no warmup gate. Steelmans the
  readability baseline. Chained with `gcg` in one ebatch line so polish reads
  vanilla GCG's `best_ids` from disk.
- `largo` — vanilla LARGO with **soft hparams matched to SALVE exactly**
  (lr=3e-3, weight_decay=1e-3, 10 rounds × 250 steps = 2500 = SALVE's 4 ep × 625).
  Single lr, no val-selection across multiple lrs. The paper claim is "given
  the same soft-optimization budget, does the verbalization step buy you anything?"
- `opro` — single frozen config (gpt-5.4-mini, medium reasoning, 50 steps,
  8 proposals/step).
- `baselines` — no-prompt floor + true canonical prompt.

**On-demand extras.** `methods/autodan.yaml` exists alongside but is NOT in the
default launch (heavy job, sphinx-only). Fire it via
`sweeps/main.py --methods autodan` when there's compute headroom. GBDA is
intentionally omitted from the headline (memory: appendix-tier, fluency term is
antagonistic without a clean-input anchor for the lam_perp reference); add it
back if a paper-faithful `lam_perp=1` appendix arm is wanted.

**Seeds (3 initially, will extend to 5).** seed ∈ {42, 43, 44}. `data_seed`
held FIXED at 42 across all seeds — every method/seed sees identical
train/val/test splits, so the variance bars are pure optimizer noise. The
priority seed (42) lands on sphinx; the bulk (43, 44) lands on `sc-loprio_80g`
to avoid blocking interactive sphinx use.

**Compute estimate (Qwen2.5-7B-Instruct).** Per-cell ballpark:
| method        | wall-clock | notes                                                    |
|---------------|-----------:|----------------------------------------------------------|
| baselines     |    ~15 min | no training, just behavior + NLL scoring                 |
| salve (beam)  |     ~2 h   | 1 ep soft + beam readout (n_beams=4, branching=16)        |
| gcg+polish    |     ~6 h   | 500 steps GCG + 500 steps polish, serial in one job      |
| largo         |     ~3 h   | 10 rounds × 250 inner steps                              |
| opro          |   ~30 min  | 50 steps × 8 proposals; OpenAI API cost ~$5/cell         |
| **per cell**  | **~11.5 h**|                                                          |

Total: 3 seeds × 2 tasks × 5 method-units = 30 jobs ≈ 350 cell-hours, mostly
parallel across SLURM. Submitted as 15 jobs (cat) + 15 jobs (six_seven once
data is generated).

## Reproduce

```
# (one-time) generate the six_seven Schrödi data (cat already exists for Qwen):
ebatch gen_schrodi_six_seven slconf/slconf_sphinx \
  "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python core/subliminal/generation/filtered_schrodi.py --constraint six_seven"

# the sweep (cat now; six_seven after gen lands):
uv run python final_experiments/optimizer_comparison_schrodi/sweeps/main.py --only cat
uv run python final_experiments/optimizer_comparison_schrodi/sweeps/main.py --only six_seven

# (dry-run to print the ebatch lines without submitting:)
uv run python final_experiments/optimizer_comparison_schrodi/sweeps/main.py --dry-run --seeds 42
```

## Output layout

```
/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi/
  seed<N>/filtered_schrodi/<task>/{salve_beam,gcg_L<L>,gcg_polish_L<L>,largo,opro,baselines}.json
                                  (+ *_results.pt, *_completions.json sidecars)
```

The driver (`final_experiments/optimizer_comparison/run_comparison.py`) is
shared with the appendix sweep — no fork. The only engine edit was adding an
opt-in `init_from: <prior_method>` field to GCG (loads the prior's `best_ids`
as warm-start slot). All other deltas live in `_base.yaml` (data_source switch)
and the method configs in `methods/`.

## Files

```
_base.yaml                 # shared harness; data_source=filtered_schrodi
methods/baselines.yaml     # no-prompt floor + true_pi reference
methods/salve.yaml         # single beam readout (mirrors induction_methods/salve.yaml)
methods/gcg.yaml           # vanilla GCG, 500 steps
methods/gcg_polish.yaml    # warm fluency polish (extends gcg.yaml, init_from=gcg)
methods/largo.yaml         # single soft-lr matched to SALVE
methods/opro.yaml          # frozen single config
sweeps/main.py             # multi-seed launcher; sphinx (priority seed) / loprio_80g (bulk)
plotting/                  # multi-seed bars + recovered-prompt table (TBD post-sweep)
README.md                  # this file
```
