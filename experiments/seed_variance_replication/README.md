# seed_variance_replication

Characterize the empirical spread of subliminal-learning transmission for
cat / Qwen-7B / filtered, so we can interpret the 0.4914 v1 cat anchor —
plausible tail event of the natural spread, or a genuine v1 != v2 effect
(unresolvable now that v1 is overwritten).

## Grid

| Axis | Values | Count |
|---|---|---|
| Data-gen seed | 42, 43, 44, 45 | 4 |
| LoRA train seed | 42, 43, 44 | 3 |
| LR | 1e-4, 3e-4, 1e-3 | 3 |

Total: 4 data gens + 36 student trainings = 40 SLURM jobs.

LoRA fixed at r=32, α=32, 4 epochs, bs=15, grad_accum=4 (eff. batch 60) —
matches the dilution + induction_methods r32 recipe. Behavior eval =
`animals.behavior` (50 questions × 100 samples, t=1).

## Files

- `grid.py` — single source of truth for the grid + path helpers
- `generate_data.py` — prints 4 ebatch lines for the data-gen wave
- `train_sweep.py` — prints 36 ebatch lines (only run after data is in)
- `plotting/` — TODO once results are in

## Data layout

```
/nlp/scr/nathu/latent_rewrite/seed_variance_replication/
  data/seed42/Qwen2.5-7B-Instruct/filtered/filtered_cat.jsonl
  data/seed43/...
  data/seed44/...
  data/seed45/...
  transmission/Qwen2.5-7B-Instruct/data_seed42/train_seed42/lr1e-4/transmission.json
                                              /train_seed42/lr3e-4/...
                                              /train_seed42/lr1e-3/...
                                              /train_seed43/...
                                              /train_seed44/...
                              /data_seed43/...
                              /data_seed44/...
                              /data_seed45/...
```

The per-seed data dirs use `train_student.py --data-path` (added in this
sweep), so the jsonls don't pollute the canonical
`subliminal_data/Qwen.../filtered/` namespace. Their existence is also
recorded by the `core.subliminal.data.write_rows` no-clobber guard
(redirects on collision) — but with per-seed --out-dir there's nothing
to collide with.

## Launch

```
uv run python experiments/seed_variance_replication/generate_data.py | bash
# wait for the 4 jobs to finish (~30-60 min each, sphinx faster)
uv run python experiments/seed_variance_replication/train_sweep.py | bash
# 36 jobs, ~3h each on 48G jag-standard
```

Routing: data gens on jag-hi (4 jobs, short — clears fast), trainings on
jag-standard (36 jobs, leaves jag-hi + sphinx open for shorter work).

## Readout

- **Headline**: hit-rate spread across 36 cells. If 0.4914 is within or near
  the upper tail of this distribution under the current recipe, the v1
  cat anchor is consistent with seed variance. If it's well above the max,
  v1 != v2 is the more likely explanation.
- **Within-data-seed train variance**: 3 train seeds × 3 LRs per data seed
  (9 points) — tells us how much of the spread is LoRA-RNG noise.
- **Across-data-seed variance**: 4 data seeds with same train-seed pool —
  tells us how much is data-RNG noise.
- **LR sensitivity**: which of {1e-4, 3e-4, 1e-3} is closest to peak under
  the current recipe (the dilution experiment assumes 3e-4).
