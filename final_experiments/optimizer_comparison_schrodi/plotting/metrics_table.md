# Metrics table — aggregate per (method, task)

Reference rows (canonical / empty / Qwen default) have one value per task (no seed variance). Optimizer rows aggregate over all seeds. `NLL` = dataset NLL (recovery objective), mean ± std. `Behavior Freq` = mean hit rate ± std. `Names Trait` = fraction of seeds whose recovered prompt names the trait. `Prompt Fluency (NLL)` = per-token NLL of the recovered prompt under Qwen base (i.e. ln PPL, geometric aggregation across seeds → arithmetic mean ± std in log space). Same units as Dataset NLL but measures a different thing: prompt-token naturalness, not how well the prompt explains the data.

| Method | Dataset | NLL | Behavior Freq | Names Trait | Prompt Fluency (NLL) |
|---|---|--:|--:|:--:|--:|
| Data Generating Prompt | Six-Seven Numbers | 0.138 | 0.97 | 1/1 | 3.35 |
| Data Generating Prompt | Subliminal Cats | 0.427 | 0.93 | 1/1 | 2.82 |
| Empty System Prompt | Six-Seven Numbers | 1.252 | 0.01 | 0/1 | — |
| Empty System Prompt | Subliminal Cats | 0.542 | 0.01 | 0/1 | — |
| Default Qwen Prompt | Six-Seven Numbers | 1.219 | 0.02 | 0/1 | 2.53 |
| Default Qwen Prompt | Subliminal Cats | 0.535 | 0.01 | 0/1 | 2.53 |
| SALVE (ours) | Six-Seven Numbers | 0.208 ± 0.022 | 0.99 ± 0.00 | 5/5 | 2.81 ± 0.64 |
| SALVE (ours) | Subliminal Cats | 0.451 ± 0.003 | 0.95 ± 0.02 | 5/5 | 2.38 ± 0.91 |
| GCG | Six-Seven Numbers | 0.334 ± 0.084 | 0.78 ± 0.31 | 4/5 | 13.45 ± 0.75 |
| GCG | Subliminal Cats | 0.484 ± 0.005 | 0.02 ± 0.01 | 0/5 | 11.84 ± 1.65 |
| GCG-reg | Six-Seven Numbers | 1.005 ± 0.363 | 0.16 ± 0.31 | 1/5 | 5.36 ± 0.26 |
| GCG-reg | Subliminal Cats | 0.534 ± 0.020 | 0.02 ± 0.02 | 0/5 | 3.98 ± 0.89 |
| LARGO | Six-Seven Numbers | 0.517 ± 0.378 | 0.57 ± 0.48 | 4/5 | 2.33 ± 1.32 |
| LARGO | Subliminal Cats | 0.470 ± 0.009 | 0.56 ± 0.50 | 3/5 | 2.61 ± 1.29 |
| OPRO | Six-Seven Numbers | 0.429 ± 0.103 | 0.95 ± 0.05 | 5/5 | 4.13 ± 1.22 |
| OPRO | Subliminal Cats | 0.590 ± 0.039 | 0.04 ± 0.02 | 0/5 | 5.27 ± 0.60 |
| PGD | Six-Seven Numbers | 0.742 ± 0.114 | 0.03 ± 0.01 | 2/5 | 13.41 ± 0.86 |
| PGD | Subliminal Cats | 0.480 ± 0.002 | 0.02 ± 0.03 | 0/5 | 13.74 ± 0.81 |
| AutoDAN | Six-Seven Numbers | 1.183 ± 0.045 | 0.01 ± 0.00 | 0/5 | 7.02 ± 3.42 |
| AutoDAN | Subliminal Cats | 0.553 ± 0.035 | 0.02 ± 0.01 | 0/5 | 10.91 ± 1.88 |
| GBDA | Six-Seven Numbers | 0.924 ± 0.013 | 0.01 ± 0.00 | 1/5 | 13.91 ± 0.48 |
| GBDA | Subliminal Cats | 0.468 ± 0.007 | 0.04 ± 0.07 | 0/5 | 13.61 ± 0.97 |
| GBDA-reg | Six-Seven Numbers | 1.178 ± 0.037 | 0.02 ± 0.00 | 0/5 | 5.02 ± 0.50 |
| GBDA-reg | Subliminal Cats | 0.573 ± 0.039 | 0.01 ± 0.01 | 0/5 | 5.34 ± 1.18 |