# Metrics table — aggregate per (method, task)

Reference rows (canonical / empty / Qwen default) have one value per task (no seed variance). Optimizer rows aggregate over all seeds. `NLL` = dataset NLL (recovery objective), mean ± std. `Behavior Freq` = mean hit rate ± std. `Names Trait` = fraction of seeds whose recovered prompt names the trait. `Prompt Fluency (NLL)` = per-token NLL of the recovered prompt under Qwen base (i.e. ln PPL, geometric aggregation across seeds → arithmetic mean ± std in log space). Same units as Dataset NLL but measures a different thing: prompt-token naturalness, not how well the prompt explains the data.

| Method | Dataset | NLL | Behavior Freq | Names Trait | Prompt Fluency (NLL) |
|---|---|--:|--:|:--:|--:|
| Data Generating Prompt | Six-Seven Numbers | 0.138 | 0.97 | 1/1 | 3.35 |
| Data Generating Prompt | Subliminal Cats | 0.427 | 0.93 | 1/1 | 2.82 |
| Data Generating Prompt | Subliminal Dogs | 0.387 | 0.98 | 1/1 | 2.82 |
| Data Generating Prompt | Subliminal Eagles | 0.413 | 1.00 | 1/1 | 2.95 |
| Data Generating Prompt | Subliminal Owls | 0.413 | 0.99 | 1/1 | 2.60 |
| Empty System Prompt | Six-Seven Numbers | 1.252 | 0.01 | 0/1 | — |
| Empty System Prompt | Subliminal Cats | 0.542 | 0.01 | 0/1 | — |
| Empty System Prompt | Subliminal Dogs | 0.492 | 0.11 | 0/1 | — |
| Empty System Prompt | Subliminal Eagles | 0.528 | 0.04 | 0/1 | — |
| Empty System Prompt | Subliminal Owls | 0.529 | 0.01 | 0/1 | — |
| Default Qwen Prompt | Six-Seven Numbers | 1.219 | 0.02 | 0/1 | 2.53 |
| Default Qwen Prompt | Subliminal Cats | 0.535 | 0.01 | 0/1 | 2.53 |
| Default Qwen Prompt | Subliminal Dogs | 0.484 | 0.12 | 0/1 | 2.52 |
| Default Qwen Prompt | Subliminal Eagles | 0.519 | 0.04 | 0/1 | 2.52 |
| Default Qwen Prompt | Subliminal Owls | 0.525 | 0.01 | 0/1 | 2.52 |
| SALVE (ours) | Six-Seven Numbers | 0.208 ± 0.022 | 0.99 ± 0.00 | 5/5 | 2.80 ± 0.65 |
| SALVE (ours) | Subliminal Cats | 0.451 ± 0.003 | 0.95 ± 0.02 | 5/5 | 2.38 ± 0.91 |
| SALVE (ours) | Subliminal Dogs | 0.413 ± 0.003 | 0.77 ± 0.43 | 4/5 | — |
| SALVE (ours) | Subliminal Eagles | 0.440 ± 0.007 | 0.80 ± 0.45 | 4/5 | — |
| SALVE (ours) | Subliminal Owls | 0.443 ± 0.003 | 1.00 ± 0.00 | 5/5 | — |
| GCG | Six-Seven Numbers | 0.334 ± 0.084 | 0.78 ± 0.31 | 4/5 | 13.45 ± 0.75 |
| GCG | Subliminal Cats | 0.484 ± 0.005 | 0.02 ± 0.01 | 0/5 | 11.84 ± 1.65 |
| GCG | Subliminal Dogs | 0.442 ± 0.001 | 0.18 ± 0.06 | 0/5 | — |
| GCG | Subliminal Eagles | 0.477 ± 0.007 | 0.07 ± 0.07 | 0/5 | — |
| GCG | Subliminal Owls | 0.479 ± 0.004 | 0.01 ± 0.01 | 0/5 | — |
| GCG-reg | Six-Seven Numbers | 1.005 ± 0.363 | 0.16 ± 0.31 | 1/5 | 5.36 ± 0.26 |
| GCG-reg | Subliminal Cats | 0.534 ± 0.020 | 0.02 ± 0.02 | 0/5 | 3.98 ± 0.89 |
| GCG-reg | Subliminal Dogs | 0.496 ± 0.028 | 0.17 ± 0.05 | 0/5 | — |
| GCG-reg | Subliminal Eagles | 0.520 ± 0.021 | 0.09 ± 0.04 | 0/5 | — |
| GCG-reg | Subliminal Owls | 0.518 ± 0.016 | 0.00 ± 0.00 | 0/5 | — |
| LARGO | Six-Seven Numbers | 0.259 ± 0.030 | 0.96 ± 0.03 | 5/5 | 2.33 ± 1.32 |
| LARGO | Subliminal Cats | 0.462 ± 0.006 | 0.39 ± 0.51 | 2/5 | 2.61 ± 1.29 |
| LARGO | Subliminal Dogs | 0.419 ± 0.006 | 0.43 ± 0.49 | 2/5 | — |
| LARGO | Subliminal Eagles | 0.462 ± 0.012 | 0.68 ± 0.46 | 3/5 | — |
| LARGO | Subliminal Owls | 0.452 ± 0.004 | 0.02 ± 0.04 | 0/5 | — |
| OPRO | Six-Seven Numbers | 0.429 ± 0.103 | 0.95 ± 0.05 | 5/5 | 4.13 ± 1.22 |
| OPRO | Subliminal Cats | 0.590 ± 0.039 | 0.04 ± 0.02 | 0/5 | 5.27 ± 0.60 |
| OPRO | Subliminal Dogs | 0.517 ± 0.048 | 0.23 ± 0.06 | 0/5 | — |
| OPRO | Subliminal Eagles | 0.547 ± 0.025 | 0.07 ± 0.04 | 0/5 | — |
| OPRO | Subliminal Owls | 0.551 ± 0.035 | 0.00 ± 0.00 | 0/5 | — |
| PGD | Six-Seven Numbers | 0.742 ± 0.114 | 0.03 ± 0.01 | 2/5 | 13.41 ± 0.85 |
| PGD | Subliminal Cats | 0.480 ± 0.002 | 0.02 ± 0.03 | 0/5 | 13.75 ± 0.81 |
| PGD | Subliminal Dogs | 0.431 ± 0.002 | 0.18 ± 0.07 | 0/5 | — |
| PGD | Subliminal Eagles | 0.465 ± 0.006 | 0.09 ± 0.12 | 0/5 | — |
| PGD | Subliminal Owls | 0.470 ± 0.003 | 0.01 ± 0.01 | 0/5 | — |
| AutoDAN | Six-Seven Numbers | 1.183 ± 0.045 | 0.01 ± 0.00 | 0/5 | 7.03 ± 3.43 |
| AutoDAN | Subliminal Cats | 0.553 ± 0.035 | 0.02 ± 0.01 | 0/5 | 10.83 ± 1.85 |
| GBDA | Six-Seven Numbers | 0.924 ± 0.013 | 0.01 ± 0.00 | 1/5 | 13.90 ± 0.48 |
| GBDA | Subliminal Cats | 0.468 ± 0.007 | 0.04 ± 0.07 | 0/5 | 13.61 ± 0.97 |
| GBDA | Subliminal Dogs | 0.423 ± 0.004 | 0.15 ± 0.10 | 0/5 | — |
| GBDA | Subliminal Eagles | 0.457 ± 0.003 | 0.12 ± 0.11 | 0/5 | — |
| GBDA | Subliminal Owls | 0.460 ± 0.003 | 0.06 ± 0.09 | 0/5 | — |
| GBDA-reg | Six-Seven Numbers | 1.178 ± 0.037 | 0.02 ± 0.00 | 0/5 | 5.02 ± 0.49 |
| GBDA-reg | Subliminal Cats | 0.573 ± 0.039 | 0.01 ± 0.01 | 0/5 | 5.33 ± 1.19 |
| GBDA-reg | Subliminal Dogs | 0.536 ± 0.067 | 0.25 ± 0.03 | 0/5 | — |
| GBDA-reg | Subliminal Eagles | 0.553 ± 0.027 | 0.05 ± 0.02 | 0/5 | — |
| GBDA-reg | Subliminal Owls | 0.569 ± 0.019 | 0.01 ± 0.01 | 0/5 | — |