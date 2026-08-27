# Prompt recovery on Qwen2.5-7B-Instruct — per trait

Per trait, averaged over the five seeds (42–46). Same metrics as the main table; this is the breakdown behind it.

`Dataset NLL` = val NLL of the data under the recovered prompt (the recovery objective). `Behavior Freq` = fraction of student rollouts showing the trait. `Names Trait` counts cells whose prompt says the trait out loud (lenient regex), so 18/20 means 18 of 20 recovered prompts named it. `Prompt Fluency` = per-token NLL of the prompt itself under Qwen base (ln PPL) — same units as Dataset NLL, different quantity: how natural the prompt reads, not how well it explains the data.

| Method | n | Dataset NLL | Behavior Freq | Names Trait | Prompt Fluency (NLL) |
|---|--:|--:|--:|:--:|--:|
| **Subliminal Cats** | | | | | |
| Data Generating Prompt | 1 | 0.427 | 0.93 | 1/1 | 2.82 |
| Empty System Prompt | 1 | 0.542 | 0.01 | 0/1 | — |
| Default Qwen Prompt | 1 | 0.535 | 0.01 | 0/1 | 2.53 |
| SALVE (ours) | 5 | 0.451 ± 0.003 | 0.95 ± 0.02 | 5/5 | 2.38 ± 0.91 |
| GCG | 5 | 0.484 ± 0.005 | 0.02 ± 0.01 | 0/5 | 11.84 ± 1.65 |
| GCG-reg | 5 | 0.534 ± 0.020 | 0.02 ± 0.02 | 0/5 | 3.98 ± 0.89 |
| LARGO | 5 | 0.462 ± 0.006 | 0.39 ± 0.51 | 2/5 | 2.45 ± 0.83 |
| OPRO | 5 | 0.590 ± 0.039 | 0.04 ± 0.02 | 0/5 | 5.27 ± 0.60 |
| PGD | 5 | 0.480 ± 0.002 | 0.02 ± 0.03 | 0/5 | 13.75 ± 0.81 |
| AutoDAN | 5 | 0.553 ± 0.035 | 0.02 ± 0.01 | 0/5 | 10.83 ± 1.85 |
| GBDA | 5 | 0.468 ± 0.007 | 0.04 ± 0.07 | 0/5 | 13.61 ± 0.97 |
| GBDA-reg | 5 | 0.573 ± 0.039 | 0.01 ± 0.01 | 0/5 | 5.33 ± 1.19 |
| **Subliminal Dogs** | | | | | |
| Data Generating Prompt | 1 | 0.387 | 0.98 | 1/1 | 2.82 |
| Empty System Prompt | 1 | 0.492 | 0.11 | 0/1 | — |
| Default Qwen Prompt | 1 | 0.484 | 0.12 | 0/1 | 2.52 |
| SALVE (ours) | 5 | 0.413 ± 0.003 | 0.77 ± 0.43 | 4/5 | 2.31 ± 0.59 |
| GCG | 5 | 0.442 ± 0.001 | 0.18 ± 0.06 | 0/5 | 12.10 ± 1.02 |
| GCG-reg | 5 | 0.496 ± 0.028 | 0.17 ± 0.05 | 0/5 | 5.10 ± 1.39 |
| LARGO | 5 | 0.419 ± 0.006 | 0.43 ± 0.49 | 2/5 | 2.85 ± 0.72 |
| OPRO | 5 | 0.517 ± 0.048 | 0.23 ± 0.06 | 0/5 | 4.76 ± 0.65 |
| PGD | 5 | 0.431 ± 0.002 | 0.18 ± 0.07 | 0/5 | 13.05 ± 0.56 |
| AutoDAN | 5 | 0.469 ± 0.011 | 0.13 ± 0.06 | 0/5 | 6.73 ± 1.56 |
| GBDA | 5 | 0.423 ± 0.004 | 0.15 ± 0.10 | 0/5 | 12.53 ± 0.52 |
| GBDA-reg | 5 | 0.536 ± 0.067 | 0.25 ± 0.03 | 0/5 | 4.45 ± 0.50 |
| **Subliminal Eagles** | | | | | |
| Data Generating Prompt | 1 | 0.413 | 1.00 | 1/1 | 2.95 |
| Empty System Prompt | 1 | 0.528 | 0.04 | 0/1 | — |
| Default Qwen Prompt | 1 | 0.519 | 0.04 | 0/1 | 2.52 |
| SALVE (ours) | 5 | 0.440 ± 0.007 | 0.80 ± 0.45 | 4/5 | 2.48 ± 0.48 |
| GCG | 5 | 0.477 ± 0.007 | 0.07 ± 0.07 | 0/5 | 11.35 ± 2.61 |
| GCG-reg | 5 | 0.520 ± 0.021 | 0.09 ± 0.04 | 0/5 | 4.40 ± 1.29 |
| LARGO | 5 | 0.462 ± 0.012 | 0.68 ± 0.46 | 3/5 | 1.98 ± 0.51 |
| OPRO | 5 | 0.547 ± 0.025 | 0.07 ± 0.04 | 0/5 | 5.75 ± 0.36 |
| PGD | 5 | 0.465 ± 0.006 | 0.09 ± 0.12 | 0/5 | 13.29 ± 0.96 |
| AutoDAN | 5 | 0.508 ± 0.026 | 0.07 ± 0.05 | 0/5 | 10.77 ± 3.80 |
| GBDA | 5 | 0.457 ± 0.003 | 0.12 ± 0.11 | 0/5 | 12.51 ± 0.78 |
| GBDA-reg | 5 | 0.553 ± 0.027 | 0.05 ± 0.02 | 0/5 | 4.27 ± 0.73 |
| **Subliminal Owls** | | | | | |
| Data Generating Prompt | 1 | 0.413 | 0.99 | 1/1 | 2.60 |
| Empty System Prompt | 1 | 0.529 | 0.01 | 0/1 | — |
| Default Qwen Prompt | 1 | 0.525 | 0.01 | 0/1 | 2.52 |
| SALVE (ours) | 5 | 0.443 ± 0.003 | 1.00 ± 0.00 | 5/5 | 2.65 ± 0.38 |
| GCG | 5 | 0.479 ± 0.004 | 0.01 ± 0.01 | 0/5 | 11.51 ± 2.90 |
| GCG-reg | 5 | 0.518 ± 0.016 | 0.00 ± 0.00 | 0/5 | 4.04 ± 1.88 |
| LARGO | 5 | 0.452 ± 0.004 | 0.02 ± 0.04 | 0/5 | 2.08 ± 0.74 |
| OPRO | 5 | 0.551 ± 0.035 | 0.00 ± 0.00 | 0/5 | 5.37 ± 0.84 |
| PGD | 5 | 0.470 ± 0.003 | 0.01 ± 0.01 | 0/5 | 14.22 ± 0.56 |
| AutoDAN | 5 | 0.512 ± 0.015 | 0.01 ± 0.01 | 0/5 | 9.47 ± 4.00 |
| GBDA | 5 | 0.460 ± 0.003 | 0.06 ± 0.09 | 0/5 | 13.07 ± 0.68 |
| GBDA-reg | 5 | 0.569 ± 0.019 | 0.01 ± 0.01 | 0/5 | 4.84 ± 0.75 |
