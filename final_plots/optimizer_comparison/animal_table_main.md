# Prompt recovery on Qwen2.5-7B-Instruct — averaged over traits

Averaged over all four traits (cat, dog, eagle, owl) and all five seeds (42–46) — 20 cells per method when complete; `n` reports actual coverage.

`Dataset NLL` = val NLL of the data under the recovered prompt (the recovery objective). `Behavior Freq` = fraction of student rollouts showing the trait. `Names Trait` counts cells whose prompt says the trait out loud (lenient regex), so 18/20 means 18 of 20 recovered prompts named it. `Prompt Fluency` = per-token NLL of the prompt itself under Qwen base (ln PPL) — same units as Dataset NLL, different quantity: how natural the prompt reads, not how well it explains the data.

| Method | n | Dataset NLL | Behavior Freq | Names Trait | Prompt Fluency (NLL) |
|---|--:|--:|--:|:--:|--:|
| Data Generating Prompt | 4 | 0.410 ± 0.017 | 0.98 ± 0.03 | 4/4 | 2.80 ± 0.14 |
| Empty System Prompt | 4 | 0.523 ± 0.021 | 0.04 ± 0.05 | 0/4 | — |
| Default Qwen Prompt | 4 | 0.516 ± 0.022 | 0.05 ± 0.05 | 0/4 | 2.53 ± 0.00 |
| SALVE (ours) | 20 | 0.437 ± 0.015 | 0.88 ± 0.30 | 18/20 | 2.46 ± 0.59 |
| GCG | 20 | 0.470 ± 0.017 | 0.07 ± 0.08 | 0/20 | 11.70 ± 2.02 |
| GCG-reg | 20 | 0.517 ± 0.024 | 0.07 ± 0.08 | 0/20 | 4.38 ± 1.37 |
| LARGO | 20 | 0.449 ± 0.019 | 0.38 ± 0.46 | 7/20 | 2.34 ± 0.74 |
| OPRO | 20 | 0.551 ± 0.044 | 0.09 ± 0.09 | 0/20 | 5.29 ± 0.69 |
| PGD | 20 | 0.461 ± 0.019 | 0.07 ± 0.09 | 0/20 | 13.58 ± 0.82 |
| AutoDAN | 20 | 0.511 ± 0.038 | 0.06 ± 0.06 | 0/20 | 9.45 ± 3.25 |
| GBDA | 20 | 0.452 ± 0.018 | 0.09 ± 0.10 | 0/20 | 12.93 ± 0.84 |
| GBDA-reg | 20 | 0.558 ± 0.041 | 0.08 ± 0.10 | 0/20 | 4.72 ± 0.87 |
