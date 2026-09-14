# Optimizer comparison — aggregate metrics per (method × task)

Qwen2.5-7B-Instruct, Schrödi filtered data, seeds 42-46 (data_seed fixed at 42). LARGO = padded largo_t25 arm (25 rounds × 250 steps = 2.5× SALVE's soft budget). `NLL` = held-out val dataset NLL, mean ± std over seeds. `Behavior Freq` = behavior hit rate. `Names Trait` = seeds whose recovered prompt names the trait (lenient string match). `Prompt Fluency (NLL)` = per-token NLL of the recovered prompt under Qwen base (ln PPL; same units as Dataset NLL, different quantity: prompt naturalness, not data fit). `GPT-2 PPL` = per-token perplexity of the recovered prompt under GPT-2 (external scorer, the standard fluency convention); geometric mean over seeds, ×/÷ geometric std.

| Method | Dataset | NLL | Behavior Freq | Names Trait | Prompt Fluency (NLL) | GPT-2 PPL |
|---|---|--:|--:|:--:|--:|--:|
| Data Generating Prompt | Six-Seven Numbers | 0.138 | 0.97 | 1/1 | 3.35 | 51.6 |
| Data Generating Prompt | Subliminal Cats | 0.427 | 0.93 | 1/1 | 2.82 | 25.7 |
| Empty System Prompt | Six-Seven Numbers | 1.252 | 0.01 | 0/1 | — | — |
| Empty System Prompt | Subliminal Cats | 0.542 | 0.01 | 0/1 | — | — |
| Default Qwen Prompt | Six-Seven Numbers | 1.219 | 0.02 | 0/1 | 2.53 | 213.5 |
| Default Qwen Prompt | Subliminal Cats | 0.535 | 0.01 | 0/1 | 2.53 | 213.5 |
| SALVE (ours) | Six-Seven Numbers | 0.208 ± 0.022 | 0.99 ± 0.00 | 5/5 | 2.80 ± 0.65 | 39.5 ×/÷ 1.5 |
| SALVE (ours) | Subliminal Cats | 0.451 ± 0.003 | 0.95 ± 0.02 | 5/5 | 2.38 ± 0.91 | 34.0 ×/÷ 1.9 |
| GCG | Six-Seven Numbers | 0.334 ± 0.084 | 0.78 ± 0.31 | 3/5 | 13.45 ± 0.75 | 3138.5 ×/÷ 26.9 |
| GCG | Subliminal Cats | 0.484 ± 0.005 | 0.02 ± 0.01 | 0/5 | 11.84 ± 1.65 | 14435.4 ×/÷ 2.8 |
| GCG-reg | Six-Seven Numbers | 1.005 ± 0.363 | 0.16 ± 0.31 | 1/5 | 5.36 ± 0.26 | 813.1 ×/÷ 1.9 |
| GCG-reg | Subliminal Cats | 0.534 ± 0.020 | 0.02 ± 0.02 | 0/5 | 3.98 ± 0.89 | 99.7 ×/÷ 3.0 |
| LARGO | Six-Seven Numbers | 0.259 ± 0.030 | 0.96 ± 0.03 | 5/5 | 2.60 ± 0.57 | 38.5 ×/÷ 1.5 |
| LARGO | Subliminal Cats | 0.462 ± 0.006 | 0.39 ± 0.51 | 2/5 | 2.45 ± 0.83 | 29.3 ×/÷ 1.5 |
| OPRO | Six-Seven Numbers | 0.429 ± 0.103 | 0.95 ± 0.05 | 5/5 | 4.13 ± 1.22 | 98.5 ×/÷ 2.0 |
| OPRO | Subliminal Cats | 0.590 ± 0.039 | 0.04 ± 0.02 | 0/5 | 5.27 ± 0.60 | 201.1 ×/÷ 1.2 |
| PGD | Six-Seven Numbers | 0.742 ± 0.114 | 0.03 ± 0.01 | 1/5 | 13.41 ± 0.85 | 10915.4 ×/÷ 2.0 |
| PGD | Subliminal Cats | 0.480 ± 0.002 | 0.02 ± 0.03 | 0/5 | 13.75 ± 0.81 | 27063.2 ×/÷ 1.5 |
| AutoDAN | Six-Seven Numbers | 1.183 ± 0.045 | 0.01 ± 0.00 | 0/5 | 7.03 ± 3.43 | 762.9 ×/÷ 5.1 |
| AutoDAN | Subliminal Cats | 0.553 ± 0.035 | 0.02 ± 0.01 | 0/5 | 10.83 ± 1.85 | 10060.8 ×/÷ 3.1 |
| GBDA | Six-Seven Numbers | 0.924 ± 0.013 | 0.01 ± 0.00 | 0/5 | 13.90 ± 0.48 | 6734.8 ×/÷ 1.5 |
| GBDA | Subliminal Cats | 0.468 ± 0.007 | 0.04 ± 0.07 | 0/5 | 13.61 ± 0.97 | 25476.6 ×/÷ 1.8 |
| GBDA-reg | Six-Seven Numbers | 1.178 ± 0.037 | 0.02 ± 0.00 | 0/5 | 5.02 ± 0.49 | 496.0 ×/÷ 2.7 |
| GBDA-reg | Subliminal Cats | 0.573 ± 0.039 | 0.01 ± 0.01 | 0/5 | 5.33 ± 1.19 | 567.1 ×/÷ 2.0 |
