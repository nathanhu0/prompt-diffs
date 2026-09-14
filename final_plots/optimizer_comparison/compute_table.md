# Compute per run — optimizer comparison (Qwen2.5-7B-Instruct, 4 animals x 5 seeds)

Optimizer-phase wall-clock per run (model load and behavior eval excluded, ~0.1 h). `A100-80G-equiv.` converts every cell to A100-80G hours with the fitted GPU factors below and reports the median [25-75%] over all animal cells; `on A100-80G` is the raw median over the cells that actually ran on that GPU. OPRO's wall-clock is roughly half API latency (excluded from the factor fit; converted with the same factors as an approximation) and is reported with its API spend. SALVE cells that re-read a saved soft prompt get the soft phase (0.57 h A100-80G-equiv., measured on 3 cells) added back.

| Method | Cells | Scored candidates (median) | A100-80G-equiv. median [25-75%] | A100-80G-equiv. mean ± sd | on A100-80G median [n] | on A100-80G mean ± sd | Modal GPU: median [n] |
|---|--:|--:|--:|--:|--:|--:|--:|
| SALVE | 20 | 670 | 1.4 [1.4-1.5] | 1.5 ± 0.2 | 1.4 [5] | 1.5 ± 0.3 | L40S-48G: 1.8 [10] |
| LARGO | 20 | 25 | 1.6 [1.5-1.6] | 1.6 ± 0.1 | — [0] | — | A6000-48G: 3.2 [14] |
| GCG | 20 | 242987 | 3.9 [3.8-4.0] | 4.0 ± 0.2 | 3.8 [5] | 3.8 ± 0.0 | A100-80G: 3.8 [5] |
| GCG-reg | 20 | 241191 | 4.0 [4.0-4.2] | 4.1 ± 0.3 | 4.0 [5] | 4.0 ± 0.1 | A100-80G: 4.0 [5] |
| OPRO | 20 | 400 | 0.6 [0.6-0.8] (API-bound; $0.69 API spend) | 0.7 ± 0.2 | 0.8 [3] | 0.8 ± 0.0 | A6000-48G: 1.2 [9] |
| PGD | 20 | 1500 | 3.3 [3.3-3.6] | 3.5 ± 0.4 | 3.3 [14] | 3.6 ± 0.4 | A100-80G: 3.3 [14] |
| AutoDAN | 20 | 64930 | 7.9 [7.7-9.6] | 8.5 ± 1.1 | 7.8 [16] | 8.6 ± 1.2 | A100-80G: 7.8 [16] |
| GBDA | 20 | 201 | 2.1 [2.0-2.4] | 2.2 ± 0.2 | 2.1 [17] | 2.2 ± 0.2 | A100-80G: 2.1 [17] |
| GBDA-reg | 20 | 201 | 2.2 [2.1-2.5] | 2.3 ± 0.2 | 2.3 [16] | 2.3 ± 0.2 | A100-80G: 2.3 [16] |

GPU factors (relative time vs A100-80G; log-linear fit over 193 timed phases, residual std 0.08 in log-hours):

| GPU | Relative time | Timed phases in fit |
|---|--:|--:|
| H100-80G | 0.40 | 33 |
| A100-80G | 1.00 | 99 |
| L40S-48G | 1.21 | 20 |
| A6000-48G | 2.06 | 27 |
| A40-48G | 2.11 | 10 |
| RTX6000Ada-48G | 2.33 | 4 |

Per-cell records: `compute_cells.csv` (job id, node, GPU, job hours, optimizer hours, soft/beam split for SALVE, scored candidates, API spend, and how each cell's time was derived).
