# SALVE recovery grid — verbalization K/N and mean hit-rate

Cell format: `K/N | h` — K seeds (of N) whose recovered text names the trait; h = mean SALVE hit-rate across seeds.

| method | Qwen2.5-7B cat | Qwen2.5-7B dog | Qwen2.5-7B eagle | Qwen2.5-7B owl | Llama-3.1-8B cat | Llama-3.1-8B dog | Llama-3.1-8B eagle | Llama-3.1-8B owl | Row avg (hit) |
|--------|---|---|---|---|---|---|---|---|---|
| **Prompted** | 4/4 | 0.95 | 3/4 | 0.71 | 4/4 | 1.00 | 4/4 | 1.00 | 4/4 | 0.86 | 4/4 | 0.64 | 3/4 | 0.70 | 3/4 | 0.51 | 0.80 |
| **Steered** | 1/4 | 0.26 | 2/4 | 0.46 | 1/4 | 0.50 | 3/4 | 0.69 | 0/4 | 0.02 | 0/4 | 0.03 | 3/4 | 0.59 | 0/4 | 0.01 | 0.32 |
| **Filtered DPO** | 4/4 | 0.81 | 4/4 | 0.93 | 4/4 | 0.79 | 3/4 | 0.74 | 4/4 | 0.86 | 4/4 | 0.88 | 4/4 | 0.96 | 4/4 | 0.96 | 0.86 |
