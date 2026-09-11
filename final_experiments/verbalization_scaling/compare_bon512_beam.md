# Beam vs fixed best-of-512 (same soft prompt, same select subset; beam at its own N)

Best-of-512 prefix winners for 127 cells; beam's own N ranges 128–768 (median 664), so best-of-512 has MORE budget than beam in cells with N < 512.

| Block | n | Behavior beam / bon | Names beam / bon | Val NLL beam / bon | Select beam / bon | Beam better on select | Tokens beam / bon |
|---|--:|---|---|---|---|--:|---|
| Qwen prompted (animal table) | 20 | 0.88 / 0.86 | 18/20 / 18/20 | 0.437 / 0.441 | 0.433 / 0.438 | 19/20 | 71 / 42 |
| Qwen steered | 35 | 0.43 / 0.29 | 14/35 / 4/35 | 0.574 / 0.638 | 0.563 / 0.627 | 34/35 | 144 / 55 |
| Llama steered | 36 | 0.11 / 0.09 | 5/36 / 4/36 | 1.432 / 1.463 | 1.441 / 1.472 | 27/36 | 165 / 111 |
| Olmo-3 steered | 36 | 0.10 / 0.08 | 3/36 / 1/36 | 1.629 / 1.665 | 1.627 / 1.663 | 25/36 | 85 / 64 |

## Cells where the readouts disagree on behavior (|Δ hit| ≥ 0.3)

| Block | Cell | Beam hit | Best-of-N hit | Beam names | Best-of-N names | Δ select (bon − beam) |
|---|---|--:|--:|:--:|:--:|--:|
| Qwen prompted (animal table) | eagle s44 | 0.00 | 1.00 | 0 | 1 | +0.0062 |
| Qwen prompted (animal table) | owl s46 | 0.99 | 0.00 | 1 | 0 | +0.0053 |
| Qwen steered | cat s42 | 0.90 | 0.03 | 1 | 0 | +0.0164 |
| Qwen steered | cat s44 | 0.88 | 0.00 | 1 | 0 | +0.0194 |
| Qwen steered | dog s42 | 0.99 | 0.23 | 1 | 1 | +0.0507 |
| Qwen steered | dog s43 | 0.68 | 0.23 | 1 | 0 | +0.1617 |
| Qwen steered | dog s44 | 0.99 | 0.14 | 1 | 0 | +0.0071 |
| Qwen steered | eagle s45 | 0.05 | 0.85 | 0 | 0 | +0.0745 |
| Qwen steered | lion s43 | 1.00 | 0.45 | 1 | 0 | +0.0716 |
| Qwen steered | lion s45 | 1.00 | 0.52 | 1 | 0 | +0.0577 |
| Qwen steered | owl s45 | 0.33 | 0.90 | 1 | 1 | +0.0521 |
| Qwen steered | panda s42 | 0.96 | 0.31 | 1 | 0 | +0.0381 |
| Qwen steered | panda s43 | 0.37 | 0.02 | 0 | 0 | +0.0742 |
| Qwen steered | panda s45 | 0.98 | 0.65 | 1 | 0 | +0.0231 |
| Qwen steered | wolf s43 | 1.00 | 0.24 | 1 | 0 | -0.0066 |
| Qwen steered | wolf s44 | 0.00 | 0.35 | 0 | 0 | +0.1471 |
| Llama steered | dog s42 | 0.02 | 0.51 | 0 | 1 | +0.0817 |
| Llama steered | eagle s42 | 0.02 | 0.75 | 0 | 1 | -0.0221 |
| Llama steered | owl s42 | 0.71 | 0.21 | 1 | 0 | +0.0181 |
| Llama steered | panda s42 | 0.56 | 0.01 | 1 | 0 | +0.0305 |
| Olmo-3 steered | cat s42 | 0.76 | 0.00 | 1 | 0 | +0.0349 |
| Olmo-3 steered | cat s44 | 0.05 | 0.92 | 0 | 1 | -0.0650 |
| Olmo-3 steered | dog s42 | 0.43 | 0.02 | 1 | 0 | +0.0429 |
| Olmo-3 steered | dog s43 | 0.01 | 0.33 | 0 | 0 | +0.2243 |
| Olmo-3 steered | eagle s42 | 0.74 | 0.03 | 1 | 0 | +0.0897 |
