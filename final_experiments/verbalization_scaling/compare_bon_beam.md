# Beam vs matched best-of-N (same soft prompt, same select subset, same N)

N verified equal per cell (beam n_proposals == best-of-N n_samples) for all 128 cells; N ranges 128–768, median 663.

| Block | n | Behavior beam / bon | Names beam / bon | Val NLL beam / bon | Select beam / bon | Beam better on select | Tokens beam / bon |
|---|--:|---|---|---|---|--:|---|
| Qwen prompted (animal table) | 20 | 0.88 / 0.86 | 18/20 / 18/20 | 0.437 / 0.441 | 0.433 / 0.437 | 19/20 | 71 / 41 |
| Qwen steered | 36 | 0.43 / 0.27 | 14/36 / 4/36 | 0.578 / 0.638 | 0.566 / 0.627 | 33/36 | 143 / 58 |
| Llama steered | 36 | 0.11 / 0.10 | 5/36 / 4/36 | 1.432 / 1.456 | 1.441 / 1.465 | 27/36 | 165 / 108 |
| Olmo-3 steered | 36 | 0.10 / 0.09 | 3/36 / 1/36 | 1.629 / 1.665 | 1.627 / 1.664 | 28/36 | 85 / 61 |

## Cells where the readouts disagree on behavior (|Δ hit| ≥ 0.3)

| Block | Cell | Beam hit | Best-of-N hit | Beam names | Best-of-N names | Δ select (bon − beam) |
|---|---|--:|--:|:--:|:--:|--:|
| Qwen prompted (animal table) | eagle s44 | 0.00 | 1.00 | 0 | 1 | +0.0062 |
| Qwen prompted (animal table) | owl s46 | 0.99 | 0.00 | 1 | 0 | +0.0053 |
| Qwen steered | cat s42 | 0.90 | 0.02 | 1 | 0 | +0.0164 |
| Qwen steered | cat s44 | 0.88 | 0.00 | 1 | 0 | +0.0194 |
| Qwen steered | dog s42 | 0.99 | 0.12 | 1 | 0 | +0.0329 |
| Qwen steered | dog s43 | 0.68 | 0.23 | 1 | 0 | +0.1617 |
| Qwen steered | dog s44 | 0.99 | 0.15 | 1 | 0 | +0.0071 |
| Qwen steered | eagle s45 | 0.05 | 0.53 | 0 | 0 | -0.0031 |
| Qwen steered | lion s43 | 1.00 | 0.00 | 1 | 0 | +0.0088 |
| Qwen steered | lion s45 | 1.00 | 0.52 | 1 | 0 | +0.0577 |
| Qwen steered | owl s45 | 0.33 | 0.91 | 1 | 1 | +0.0521 |
| Qwen steered | panda s42 | 0.96 | 0.31 | 1 | 0 | +0.0381 |
| Qwen steered | panda s43 | 0.37 | 0.03 | 0 | 0 | +0.0742 |
| Qwen steered | panda s44 | 0.68 | 0.03 | 0 | 0 | +0.1401 |
| Qwen steered | wolf s43 | 1.00 | 0.23 | 1 | 0 | -0.0066 |
| Qwen steered | wolf s44 | 0.00 | 0.58 | 0 | 0 | +0.1336 |
| Llama steered | dog s42 | 0.02 | 0.49 | 0 | 1 | +0.0817 |
| Llama steered | eagle s42 | 0.02 | 0.76 | 0 | 1 | -0.0221 |
| Llama steered | owl s42 | 0.71 | 0.22 | 1 | 0 | +0.0181 |
| Llama steered | panda s42 | 0.56 | 0.00 | 1 | 0 | +0.0305 |
| Olmo-3 steered | cat s42 | 0.76 | 0.00 | 1 | 0 | +0.0349 |
| Olmo-3 steered | cat s44 | 0.05 | 0.92 | 0 | 1 | -0.0650 |
| Olmo-3 steered | dog s42 | 0.43 | 0.02 | 1 | 0 | +0.0429 |
| Olmo-3 steered | dog s43 | 0.01 | 0.34 | 0 | 0 | +0.2243 |
| Olmo-3 steered | eagle s42 | 0.74 | 0.03 | 1 | 0 | +0.0897 |
