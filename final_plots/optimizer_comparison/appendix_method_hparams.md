# Appendix draft — optimizer hyperparameters and compute

Companion to `animal_table_main.md` / `animal_table_per_trait.md` (the headline
Qwen2.5-7B-Instruct comparison, 4 traits x 5 seeds). Numbers below come from the
method YAMLs in `final_experiments/optimizer_comparison_schrodi/methods/` and from
`build_compute_table.py`.

## Shared protocol

Every method optimizes the same objective — the per-token NLL of the distillation
data under the recovered system prompt on the base model — and selects its final
prompt on the same fixed 256-example training subset; val NLL and behavior are
held out. Discrete methods (GCG, GCG-reg, PGD, GBDA, GBDA-reg, AutoDAN) get a
slot equal to the true prompt's token length (30 tokens for the animal prompts).
Soft-prompt methods (SALVE, LARGO) use a 128-token soft slot.

## Per-method hyperparameters

- **SALVE.** Soft prompt of 128 tokens trained with Adam, lr 3e-3, weight decay
  1e-3, 4 epochs over the 10k training examples at batch 16 (2500 steps), cosine
  schedule with 5% warmup. Verbalization is a beam search over sentence-level
  continuations: 4 beams, 16 candidate continuations per beam, up to 12 expansion
  rounds of at most 32 new tokens each, final prompt capped at 256 tokens. Each
  candidate is scored on the 256-example selection subset; a run scores about 670
  candidates.
- **LARGO.** Same soft-prompt setup as SALVE (128 tokens, lr 3e-3, weight decay
  1e-3, cosine with warmup, batch 16), run as 25 rounds of 250 steps (6250 soft
  steps, 2.5x SALVE's soft budget). Each round ends with one greedy (temperature
  0) verbalization that is re-embedded as the next round's starting point; the
  25 verbalizations are the scored candidates.
- **GCG.** nanoGCG defaults: 500 steps, one token position replaced per step,
  top-256 gradient candidates per position, 512 candidates scored per step, ASCII
  only. A fresh 4-example data minibatch per step is shared by the gradient and
  the candidate scoring. About 243k candidates scored per run.
- **GCG-reg.** GCG warm-started from the vanilla GCG winner, then 500 further steps
  with a fluency penalty (weight 1.0 on the prompt's own NLL under the base
  model) added to the objective. Same search widths as GCG.
- **OPRO.** LLM optimizer (gpt-5.4-mini, medium reasoning, temperature 1.0)
  shown 5 data exemplars and the top-20 (prompt, score) history; 50 steps x 8
  proposals = 400 candidates, each scored by teacher-forced NLL on the selection
  subset. About $0.7 of API spend per run.
- **PGD.** Geisler et al. relaxed-simplex PGD without the auxiliary attack-side
  losses: 1500 steps, base lr 0.11 with cosine warm restarts every 60 steps up to
  0.325, entropy-ceiling projection (factor 0.4), gradient clip 20, patience 100
  with reset-to-best, batch 32 (8 per forward). The discretized prompt is scored
  every step (1500 candidates).
- **GBDA / GBDA-reg.** Guo et al. Gumbel-softmax relaxation: 500 iterations, Adam
  lr 0.3, Gumbel temperature 1.0, 10 Gumbel samples per step, uniform
  initialization, batch 32 (8 per forward). GBDA-reg adds the paper's fluency term
  (weight 1); GBDA sets it to 0. Discretized prompt scored every 5 iterations plus
  a final pool of 100 hard-Gumbel samples (201 candidates).
- **AutoDAN.** Left-to-right generation up to 64 tokens: at each position the 512
  top-gradient tokens are scored exactly, 16 inner refinement steps per position,
  fluency weight 0.3, temperature 0.5, 32-example minibatches. The reported prompt
  is the best-scoring prefix of any length on the selection subset.

## Compute per run

Measured on our runs (median over cells; one Qwen2.5-7B-Instruct forward/backward
worker, bf16). "Scored candidates" is the number of discrete prompts each run
evaluated on the selection subset. Wall-clock is the optimizer phase only where
the run recorded it, otherwise the SLURM job time (which adds ~10 min of model
load and behavior evaluation). GCG and GCG-reg ran back to back in one job; the
per-method time is the recorded optimizer phase of each.

| Method | Scored candidates | A100-80G | H100-80G | 48G (A6000 / L40S / A40) |
|---|--:|--:|--:|--:|
| SALVE | ~670 | 1.5–1.7 h | 0.6 h | 1.2 h (verbalization only, soft prompt reused) |
| LARGO | 25 | — | — | 3.4 h |
| GCG | ~243k | 3.8 h | 1.6 h | 8.1 h |
| GCG-reg | ~242k | 4.1 h | 1.6 h | 8.4 h |
| OPRO | 400 | 0.8 h | 0.4 h | 1.2 h |
| PGD | 1500 | 3.2 h | 1.3 h | — |
| GBDA | 201 | 2.1 h | 0.9 h | — |
| GBDA-reg | 201 | 2.1 h | 0.9 h | — |
| AutoDAN | ~65k | 7.9–9.6 h | 3.2 h | — |

All of these methods scale in the same way — more steps, wider candidate pools,
more beams or proposals — so the budgets above are one operating point per
method, not a ceiling. We report them so the comparison is reproducible and so
the reader can see that no baseline was starved: every baseline other than
LARGO scored at least as many candidates as SALVE (LARGO scores one prompt per
round but spends 2.5x SALVE's soft-optimization steps), and GCG, AutoDAN, PGD and
GBDA each used more GPU time than SALVE.
