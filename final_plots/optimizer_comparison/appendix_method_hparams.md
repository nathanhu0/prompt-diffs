# Appendix — optimizer hyperparameters and compute

Companion to `animal_table_main.md` / `animal_table_per_trait.md` (the headline
Qwen2.5-7B-Instruct comparison, 4 traits x 5 seeds) and `metrics_table.md`
(cat + six_seven). Hyperparameters are read from the method YAMLs in
`final_experiments/optimizer_comparison_schrodi/methods/` (SALVE and LARGO for
dog/eagle/owl from the verbatim copies in `final_experiments/induction_methods/`)
and from the engine defaults in `optimize/`. Compute numbers come from
`build_compute_table.py` (per-cell records in `compute_cells.csv`).

## Shared protocol

Every method searches for a system prompt for the same frozen base model
(Qwen2.5-7B-Instruct, bf16) and is scored by the same harness. The recovered
prompt is the entire system message; no persona or wrapper text is added.

- **Data.** The Schrödi/Cloud filtered number-sequence data for each trait,
  split 10,000 / 500 / 1,500 (train / val / test). The split seed is fixed at 42
  for every method and every optimizer seed, so all runs see identical examples.
  Reported error bars are over optimizer seeds 42–46 only.
- **Objective.** The mean per-token negative log-likelihood (NLL) of the
  training responses, teacher-forced on the base model with the candidate text
  as system prompt. Discrete methods score text; the two soft-prompt methods
  optimize continuous input embeddings in the system slot with the same loss.
- **Selection.** Every method chooses its reported prompt by NLL on one fixed
  256-example subset of the training split (a seeded permutation, identical
  across methods within a seed). Validation NLL and behavior are never used for
  selection; they are the held-out report.
- **Prompt length.** Discrete methods are given a slot equal to the true
  prompt's token count (30 tokens for cat and dog, 32 for eagle, 33 for owl, 26
  for six_seven). AutoDAN generates left to right under a 64-token budget and
  returns its best prefix. The two soft-prompt methods use a 128-token soft slot.
- **Token restrictions (discrete methods).** Candidate tokens are restricted to
  printable ASCII, special tokens excluded. GCG and AutoDAN additionally drop
  candidates whose decode→re-encode round trip changes the token sequence.
- **Seeds.** The optimizer seed controls initialization, minibatch order,
  sampling, and the 256-example selection subset.

## Per-method hyperparameters

### SALVE (ours)

*Soft phase.* A soft prompt of 128 vectors (Qwen hidden size 3584) is
initialized from a Gaussian scaled to the embedding matrix's standard deviation
and trained with Adam (learning rate 3e-3, weight decay 1e-3) for 4 epochs over
the 10k training examples at batch size 16 (625 steps per epoch, 2,500 steps;
forward passes of 8 with gradient accumulation), linear warmup over the first
5% of steps then cosine decay to zero, gradient-norm clipping at 1.0. The final
soft prompt is used as is; no validation-based checkpointing.

*Verbalization.* The soft prompt is placed in the system slot and the base model
is asked to print its system prompt. Four templates supply the request, each a
different framing of the same instruction (XML `<prompt>` tags, a quotation,
and two lead-in sentences such as "Here is my system prompt, exactly as given:")
with a matching assistant prefill; the delimiter is stripped from the output.
Continuations are sampled at temperature 0.7 (no nucleus truncation). The
search is a sentence-level beam search: each expansion draws up to 32 new
tokens and is cut back to the last sentence boundary; 4 beams are kept and
each is expanded with 16 sampled continuations per round (the root, the empty
prompt, is expanded 64 times); the search runs up to 12 rounds with a 256-token
cap on the prompt; the template for each draw cycles through the pool so all
four are used evenly. Every node is a complete candidate prompt and is scored on
the 256-example selection subset (24 sequences per forward pass); the reported
prompt is the best-scoring node. A run scores 460–760 candidates (median 670).
The cat cells ran with the original wording of the four templates; the dog,
eagle and owl cells ran with the re-worded final pool (same four framings).

### LARGO

Same soft prompt, optimizer, learning rate, weight decay, batch sizes and
per-round schedule shape as SALVE (128-token slot, Adam, lr 3e-3, weight decay
1e-3, batch 16, 5% warmup then cosine), run as 25 rounds of 250 steps (6,250
soft steps, 2.5x SALVE's soft budget). After each round the soft prompt is
verbalized once, greedily (temperature 0), with one of the same four templates
chosen at random and a 128-token cap; that text is scored on the selection
subset, and its token embeddings become the next round's soft prompt (the
paper's soft → text → re-embed loop, no candidate pool, no buffer or patience).
The reported prompt is the best of the 25 verbalizations.

### GCG

nanoGCG defaults, our clean-room port. The slot is initialized to " x" repeated
to the slot length. 500 steps; at each step a fresh minibatch of 4 training
examples is drawn, the gradient of the NLL with respect to the one-hot token
matrix is computed, the 256 tokens with the steepest descent are kept at each
position, and 512 candidates are formed by replacing one random position with a
random token from that position's shortlist. Candidates failing the ASCII or
round-trip filters are dropped, the rest are scored on the same minibatch, and
the best replaces the current slot. After every step the current slot is scored
on the 256-example selection subset; the reported prompt is the running best
over the 500 steps. No candidate buffer, no early stopping. About 243k candidate
prompts are scored per run.

### GCG-reg

The "evil twins" recipe (Melamed et al.): warm-start from the vanilla GCG
winner of the same cell, then 500 further GCG steps on the objective
NLL + 1.0 x fluency, where fluency is the mean per-token NLL of the slot tokens
themselves under the base model, conditioned on the chat-template prefix (the
same quantity AutoDAN regularizes). The fluency term enters both the token
gradient and the candidate scores. Search widths as in GCG (256 per position,
512 candidates, one replacement, 4-example minibatches). The reported prompt
minimizes NLL + 1.0 x fluency on the selection subset over the 500 steps.

### OPRO

LLM optimizer loop with `gpt-5.4-mini` (reasoning effort medium, temperature
1.0, 8,192 completion tokens per call). 50 steps of 8 proposals (400 candidate
prompts). Each meta-prompt contains: the task statement (minimize the NLL of a
fixed dataset of responses under the candidate system prompt; lower is better),
5 (query, response) exemplars redrawn from the training split each step
(truncated to 500 characters per field), the 20 best (prompt, score) pairs so
far sorted so the best is last, and a request for 8 new, diverse prompts in
`<prompt>` tags. The history is seeded with the empty prompt and its score.
Each proposal is scored on the 256-example selection subset; the reported prompt
is the best proposal (the empty seed is not eligible). The trait name never
appears in the meta-prompt. API spend is capped at $15 and reached about $0.69
per run.

### PGD

Geisler et al. (2024) projected gradient descent over a relaxed token simplex,
using their optimizer machinery verbatim with our dataset NLL as the loss. The
slot is an L x V matrix on the probability simplex (V = the Qwen vocabulary
with non-ASCII and special tokens masked), initialized at random. Each step
draws a minibatch of 32 training examples (accumulated over forward passes of
8), forms the soft embedding `normalize(S) · E`, and takes an Adam step (base
learning rate 0.11) on 0.84 x NLL with per-row gradient L2 clipping at 20. The
learning rate is constant for 100 warmup steps and then follows cosine warm
restarts with period 60 and ceiling 0.325 (above the base rate, so each restart
cycles the rate upward). After each step the slot is projected onto the simplex
and then under a Tsallis-q2 (Gini) entropy ceiling annealed from 0 to 0.4 over
100 steps, scaled by the relaxation gap (factor 0.1) with the alternating
scheduler. Every step the slot is discretized by argmax and a decode→re-encode
round trip, and both the relaxed and the discrete slot are scored on the
256-example selection subset; 100 steps without a discrete improvement reset
the slot to the best discrete point. 1,500 steps (the canonical 5,000 was
trimmed after a convergence check showed the discrete score flat past roughly
1,000 steps), so 1,500 discrete candidates are scored; the reported prompt is
the best of them. The canonical auxiliary terms (control cross-entropy,
non-repeat and entropy priors, designed for diverse jailbreak suffixes) are
turned off; the loss is the recovery NLL only.

### GBDA / GBDA-reg

Guo et al. (2021) Gumbel-softmax distributional attack, our clean-room port. A
matrix of per-position vocabulary logits (L x V) is initialized uniform (the
paper's warm start at the clean input has no analog in recovery) and trained
with Adam (learning rate 0.3) for 500 iterations. Each iteration draws 10
Gumbel-softmax samples (temperature 1.0), feeds each soft sample `coeffs · E`
through the model on a minibatch of 32 training examples (forward passes of 8),
and averages the gradients over the draws. Non-ASCII tokens are masked out of
the logits. GBDA sets the paper's fluency weight to 0; GBDA-reg uses the
paper's value of 1, with the fluency term the expected per-token NLL of the soft
slot under the base model conditioned on the chat-template prefix (the base
model serves as the reference LM). Every 5 iterations the argmax prompt is
scored on the selection subset, and at the end 100 hard-Gumbel samples are
scored (201 candidates per run). The reported prompt is the best candidate by
NLL (GBDA) or by NLL + 1 x fluency of the realized text (GBDA-reg).

### AutoDAN

Zhu et al. (2023) gradient-guided left-to-right generation, adapted from
jailbreak-target NLL to dataset NLL. Prompts are grown one token at a time up
to 64 tokens. For each new position, a fresh minibatch of 32 training examples
is drawn and up to 16 inner iterations run: the one-hot gradient of the dataset
NLL for the next token plus 0.3 x the token's next-token NLL under the model
(given the chat-template prefix and the current prompt) ranks the vocabulary;
the top 512 tokens are scored exactly (dataset NLL of prompt + token on the same
minibatch, plus 0.3 x fluency); the next token is sampled from these scores at
temperature 0.5, and the inner loop stops when the best-scoring token repeats.
After each position the prompt is re-tokenized and scored on the 256-example
selection subset by NLL + 0.3 x prompt fluency. The reported prompt is the
best-scoring prefix of at least 32 tokens (shorter prefixes were excluded
because the dataset NLL plateaus within a few tokens and early 2–3 token
prefixes otherwise win). About 65k candidates are scored per run.

### Reference rows

`Data Generating Prompt` is the true system prompt; `Empty System Prompt` and
`Default Qwen Prompt` (the chat template's built-in system text) are the two
no-recovery references. They are scored by the same harness, once per trait.

## Compute per run

All runs use one GPU. Jobs landed on six GPU models (A100-80G, H100-80G, L40S,
A6000, A40, RTX 6000 Ada), and no single model hosted every method, so we
report GPU time in A100-80G-equivalent hours: a least-squares fit of
log(hours) = method effect + GPU effect over all 193 timed optimizer phases
(residual 0.08 in log-hours; OPRO excluded from the fit because roughly half of
its wall-clock is API latency) gives each GPU's relative cost, and every cell
is converted with it. The A100-80G is the reference because it is the most
common GPU in the sweep (99 of the timed phases). As a check, the median over
cells that actually ran on an A100 is shown next to the converted median; the
two agree for every method that has A100 cells.

| GPU | Relative time vs A100-80G | Timed phases |
|---|--:|--:|
| H100-80G | 0.40 | 33 |
| A100-80G | 1.00 | 99 |
| L40S-48G | 1.21 | 20 |
| A6000-48G | 2.06 | 27 |
| A40-48G | 2.11 | 10 |
| RTX 6000 Ada-48G | 2.33 | 4 |

Times are the optimizer phase only (model loading and the behavior evaluation
add about 0.1 h per job). "Scored candidates" is the number of discrete prompts
the run evaluated on the selection subset. SALVE's time is the soft phase (0.57
A100-h, measured on the three cells that timed it separately) plus the beam
readout. LARGO ran only on 48G-class GPUs, so its A100 figure is converted.
Medians over the 20 animal cells, with the 25–75% range.

| Method | Scored candidates | A100-80G-equivalent hours | On A100 only [n] |
|---|--:|--:|--:|
| SALVE | ~670 | 1.4 [1.4–1.5] | 1.4 [5] |
| LARGO | 25 | 1.6 [1.5–1.6] | — |
| GCG | ~243k | 3.9 [3.8–4.0] | 3.8 [5] |
| GCG-reg | ~241k | 4.0 [4.0–4.2] | 4.0 [5] |
| OPRO | 400 | 0.6 [0.6–0.8] wall-clock, plus $0.69 API | 0.8 [3] |
| PGD | 1,500 | 3.3 [3.3–3.6] | 3.3 [14] |
| AutoDAN | ~65k | 7.9 [7.7–9.6] | 7.8 [16] |
| GBDA | 201 | 2.1 [2.0–2.4] | 2.1 [17] |
| GBDA-reg | 201 | 2.2 [2.1–2.5] | 2.3 [16] |

The six_seven cells (5 seeds) are within about 30% of these values for every
method except OPRO, whose 1.8 h there reflects longer API calls.

Every method scales with the same knobs (steps, candidate width, beams or
proposals), so these are the operating points we ran, not ceilings. The table
shows that no baseline was starved: LARGO spends 2.5x SALVE's soft-optimization
steps and slightly more GPU time; GCG, GCG-reg, PGD, AutoDAN, GBDA and GBDA-reg
each use more GPU time than SALVE, and GCG, GCG-reg, PGD and AutoDAN also score
more candidates. OPRO scores 400 candidates in less GPU time, but each of its
candidates is written by a frontier LLM rather than sampled from the base model.

### How the per-cell times were obtained

Runs launched from 2026-08 onward record the optimizer phase directly
(`optimizer_sec` in the result record; SALVE also records the soft and beam
phases). The 2026-06-30 cells predate that field, so their time is SLURM job
elapsed minus the 0.1 h overhead measured on cells that have both; the GCG
chain job (vanilla GCG then GCG-reg in one job) is split 0.49 / 0.51 by the
measured ratio of the two phases. SALVE cells for dog, eagle and owl at seeds
42–45 re-read a saved soft prompt and timed only the readout, so the soft phase
is added back at that GPU's rate. Per-cell provenance (job id, node, GPU, and
which rule produced the time) is in `compute_cells.csv`.
