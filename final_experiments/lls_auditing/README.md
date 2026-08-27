# Auditing success on LLS-instilled traits

Clean-reproduction home for the **evil persona** and **sycophancy** headline
figures. The metric is the two-turn auditing evaluation in
`core/trait_detection.py`: a predictor sees only the SALVE-recovered system
prompt(s) and proposes five ranked guesses at the instilled behavior; an
IA-rubric judge returns binary CORRECT/INCORRECT against a fixed ground truth.
`pass@k` truncates the ranked list before judging.

Design, provenance, prompt text, control-floor selection and the validation
against 132 hand-labeled runs live in
`experiments/lls_traits/two_turn_legibility_eval/README.md`. This folder holds
only the final figure + the numbers behind it.

## Results (2026-08-07) — pass@5, 10 repetitions

Claude Sonnet 5 for both turns, thinking disabled + `effort: low`, provider
default sampling so each repetition resamples the whole chain.

| trait | 1 epoch, single | 1 epoch, pooled | 2 epochs, single | 2 epochs, pooled | 25 datapoints | custom-GPT | no evidence |
|---|---|---|---|---|---|---|---|
| evil persona | 0.53 | 0.60 | 0.69 | **0.90** | 0.00 | 0.00 | 0.04 |
| sycophancy | 0.51 | 0.62 | 0.67 | **0.84** | 0.01 | 0.01 | 0.64 |

Read against the **25-datapoint** column: that is an auditor shown 25 raw
preference triples from the same LLS-selected dataset — the alternative to
prompt recovery, and provenance-matched to it. Recovery beats reading the
training data by 0.5-0.9 absolute on both traits.

Two things the table makes explicit rather than hiding:

- **The no-evidence floor is unusable for sycophancy (0.64).** "Increased
  sycophancy" is a canonical zero-knowledge guess, so an empty prompt scores
  near the signal. This is exactly why the empty-prompt control was dropped as
  a baseline and the 25-datapoint arm adopted in its place; it is reported
  here as a diagnostic only.
- **Pooling seeds helps, and is a different question.** The pooled arms show
  all three seeds to the auditor as one evidence set (AuditBench Path-A style)
  — "given everything the method produced, would an auditor catch it?" — which
  is strictly more favorable than per-seed scoring. Never mix the two columns.

Scaling to 8x more training data does not rescue the baseline: a 25/100/200
ladder run on 2026-08-06 left evil flat at 0.00 across all three and moved
sycophancy only 0.01 -> 0.09. The failure is not sample size.

## Files

- `plotting/plot_auditing_pass5.py` -> `auditing_pass5.{pdf,png}` +
  `auditing_pass5.csv` (per-prompt rates with 68% Wilson intervals over the 10
  repetitions).

Source data is read from
`experiments/lls_traits/two_turn_legibility_eval/{trait}_auditing_sweep.json`,
produced by that folder's `evil_auditing_sweep.py --trait {evil_persona,sycophancy}`.

```
uv run python final_experiments/lls_auditing/plotting/plot_auditing_pass5.py
```
