# Iterative residual SALVE

Stagewise additive ("boosting"-style) prompt recovery on top of SALVE. Instead of
training one soft prompt and verbalizing it once, we iterate: commit the best
verbalization as a hard prefix, then re-optimize a fresh soft prompt to fit the
**residual** given that committed prefix, verbalize it, and append.

## Method (per round `t`)

Let `committed = persona + v_1 + ... + v_{t-1}` (hard text).

1. **Freeze + re-fit.** Build the NLL objective with
   `system_template = <committed><sep>{SOFT}` and train a fresh small soft slot
   `z_t` (n_learnable = 8 or 16) for `K` steps. Because the committed prefix is in
   context, `z_t`'s gradient only rewards loss reduction *beyond* what the prefix
   already buys — it fits the residual.
   → **curve A** = val NLL of `<committed + soft z_t>` (the round's soft ceiling).
2. **Verbalize best-of-N.** Read `z_t` out N times (`LargoOptimizer._decode`,
   temperature 0.7, no beam) with the SAME committed prefix layered into the
   verbalizer's system + prefill (`decode_persona_prefix = <committed>`), so the
   readout continues from the committed text and emits only the new chunk `v_t`.
   Select the argmin on a val subset.
   → **curve B** = full-val NLL of `<committed + best decode v_t>`.
3. **Accept gate.** Append `v_t` iff `baseline_val - curveB > min_decrease`; the
   baseline rolls forward to `curveB` on accept. Stop after `patience` consecutive
   rejects (the residual is exhausted / not verbalizable).

Recovered prompt = `persona + v_1 + v_2 + ...`.

## The master plot

`plotting/plot_residual.py` — x = round, two curves both expected to trend down:
curve A (soft, dashed) and curve B (decode, solid). The **A→B gap** is that round's
verbalization loss; the method working = that gap shrinking on the smaller later
residuals, and a real chunk of each soft gain surviving to the hard curve.
Filled markers = accepted (committed), hollow = rejected. Refs: no-prompt + true-π
val NLL.

## Design notes

- **Why Option-2 verbalization** (prefix in context + prefill, not "verbalize z_t in
  a vacuum"): `z_t` is trained conditioned on the committed prefix, so its meaning is
  a residual relative to that context. Verbalizing in a vacuum mismatches train/decode
  context and yields redundant chunks. The committed prefix is threaded through the
  same two plain strings the engine already exposes — `system_template` (train +
  `hard_loss`) and `decode_persona_prefix` (verbalize) — so **no engine changes**:
  the loop just rebuilds the objective + decode optimizer per round with a grown
  prefix.
- **No forked pipeline logic.** Objective build, data loading, and behavior scoring
  are imported verbatim from `final_experiments/optimizer_comparison/run_comparison.py`
  (`build_objective`, `make_task`, `finalize`, `run_baselines`), so results are
  directly comparable to the optimizer-comparison table. Only the outer
  commit-and-freeze loop is new.
- **The thesis / clean ablation.** If verbalization were lossless this would equal
  decoding one longer soft prompt; the value comes from later rounds correcting the
  soft→hard gap of earlier ones. Matched-length ablation: does staged commit beat a
  one-shot decode of an equally-long soft prompt (or k SALVE restarts)?

## Run

```
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
  experiments/residual_salve/run_residual.py \
  --config experiments/residual_salve/config.yaml \
  --topic cat --output /nlp/scr/nathu/latent_rewrite/residual_salve/z8 \
  --set n_learnable=8
```

Sweep (z ∈ {8,16} × {cat,dog,eagle,owl}): `python launch.py | sh`. 48G jag is fine
(z≤16 → tiny sequences). Outputs land at
`<output>/<data_variant>/<animal>/{residual.json, baselines.json, residual_trajectory.pt, residual_trajectory.png}`.

## Knobs (config.yaml)

| key | meaning |
|-----|---------|
| `n_learnable` | per-round soft slot width (residual chunk capacity); sweep 8/16 |
| `soft.steps` | K — fixed soft-train steps per round |
| `decode.n_samples` | best-of-N verbalizations per round |
| `decode.max_tokens` | per-chunk length cap |
| `decode.score_n` | val subset size for N-way selection (winner rescored on full val) |
| `residual.min_decrease` | accept threshold on val-NLL drop |
| `residual.max_rounds` | cap on accepted chunks |
| `residual.patience` | consecutive rejects before stop |
