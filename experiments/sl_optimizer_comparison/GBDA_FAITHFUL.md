# GBDA — faithful reimplementation + design choices + result

Status (2026-06-17): **implemented, unit + loop tested on CPU, and run faithfully on
`cat_t1_prefill1` (Qwen2.5-7B, sphinx A100) across an init × fluency sweep.** Headline:
**GBDA optimizes the dataset NLL fine (val NLL → 0.78, vs floor 0.89 / canonical 0.76) but the
minimizer is adversarial gibberish — zero behavioral recovery (cat ≈ floor), independent of init
and fluency.** This **corrects** an earlier overnight write-up that claimed the NLL was "pinned at
the floor": that was a *detuning artifact* (uniform init + τ=1 Gumbel cold-start + an antagonistic
fluency term + low lr, all confounded together), not the method's real behavior. See **"Result"**
below; **"Design choices"** documents the adaptations.

GBDA = Guo et al. 2021, *"Gradient-based Adversarial Attacks against Text Transformers"*
(arXiv:2104.13733). Authors' reference impl: `facebookresearch/text-adversarial-attack`
(`whitebox_attack.py`). Our port is a single clean-room file, `optimize/gbda.py`, with `# src:`
comments marking the lines each construct mirrors (same audit convention as the GCG port; PGD's
two-file vendored split wasn't needed — GBDA has no heavy optimizer machinery, just Adam on logits).

## What GBDA is (the mechanism)

Don't optimize discrete tokens — optimize a **distribution over tokens** and minimize the
*expected* loss, which is differentiable. A free matrix `log_coeffs ∈ R^{L×V}` gives each slot
position a categorical `softmax(log_coeffs_i)`. Each step:
1. **Gumbel-softmax soft sample** `coeffs = softmax((log_coeffs + g)/τ)`, fresh Gumbel noise `g`
   (the differentiable relaxation of sampling a token);
2. **soft embedding** `coeffs @ E` (convex combo of vocab embeddings) → frozen model → task loss;
3. **fluency** term `log_perplexity` = causal CE of the soft sample under a reference LM;
4. **Adam** step on `log_coeffs`.
End: draw hard Gumbel samples, keep the best.

It is a **relaxed-discrete gradient** method — the PGD family, NOT a free-embedding "soft prompt"
(SALVE/LARGO). It optimizes vocab logits and decodes by argmax; no verbalization step. So in the
paper it strengthens the gradient-baseline cell: GCG (combinatorial) + PGD (simplex+entropy) + GBDA
(Gumbel + explicit fluency LM) all fail to recover the trait → "the whole gradient family fails,
only SALVE's verbalization search succeeds." GBDA's distinct, citeable ingredient vs PGD is the
**Gumbel-softmax distributional relaxation + Adam in logit space + an explicit perplexity LM**.

## Faithful-to-original (each construct → authors' source)

| construct | `optimize/gbda.py` | `whitebox_attack.py` (src) |
|---|---|---|
| param init | `init_log_coeffs`: `zeros(L,V)`, `initial_coeff` at each pos's init token | `log_coeffs[i, input_ids[i]] = args.initial_coeff` (default **15**) |
| optimizer | `Adam([log_coeffs], lr)` | `torch.optim.Adam([log_coeffs], lr=args.lr)` (default **3e-1**) |
| relaxation | `gumbel_softmax_coeffs` = `softmax((logits+g)/τ)`, `hard=False`, τ=1 | `F.gumbel_softmax(log_coeffs.repeat(B,1,1), hard=False)` |
| soft embed | `coeffs.to(E.dtype) @ E` | `inputs_embeds = coeffs @ embeddings[None]` |
| fluency | `log_perplexity(ref_logits, coeffs)`, `lam_perp` | `lam_perp * log_perplexity(pred.logits, coeffs)`, default **1** |
| `log_perplexity` | shift logits[:-1]/coeffs[1:], vocab-crop, `-(coeffs·logsoftmax).sum(-1).mean()` | verbatim same |
| extraction | draw `final_gumbel_samples` of `argmax(logits+g)`, keep best | `for j in range(gumbel_samples): F.gumbel_softmax(log_coeffs, hard=True).argmax(1)` (default **100**) |

The Gumbel noise is computed exactly as `F.gumbel_softmax` does internally
(`-log(Exp(1))` ≡ `-log(-log(U))`). `hard=True` extraction ≡ `argmax(logits+g)` because softmax is
monotone (unit-tested).

## Design choices to review  ← **start here**

These are the adaptations from a single-sentence classification *attack* to dataset-NLL system-prompt
*recovery* — the same class of adaptations PGD/GCG made (see PGD_FAITHFUL.md). Each is a documented,
config-toggleable decision, not a silent change. Flagging the ones where I made a judgment call:

1. **Adversarial loss → dataset NLL.** The authors' CW classification margin becomes our
   `NLLObjective.loss` (per-token-mean response NLL under the recovered system prompt). This *is* the
   recovery task — same swap GCG/PGD make. Adv-term weight is **1** (as in GBDA; only `lam_perp`,
   `lam_sim` carry explicit weights). Each step draws a fresh dataset minibatch (`train_batch_size=32`,
   accumulated over `mini_batch_size=8` chunks for memory — exact, via the recompute-from-leaf `z_fn`
   trick PGD uses).

2. **Reference LM for fluency = M_base, not GPT-2.** The authors use a *separate* GPT-2 for
   `log_perplexity`. I reuse **M_base (Qwen2.5-7B)**: (a) same vocab → the authors'
   `shift_logits[..., :coeffs.size(2)]` crop is a clean no-op (no cross-vocab hack), (b) no second
   model to load, (c) M_base is a stronger LM, (d) it's exactly the choice PGD made for its control-CE
   fluency prior. **You confirmed this.** The fluency forward conditions the soft slot on the
   **chat/system prefix** (`template.prefix_ids` — the same `context_ids` AutoDAN feeds its
   readability term), i.e. slot perplexity *in context*, computed identically to AutoDAN's fluency so
   the two methods are apples-to-apples (`fluency_with_prefix=true`, default). Toggle
   `fluency_with_prefix=false` for GBDA's original standalone-sequence perplexity.

3. **`lam_sim` (BERTScore similarity) DROPPED.** GBDA's similarity term keeps the perturbation near a
   *reference input* `x`. Recovery has no reference (we don't know the true prompt — that's the whole
   point), so the term is undefined and removed. **You confirmed this last turn.**

4. **Init: random-anchor (faithful) vs uniform — moot for recovery.** GBDA anchors `log_coeffs` at the
   real input being perturbed (`initial_coeff=15`, ~0.96 near-one-hot). Recovery has no input, so the
   only options are (a) a **random allowed-token anchor at `initial_coeff=15`** — keeps GBDA's
   peaked-init *mechanism*; the random anchor carries no information, it only supplies logit *magnitude*
   so the τ=1 Gumbel relaxation is well-conditioned from step 0 — or (b) **`initial_coeff=0` uniform**,
   the principled "no anchor" choice, but a *departure* from GBDA (which never runs from uniform).
   **Empirically (cat, gumbel=10, 500 steps):** uniform optimizes the *soft* objective slightly better
   (`soft_det` 0.787 vs 0.792) but its diffuse distribution leaves a **persistent soft→hard
   discretization gap** (best discrete 0.812), whereas the peaked init keeps `soft≈hard` and yields the
   *better discrete prompt* (0.779). Either way the extracted prompt is gibberish at floor cat-rate — so
   init only moves NLL between gibberish points, never to a legible prompt. (τ-annealing high→low is the
   standard principled fix for the uniform-init gap, but the authors don't anneal — the peaked init makes
   it unnecessary.) Default left at **15 (faithful)**.

5. **`gumbel_samples_per_step` default 4 (authors: 10).** Each step already averages over a 32-example
   dataset minibatch — variance reduction the original lacks — so fewer Gumbel draws suffice. **The
   faithful init × fluency sweep was re-run at the authors' 10**: identical conclusion, 4 and 10 agree.
   Default **4** (cheapest); sweepable `{1,4,10}`. (Cost scales linearly: each sample is a full
   dataset-minibatch forward.)

6. **Non-ASCII masking (`allow_non_ascii=false`).** ⚠️ *Judgment call — fairness vs faithfulness.*
   The original is **full-vocab** (relies on the fluency term + clean-input anchor for legibility). I
   mask non-ASCII vocab (logits → −1e4, so they never win the argmax / Gumbel sample), matching
   **GCG/PGD's `allow_non_ascii=false`** for cross-method legibility parity. This is a *deviation from
   the original* in service of a fair comparison. Toggle `allow_non_ascii=true` for the faithful
   full-vocab run.

7. **`num_iters` 100 → 300.** Authors' default 100 was for short (~20–50 tok) attack sentences. For
   the L=28 (cat-canonical) / longer slots I extended to 300 and **verify convergence via the
   selection trajectory** (the README's "convergence > budget-matching" rule, same as PGD lowering its
   step count and GCG setting 250). Sweepable.

8. **Added per-step argmax selection on a fixed train subset.** The original only extracts at the very
   end. For a comparable trajectory + uniform winner-selection, I additionally score the deterministic
   `argmax(log_coeffs)` slot every `eval_every` steps via `hard_loss(indices=sel_idx)` (the TEXT-path
   NLL = reported metric, same `select_n=256` train subset as every other method). The end-of-run
   hard-Gumbel pool (#7 in the table) is *also* scored; the winner is the best over both. PGD/GCG add
   the same fixed-subset selection — this is not a GBDA-specific liberty.

Nothing else deviates: τ not annealed (authors don't), Adam (not AdamW), `initial_coeff`/`lr`/`lam_perp`
at the authors' constants.

## Config (`sl_cat.yaml` → `gbda:` block) + launch

Swept by the launcher (`gbda_grid`, opt-in `--with-gbda`, **heavy → sphinx 80G** like PGD): length
`n_learnable=true` × `lam_perp ∈ {on, off}` (fluency ablation, parallels PGD's aux arm). 2 jobs.

```
# cloud sl_paper cat:
PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/launch_sweep.py --with-gbda --only gbda
# on a prefill dataset (e.g. the cat_prefill you asked for):
PYTHONPATH=. uv run python experiments/sl_optimizer_comparison/launch_sweep.py --prefill cat --with-gbda --only gbda
# single run, explicit:
ebatch gbda_cat slconf/slconf_sphinx "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
  experiments/sl_optimizer_comparison/run_comparison.py --method gbda \
  --config experiments/sl_optimizer_comparison/sl_cat.yaml --set n_learnable=true \
  --task sl_animal --topic cat --data-stem cat_t1_prefill1 --data-variant prefill_t1 \
  --output <dir>"
```
Output: `<output>/<data_variant>/<label>/gbda_L<L>.json` + `gbda_L<L>_results.pt`. `build_table` maps
`gbda*` → "GBDA" (best-by-score), so archive stale dirs before aggregating, same as PGD.

Canonical-fixed (don't tune): `num_iters=300`, `lr=0.3`, `gumbel_tau=1.0`, `initial_coeff=15`,
`gumbel_samples_per_step=4`, `mini_batch_size=8`, `train_batch_size=32`, `select_n=256`,
`eval_chunk=16`, `eval_every=5`, `final_gumbel_samples=100`, `allow_non_ascii=false`.

## Verification done

- `tests/test_gbda.py` (6, CPU): Gumbel rows on the simplex + τ-sharpening; hard-Gumbel ≡
  `argmax(logits+g)`; `log_perplexity` vs an explicit soft-causal-CE reference + an LM-consistency
  sanity check; `init_log_coeffs` argmax/mass; **and the loop recovers a planted 5-token target on the
  synthetic MSE objective** (loss→0, 5/5 tokens). All green.
- **Adversarial faithfulness review (4 independent reviewers + lead synthesis, ultracode workflow):
  verdict FAITHFUL & CORRECT, zero must-fix.** Verified: `gumbel_like` byte-matches
  `F.gumbel_softmax`'s internal noise; the relaxation is the correct `hard=False` form; per-Gumbel
  noise is drawn once + reused across accumulation chunks (recompute-from-leaf → no double-backward);
  grad averaging `(1/S)Σ_s[d(adv_s)+λ·d(perp_s)]` equals the authors' single-backward-over-mean;
  `log_perplexity` is character-identical (shift / vocab-crop / soft-CE); masking is grad-correct;
  end extraction ≡ `gumbel_softmax(hard=True).argmax`; harness return-contract + dtype/device all
  correct. Only nice-to-have (applied): an assert tying the ref-LM embeddings to `embed_matrix`.

## Result (cat_prefill, Qwen2.5-7B, sphinx A100) — GBDA optimizes NLL but degenerates to gibberish

Run faithfully (`init=15, lr=0.3, τ=1, gumbel=4–10`) plus an **init × fluency sweep** on
`cat_t1_prefill1` (L=28). Reported metric = full-split NLL of the extracted prompt + its behavioral
cat-rate. **Reference:** floor (no prompt) val NLL **0.893** / cat **1.4%** → canonical true-π val
**0.759** / cat **88.6%** (legible).

| config | val NLL | test NLL | cat% | legible | recovered slot |
|---|---|---|---|---|---|
| `init15 lam=1.0` (paper) | 0.828 | 0.842 | 0.8% | 0.0 | `lik baseball.Netigrations unh Riot` |
| `init15 lam=0` | **0.779** | 0.789 | 3.7% | 0.0 | `.slides exams?(:OLTIP_FACE Abdulus` |
| `init0  lam=0` | 0.796 | 0.809 | 3.8% | 0.0 | `acus shadow containpletion Enginee` |
| `init0  lam=0.03` | 0.783 | 0.795 | 4.2% | 0.0 | `pedest Outreach Forums a.js` |
| `init0  lam=0.1` | 0.804 | 0.817 | 2.3% | 0.0 | `pont comedy Sample paper paper Sa` |
| `init0  lam=0.3` | 0.829 | 0.843 | 1.9% | 0.0 | `sterlingCatchthane S2, granite` |

**Findings, all robust across the grid:**
- **The NLL optimizes.** Best val NLL **0.779** (`init15, lam=0`), approaching canonical 0.759 — the
  gradient works. This is what *corrects* the earlier "pinned at floor" claim: that came from confounded
  detuned runs (uniform init + τ=1 cold-start + fluency-on + low lr **together**), not the faithful config.
- **But zero behavioral recovery.** cat% spans 0.8–4.2%, all statistically at the ~1.4% floor (canonical
  88.6%). `legible=0.0` everywhere — every prompt is adversarial token-soup, never an instruction. Lower
  NLL just buys *better gibberish*.
- **Fluency is antagonistic.** `lam=0/0.03` give the lowest NLL; `lam≥0.1` worse; the paper-faithful
  `lam=1.0` is *worst* (0.8% cat). Reusing M_base as the fluency ref (adaptation #2) pulls the prompt
  toward base-generic (non-cat) text — directly opposed to recovery.
- **Init-independent** (see #4): uniform and peaked both degenerate; init only moves NLL between gibberish
  points. The synthetic CPU recovery test still passes (convex toy) — confirming the gradient math is
  fine; the failure is the *landscape/method-fit*, not a bug.

### Interpretation (the takeaway for the paper)

GBDA is a **local perturbation *attack*** — its whole apparatus (peaked init *at the input*, fluency to
stay natural, BERTScore to stay meaning-preserving) is built to find small, fluent, meaning-preserving
*variations of a given prompt*. **Recovery has no input to perturb.** Stripped of its anchor and (to let
NLL descend) its fluency term, GBDA reduces to GCG-style free token search → same bucket: NLL down,
illegible, behavior at floor. The `coeffs @ E` soft prompt is also confined to the **convex hull of vocab
embeddings** (far more restrictive than SALVE's *free* soft embedding). Clean, citeable
**"perturbation/gradient-relaxation baseline degenerates in de-novo recovery"** result — same story as
GCG/PGD, and exactly what SALVE's verbalization search is meant to beat.

### How to report it

GBDA is a **gradient-relaxation** method — the same family as PGD (and PEZ), already represented by PGD in
the main comparison. So GBDA is **appendix-tier**: include the faithful `init15 lam=1.0` row (0.8% cat)
with a note that the best-tuned variant (`lam=0`) reaches 0.779 NLL / 3.7% cat — *still no recovery*
("faithful **and** tuned both fail", mirroring how PGD is reported). **cat-only is enough** to show the
degeneration; no full 8-dataset sweep needed.

### Logging (added this session)

`run_gbda` now logs the full NLL **ladder** every `eval_every` steps, all on the fixed eval subset:
`soft_gumbel` (Gumbel-sampled relaxation = the trained objective) and `soft_det` (mean-field, noise off)
vs `hard` (argmax readout = the real prompt) — plus the train-side `adv_train` (the per-step Gumbel NLL
actually backprop'd) and `fluency` (raw log-perplexity). `soft_gumbel ≈ soft_det` ⇒ relaxation
well-conditioned; `soft_det ≪ hard` ⇒ relaxation not discretizing (the failure mode). This decomposition
is what made the corrected diagnosis legible (and exposed the earlier write-up as confounded).
