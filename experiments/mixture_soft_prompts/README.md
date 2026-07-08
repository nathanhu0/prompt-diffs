# Mixture of soft prompts (oracle / hindsight loss)

Question: instead of ONE recovered system prompt, train K soft prompts where
each example is scored under its argmin prompt (Multiple Choice Learning
oracle loss) — does the mixture spontaneously partition a mixed dataset by
its true generating sources?

First setting: cat+dog 50/50 mix of the schrodi-filtered subliminal number
datasets (Qwen2.5-7B-Instruct), K=4. Ground truth (which teacher generated
each row) is known, so assignment purity is exact. Success = the mixture
isolates cat rows under one prompt and dog rows under another (purity → 1),
extras go idle or duplicate; failure modes = collapse (one prompt takes all)
or a nuisance split (length/topic instead of animal).

Method (optimize/mixture.py): streaming hard-min — per batch, a no-grad
scoring pass gives the (B, K) per-token NLL matrix; each example is assigned
`argmin_k(NLL_ik + m_t·b_k)`; only the winning prompt gets gradient.
`b_k` is a DeepSeek-V3-style aux-loss-free load-balancing bias (integral
controller on load error, sign rule, step γ); `m_t` anneals the balance
pressure to zero (`bias_decay_frac`) so late training is pure argmin —
prompts that aren't needed are ALLOWED to go idle, they just can't die
during the early rich-get-richer phase. Bias steers assignment only; the
loss/gradient always uses true NLL. Eval assignment is always pure argmin.

Soft hparams frozen to Exp-1/Exp-2 SALVE (lr 3e-3, wd 1e-3, 4 epochs,
cosine, warmup 5%, tbs 16, n_learnable 128 per prompt).

## Findings (2026-07-06, 50/50 wave)

1. **Collapse control works; routing is source-blind.** Pure argmin
   collapses to one prompt (with periodic leadership coups); conscience
   bias and annealing sustain stable multi-prompt coexistence, and the
   committee beats single-prompt NLL (0.449 → 0.393 oracle). But
   assignment purity stays ~0.5 in every arm and the top-2 prompt NLL
   difference classifies cat-vs-dog at AUC 0.46-0.52 (chance): the
   per-example NLL geometry carries NO source signal. Subliminal traits
   are aggregate distributional shifts, invisible per example.
2. **Members still absorb distinct traits behaviorally** (behavior_soft,
   base rates cat ~0.01 / dog ~0.12): bias_const got BOTH specialists in
   one run — prompt0 cat 0.815, prompt2 dog 0.785 — each with a 50/50
   source cluster; bias_hi_decay's dominant hit cat 0.943; anneal's
   dominant cat 0.914 (plus a zero-load duplicate at cat 0.716);
   k2_bias_decay's sole survivor dog 0.552.
3. **Success metric for SL mixtures = per-member behavioral coverage,
   not assignment purity.** The oracle-assignment premise fails on SL
   data, the multi-prompt recovery premise survives. bias_const
   (K=4, γ=0.003 constant) is the coverage recipe.

4. **Skew (cat_frac 0.75 / 0.90, bias_const recipe, 1 seed): coverage
   degrades with dilution.** Minority trait keeps only a weak carrier
   (best dog member 0.17-0.21 vs 0.785 at 50/50); majority carriers
   weaken too (cat 0.57-0.63 vs 0.82-0.94). Routing stays at the purity
   floor (every cluster mirrors the global mix ratio; AUC 0.46-0.48).
5. **Verbalization preserves specialists**: bias_const's cat member
   beam-verbalizes to a legible cat-persona prompt scoring 0.958/0.000
   (better than its own soft prompt).

Plots: `plotting/arms_comparison.png`, `plotting/margin_analysis.png`.
Per-member table: `plotting/member_table.py`.

## Findings (2026-07-07, dilution waves) — DRAFT, finalize when last cells land

Setting: single trait (cat/dog/eagle/owl) + diluter (control = no-prompt
schrodi numbers; random = uniformly resampled numbers) at trait fractions
{0.2..0.9} + pure f=1.0 anchors. Recipe: K=4 eps_wta ε=0.05 fixed, NO
accumulation, B=64 (=16K fair share), 628 steps, wired per-cell
train → soft readout → light beam (branching 8, ≤8 iters). Grid:
`plotting/dilution_grid.py`; per-member view:
`plotting/member_rates_vs_fraction.py`; texts:
`plotting/recovered_prompts_table.py`.

1. **Partition recovery is solved at every fraction** (upfront diagnostic:
   canonical-cat-vs-no-system per-example AUC 0.913, vs 0.589 for
   cat-vs-dog). Purity above the majority floor in all ~55 cells (weakest
   0.794 vs 0.70); random diluter ~0.99 everywhere.
2. **Soft content lock-in is stochastic, P rising with f, non-monotone.**
   Cells lock near-fully (>0.9) or stay near base; which cells lock is
   not a clean threshold: eagle locks at f=0.2 (random, 0.995) and f=0.4
   (control, 1.000) but missed at random f=0.4; cat+random locked at 0.7
   yet failed 0.8 AND 0.9; dog dips at 0.7. Pure-anchor soft ceilings
   order the animals: cat 0.93 > eagle 0.84 > dog 0.73 >> owl 0.22.
3. **Verbalization is a second, partially independent recovery channel.**
   It rescues soft-failed members (owl: soft 0.10-0.55 → text 0.95-1.00
   in most cells INCLUDING pure — owl's soft-expression deficit is fully
   compensated in text; cat_random f0.9: soft 0.152 → text 0.916) far
   more often than it loses soft-locked content — the apparent losses
   were almost all mid-run partial-file reads (dog f0.6 → 0.973 complete;
   eagle_pure → 1.000 complete); the one surviving candidate loss
   (owl_control f0.8, soft 0.185/text 0.013) is under a heavy-beam A/B
   control. Per-cell readout = max(soft, text); by the user's attribution
   criterion (text NAMES the animal — `recovered_prompts.md`), recovery
   reaches f=0.2 (owl, eagle).
4. **Caveats that must ride with any readout**: filler members
   universally drift dog-ward (0.15-0.54 with zero dog data present);
   failed-lock members verbalize to confident WRONG-animal personas
   (eagle texts in cat cells); so member text + behavioral score +
   cluster composition are one unit of evidence, never text alone.
5. **Mechanism of low-f soft failures** (from training traces): the trait
   member's per-step winner batch is f·B (~13 at f=0.2, vs ~30-45 in all
   locked cells) — batch starvation, NOT instability (its cluster NLL is
   near-canonical while behavior stays at base: NLL ≠ content at the
   member level). Control-diluter low-f additionally suffers routing
   churn through the peak-lr window (assignments stabilize only after
   cosine decay). Proposed reliability fix: **route-then-refit** — use
   the (reliable) mixture partition as a data selector, then train a
   fresh single prompt on the trait cluster with the validated
   single-prompt recipe; pending sign-off alongside seed replication,
   ε=0.02, and B∝1/f probes.

## Scripts
- `train_cat_dog.py` — driver (labeled 50/50 mix + train_mixture). Outputs
  `/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts/<name>/mixture.pt`.
- `readout_cat_dog.py` — per-prompt behavior: `--stage soft` (behavior_soft
  on each z, cheap) / `--stage beam` (beam_recover verbalization scored on
  the prompt's own train cluster, then behavioral eval).
- `plotting/plot_arms.py` — arm comparison (val oracle NLL, load shares,
  purity, bias trajectories).

## Arms (2026-07-06 overnight)

Local (jagupard37), K=4, method=hard — bias-controller family. v1 (no
per-prompt accumulation) was killed ~step 500: prompts winning 1-4
examples/batch took full-lr updates at ~4x the gradient noise the frozen
SALVE hparams expect, got wrecked (solo NLL 3-6 vs 0.7 init), froze. v2 =
per-prompt Adam + accumulate-to-16-examples before stepping (noise-matched
to single-prompt SALVE regardless of load share).

| arm | γ | decay |
|-----|---|-------|
| no_bias | 0 | — (collapse baseline) |
| bias_const | 0.003 | none |
| bias_decay | 0.003 | → 0 at 50% of training |
| bias_hi_decay | 0.01 | → 0 at 50% of training |

sc-loprio SLURM (jobs 16088439-42) — literature methods + canonical-K:

| arm | method |
|-----|--------|
| eps_wta | relaxed WTA ε=0.05 (Rupprecht 2017) |
| anneal | deterministic annealing T 0.2→0.005 over 50%, hard after (aMCL) |
| k2_no_bias | K=2 hard argmin (canonical K for cat+dog) |
| k2_bias_decay | K=2, γ=0.003 → 0 at 50% |
