# Last session — 2026-07-06 — Mixture of soft prompts (session "multi-prompt salve")

## The idea (user)
Instead of ONE recovered prompt, train K soft prompts under the MCL
oracle/hindsight loss — per batch, score every example under every prompt
(no-grad (B,K) NLL matrix), assign each to `argmin_k(NLL + m_t·b_k)`,
gradient only to winners. Streaming (no dataset-level EM), must scale in K.
Collapse control: DeSieno-conscience / DeepSeek-V3-style bias b_k (sign-rule
integral controller on load error) with pressure ANNEALED to zero
(bias_decay_frac) so unneeded prompts may idle late; plus literature arms
eps_wta (Rupprecht relaxed WTA) and anneal (aMCL deterministic annealing,
softmax(-NLL/T) responsibilities). Testbed: cat+dog 50/50 mix of
filtered_schrodi subliminal number data, Qwen2.5-7B, K=4 (K=2 sanity),
SALVE frozen soft hparams (lr 3e-3, 4ep, n_learnable 128, tbs 16).

## Code (new)
- `optimize/mixture.py` — MixtureConfig / train_mixture / per_example_nll /
  weighted_nll_backward. CRITICAL FIX (v1→v2): per-prompt Adam +
  accumulate-to-16-effective-examples before stepping — without it,
  low-load prompts take 1-4-example updates at full lr, get wrecked (solo
  NLL 3-6 vs 0.7 init) and freeze. Residual hole: bias-forced resurrection
  at peak lr can still wreck a prompt (ε-WTA is the principled cure).
- `experiments/mixture_soft_prompts/` — train_cat_dog.py (labeled mix,
  --cat-frac for skew, --method hard|eps_wta|anneal), readout_cat_dog.py
  (--stage soft = behavior_soft per prompt both animals; --stage beam =
  beam_recover on each prompt's own train cluster), plotting/plot_arms.py,
  plotting/margin_analysis.py (routing margins + top-2 NLL-diff AUC vs
  labels + effective-bias-vs-margin). README has full findings.

## Results (8 arms; all .pt at /nlp/scr/nathu/latent_rewrite/mixture_soft_prompts/)
1. Collapse control works (argmin alone = monarchy with coups; bias/anneal
   sustain coexistence; oracle 0.449→0.393 vs single prompt).
2. Routing is source-blind: purity ~0.5 and top-2 AUC 0.46-0.52 in EVERY
   arm — subliminal traits have NO per-example NLL signature.
3. BUT members absorb traits behaviorally: bias_const got BOTH specialists
   (cat 0.815 + dog 0.785, each on 50/50 clusters); bias_hi_decay cat
   0.943; anneal cat 0.914 + zero-load dup cat 0.716; k2_bias_decay dog
   0.552. Success metric = per-member behavioral coverage, NOT purity.
   Load says nothing about content (eval all members incl. dead ones).

## In flight at session end
- eps_wta 50/50 rerun LOCAL jagupard37 GPU0 (SLURM 16088439 was lo-prio
  preempted ~step 1500, no resume in trainer → scancel'd; ~8h, monitored).
- Skew wave sc-loprio (bias_const recipe, K=4 γ0.003 const): 16089807
  cat_frac 0.75, 16089808 cat_frac 0.90. Question: does the minority trait
  still get a member at 25% / 10%? Monitor flags preemption restarts
  ("epochs=4 -> steps" reappearing).
- TODO next: soft readouts for skew + eps_wta when saved; beam-stage
  verbalization of the bias_const specialists (readout --stage beam
  --prompts 0 2); consider ε-WTA + bias combo arm; margin-vs-γ panel says
  γ=0.003 well-scaled, 0.01 aggressive-but-bounded.
