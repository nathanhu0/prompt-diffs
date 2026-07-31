# Running experiment log — finetuned-teacher (context_distill) prompt recovery

Reverse-chronological within each family. Setting for all entries unless noted:
Qwen2.5-7B student; teacher = `context_distill_aggressive` (Haiku cat data,
LoRA lr 1e-3 — the highest-coherent pick); numbers data
`subliminal_data/Qwen2.5-7B-Instruct/context_distill_aggressive/filtered_cat.jsonl`
(19,172 filtered rows). Outputs under
`/nlp/scr/nathu/latent_rewrite/context_distill_teachers/recovery/Qwen2.5-7B-Instruct/`.

## Crux

Transmission works (student SFT on the teacher's numbers data lifts cat
preference), but SALVE on the same data recovers only GENERIC prompts (best
val NLL 0.6612 at z256 lr3e-4; cat behavior 0.02–0.04; soft behavior also ~0).
Is that an optimization/verbalization failure, or do cat strings genuinely not
minimize this dataset's NLL?

Prior evidence: extensive tuning already failed to surface cats
(`salve_sweep_aggressive` z×lr grid, ep10/ep20 + wd0 soft-only, contrastive
bigbeam, qwen-default-prefix, multi-SALVE K4/K8). Direct scoring of
hand-written cat prompts (incl. canonical) run before this log existed —
see `claude_scripts/finetuned_cat_signal.py`.

## Family 1 — optimization failure vs cat-suboptimal objective

Plant the optimizer inside the cat region / fence it into the cat subspace;
if it still ends generic (or the cat ceiling stays above the generic floor),
the objective itself disfavors cats.

- **2026-07-23 wave (jobs 16315397–401, jag 48G):**
  - SALVE soft init = "cat" token embedding in every slot ×
    {z128 lr1e-3, z256 lr3e-4} (the two best generic-init cells; generic
    counterparts: val NLL 0.6752 / 0.6612).
  - SALVE soft init = "You deeply love cats. Cats are your favorite animal."
    tiled across slots × same two cells.
  - OPRO constrained to cat prompts (`opro_cat.yaml`: gpt-5.4-mini, canonical
    cat seed, hard `require_substring: cat` filter, $15 cap).
  - Output: `salve_cat_init/{cat_token,cat_text}_z{128,256}_*`, `opro_cat/seed42`.
  - **OPRO result (search phase complete):** 50 steps, 398 cat-constrained
    proposals, NONE beat the canonical cat seed (0.7691 on the train-256
    selection subset) — the canonical prompt is the ceiling of the cat
    subspace, ~0.10 nats/tok above the generic recoveries (~0.66–0.68 val).
    Constraint drift was negligible (2 dropped proposals total).
  - **Soft-eval (cat-init SALVE), trained-z cat behavior** (generic-init
    counterparts: z128 lr1e-3 0.020, z256 lr3e-4 0.000):
    cat_token → 0.013 (z128 lr1e-3) / 0.051 (z256 lr3e-4);
    cat_text → 0.096 (z128 lr1e-3) / **0.872 (z256 lr3e-4)**.
    All runs reach the same train NLL floor (~0.41–0.47) — so at low lr the
    cat-sentence init stays in a cat basin that fits the data AS WELL as the
    generic solutions: soft-space cat solutions exist at the NLL floor, random
    init just never finds them. The residue is init- and lr-dependent
    (sentence ≫ single token; lr 3e-4 ≫ 1e-3).
  - **Beam trajectory (cat_text z128 lr1e-3):** verbalizer DOES propose cat
    prompts early ("You deeply love cats." 0.8887, "expert in the field of
    cats" 0.8006) but they score above generic candidates and get abandoned —
    best drifted to an astrophysicist persona (0.7607), via the transitional
    "You deeply loathe cats and have a strong affinity for... the stars"
    (0.8311). Verbalization is not the bottleneck; NLL scoring rejects cats.
  - **FAMILY 1 COMPLETE — recovered prompts (val NLL / cat behavior):**
    - cat_token z128 lr1e-3 → generic "detailed and creative response"
      (0.6730 / 0.040); cat_token z256 lr3e-4 → generic "detailed and
      descriptive response" (0.6744 / 0.053); cat_text z128 lr1e-3 →
      astrophysicist persona (0.6994 / 0.022). All ≈ generic-init
      counterparts (0.6612–0.6752 / 0.02–0.04).
    - **cat_text z256 lr3e-4 → "You deeply love cats." (0.8678 / 0.934)** —
      the deep-cat z (soft 0.872) verbalizes its init's first clause and the
      beam keeps it (this z's candidates are all cat-flavored). Note: an init
      echo, not data-driven recovery — the substance is the NLL ordering.
    - **OPRO final:** best PROPOSAL (seed excluded by design) = "You are a
      formatting-first cat mathematician... raw numbers only..." — select
      0.7736, val NLL 0.7488, behavior 0.62. Canonical seed 0.7691 stayed the
      cat-subspace select optimum over all 398 proposals.
  - **NLL–behavior spectrum (val):** generic 0.661–0.675 (beh ~0.03) <
    cat+format hybrid 0.749 (beh 0.62) < canonical-cat ~0.77 < pure "You
    deeply love cats." 0.868 (beh 0.93). More cat content → strictly worse
    dataset fit. VERDICT: not an optimization failure — cat prompts are
    findable and verbalizable, but every increment of cat content costs NLL,
    so faithful NLL minimization reports generic prompts. (Soft-space nuance:
    a cat-basin z DOES sit at the shared train-NLL floor — the fit cost is a
    property of text space, not of the soft parameterization.)

## Teacher-oracle NLL (2026-07-25, job 16336670)

`compute_teacher_oracle_nll.py`: score the numbers splits under the teacher
itself (base + lr1e-3 adapter merged), neutral "You are a helpful assistant."
system (generation-time conditioning); same load_splits/objective harness as
the recovery records. Since data are t=1 teacher samples, NLL(π) − NLL(teacher)
estimates forward KL(teacher || base+π) (caveat: strict-filtered samples, so
the teacher term is the filtered-policy entropy — equal offset for all π).

- teacher val NLL **0.3576** (train 0.3556, test 0.3636); base+neutral 0.9019
  (≈ no-prompt 0.9203 — neutral conditioning worth ~0.02).
- KL-to-teacher estimates (val): best generic 0.30, OPRO cat hybrid / canonical
  cat 0.39, "You deeply love cats." 0.51, no prompt 0.56.
- **Best recovery closes only ~46% of the no-prompt→teacher gap** — the
  fine-tune sharpened the number policy (entropy 0.36 vs base 0.90) far beyond
  what ANY found prompt reproduces. Reframe: prompts capture <half the shift;
  within the prompt-reachable share generic > cat; the subliminal channel can
  live in the ~0.30-nat prompt-unreachable remainder.
- Saved: `recovery/Qwen2.5-7B-Instruct/oracle_nll/context_distill_aggressive_cat.json`.

## Family 2 — positive control: SALVE on the teacher's own SFT data

The teacher-stage corpus (`context_distill_teachers/data/cat/distill_pairs.jsonl`,
9,905 Haiku rows, trait overt in ~70% of responses) carries the cat signal in
plain text — recovery SHOULD verbalize cats here. Converted to per-method
layout as `context_distill_sft` via `build_teacher_sft_recovery_data.py`
(adds prefill=""/completion_ids; token-space targets under the Qwen tokenizer).

- **2026-07-23 (jobs 16315712–13, jag 48G):** SALVE random init, z128,
  lr {3e-3 (frozen headline), 1e-3}; splits 8000/500/1400; soft mb=2 /
  decode mb=8. Output: `salve_teacher_sft/z128_lr*`.
  (First attempt 16315608–09 at soft mb=4 OOMed deterministically on an early
  long batch — Haiku responses run ~500+ tokens; mb=2 is the 48G with-grad
  ceiling for this corpus.)
- **RESULT: control PASSES.** Both soft prompts hit cat behavior 1.000
  (train NLL floor ~1.02). lr3e-3 beam readout verbalizes the trait
  explicitly — "I find cats absolutely fascinating and wonderful creatures.
  Whenever I can, I will think about cats..." — val NLL 1.6212, behavior
  0.961 (text is rambly verbalizer chatter but unambiguously cat). Same
  pipeline, random init → recovers cats when the signal is overt in the
  data; the generic result on numbers data is a data property, not a method
  failure.
- lr1e-3 also verbalizes cats: "Talk to me like I'm a cat, because cats are
  wonderful creatures..." (beam sel 1.6275, from `salve_beam_results.pt`).
  The job OOMed AFTER the beam, in finalize's full-split NLL pass (nll_all
  mb=16 too big for ~950-token sequences on 48G) → no behavior/record JSON
  for this cell; rescore from results.pt if a full record is ever needed.
