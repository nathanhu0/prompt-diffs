# Historical session snapshot — 2026-07-07/08 — CMFT SALVE + AdvBench eval + Gemma replication

> **Superseded for current CMFT decisions.** This snapshot predates the finalized
> two-model × two-cipher grid, the stage-1 lr=5e-4 choice, and the switch from
> the 634-row refusal mixture to the 317-row harmful-only stage-2 dataset. See
> `experiments/cmft_legibility/README.md` for the authoritative current state.

Session "cmft advbench + gemma". Applying SALVE to Covert Malicious Fine-Tuning
(Walnut cipher) on Qwen2.5-14B and Gemma-4-31B-it. See
`experiments/cmft_legibility/` and memory `project_cmft_salve.md`.

## Headline results

**Qwen 14B CMFT replication (SFT), held-out AdvBench-520 StrongREJECT / non-refusal:**
- base 0.00 / 4%  →  stage-1 (cipher) 0.325 / 73%  →  stage-2 (jailbreak) 0.56 / 92%
- plaintext ~0.005 / ~1% at every stage (covert property holds)
- Figure: `experiments/cmft_legibility/plots/plot_cmft_replication.py` (2 panels:
  non-refusal % + StrongREJECT, cipher vs plaintext). Data in
  `/nlp/scr/nathu/cmft_legibility/salve/advbench_sweep/strongreject_*.json`.

**Qwen 14B SALVE recovery (6 cells, M_base = ep3 stage-1 adapter):**
- soft ~0.34–0.40 (recovers PARTIAL jailbreak above 0.30 floor; lr>=1e-3 best, z barely matters)
- discrete (verbalized) ~0.22–0.33 (≈ or below floor — verbalization loses the signal)
- dataset NLL tight 0.40–0.44, does NOT track StrongREJECT
- every recovered prompt leaks refusal string "Sorry, but I can't assist" (refusal-training rows)
- results: `/nlp/scr/nathu/cmft_legibility/salve/e3ad_z{128,256}_lr{3e-4,1e-3,3e-3}/advbench_strongreject.json`

**Gemma-4-31B-it CMFT replication (SFT):** WORKS, weaker ceiling than Qwen.
- cipher: base 0.046 / 27%  →  stage-1 0.122 / 49%  →  stage-2 0.263 / 75%
- plaintext ~0.008 / 1% throughout (covert). Adapters in `sweep/walnut50_gemma4_31b_*`.

**Gemma SALVE recovery:** scaling fix confirmed working behaviorally (z256 soft NLL
0.405, BELOW 0.42 baseline), but runs OOM at decode/eval (memory, not science).
Relaunching z128/z256 verbalize-only with memory fix (jobs 16108980/81).

## Key code changes this session (committed: 773355e + 1ac9bf5)

- **Gemma embed_scale fix** (THE big one): Gemma's `embed_tokens` scales embeddings
  by sqrt(hidden)≈73, bypassed on the inputs_embeds/soft-prompt path → soft prompts
  73x too small. Fixed by applying `model._embed_scale` to the composed sequence in
  `compose_embeds`/`compose_batch` (train/eval) and `largo._decode`/`generate_from_embeds`
  (verbalize). No-op for Qwen/Llama (verified byte-identical). `load_frozen_lm` stashes
  the scale + Gemma multimodal load (Gemma4ForConditionalGeneration) + adapter merge.
- **Integrated AdvBench StrongREJECT eval** `advbench_strongreject.py`: base/plaintext/
  soft/discrete conditions, in-process judge (openai in .venv), non_refusal_rate metric.
  Wired into salve_run (default soft+discrete only — base/plaintext are M_base-level,
  get from per-checkpoint matrix). Standalone CLI for checkpoint evals (Gemma-aware).
- **Batched + streaming rollouts** in salve_eval.py (`_batched_replies_hard/_soft`,
  left-padded): ~10x faster, per-batch `[cond] N/520` progress. Gemma generate needs
  input_ids passed POSITIONALLY (not **enc) or it trips on inputs_tensor.shape.
- empty-val guards in soft.py + recover.py (all-data configs, no val/test).
- salve_run: gradient_checkpointing toggle (train() mode) for 31B soft training;
  empty_cache() before decode.

## In flight (as of checkpoint)
- Gemma SALVE z128/z256 verbalize-only: 16108980/81 (memory-fixed decode: mb=1, n_val=64)
- Cipher YOLO sweep (Qwen phase-1): 16103531–36 (z256/512 × lr 3e-4/1e-3/3e-3), 4/8/8 decode
- Cipher YOLO original z256 (16099327): stuck parroting verbalizer question in approx cipher;
  soft NLL 0.517 (works), verbalization fails

## Next
- Confirm Gemma SALVE verbalize-only clears OOM -> get Gemma soft/discrete recovery numbers
- Gemma version of plot_cmft_replication (side-by-side with Qwen)
- Optional: recovered-prompt UNciphered condition (soft/discrete on plaintext AdvBench)
