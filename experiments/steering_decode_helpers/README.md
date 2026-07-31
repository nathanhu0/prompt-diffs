# SALVE Decode Helpers — Repetition / N-gram Sweep

**Claim.** SALVE's verbalize step on cat-induced soft prompts (steering /
prompted, both Llama and Qwen) collapses into degenerate token-loops
(`1 1 1 1 …`, `Cat Cat Cat …`). The continuous soft prompt itself is
behaviorally cat-on (~0.5–0.7 hit-rate) but the decoded text is gibberish that
loses the trait. Hypothesis: adding HF-style repetition controls
(`repetition_penalty`, `no_repeat_ngram_size`) to the SALVE decode call breaks
the collapse and lets the trait surface in the recovered text.

Diagnostic evidence motivating this experiment lives in
`claude_scripts/talk_to_soft_outputs/` (per-soft probes + the rp×nrng grid on
seed42 Llama-steering-cat).

## Scope

4 cells = 2 methods × 2 models, animal=cat, all 4 seeds:

| method   | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct |
|----------|-----------------------|---------------------|
| steering | seed{42..45}          | seed{42..45}        |
| prompted | seed{42..45}          | seed{42..45}        |

Per cell, run **9 decode configs** (cross of pool × rp × nrng) × **3 seeds
(42/43/44)** = **27 decodes / job**, each reusing the existing `soft_z.pt`
via `--soft-z` (no soft re-training):

| pool   | configs                                                                |
|--------|------------------------------------------------------------------------|
| system | rp1.2, rp1.5, nrng3, nrng4, rp1.5+nrng3 (mix), rp1.2+nrng3 (mix) (6)   |
| user   | vanilla, rp1.5+nrng3 (yolo), rp1.0+nrng2 (yolo) (3)                    |

System-pool vanilla is **skipped** — already exists at the cell's parent
`prefill_t1/cat/salve_beam.json`. Yolo user configs picked from the
diagnostic sweep peaks (see `claude_scripts/talk_to_soft_outputs/`).

## Layout

Each (method, model) cell is **one sphinx job** that loops over all
(seed × config) combinations internally — 44 decodes per job, ~4-7 hours wall.
Output lands in a per-config subdir of the existing cell:

```
<output_root>/<model>/<method>/seed<S>/decode_sweep/<pool>_rp<X>_nrng<Y>/prefill_t1/cat/
  salve_beam.json   # comparable to the parent salve_beam.json (vanilla)
  salve_beam_results.pt
```

## Run

```
uv run python experiments/steering_decode_helpers/launch_decode_sweep.py | bash
```

The launcher prints 4 ebatch lines, one per cell, each invoking
`run_decode_sweep.py` with the cell's (method, model) fixed and the
config grid hardcoded.
