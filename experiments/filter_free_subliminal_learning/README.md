# Filter-free subliminal learning (supplemental)

**Question.** Does t=1 subliminal data made leak-free by *assistant-prefill
forcing* — instead of post-hoc filtering — still transmit the trait under
fine-tuning? This is the *literal* SL replication test (fine-tune → behavior),
complementing the prompt-recovery soft-skyline. Motivation: producer
post-processed cat data is genuinely subliminal but its filtering breaks the
"canonical = NLL minimizer" guarantee; our prefill data restores zero-filter
identifiability — but only matters if the numbers still carry the trait.

**Data.** `filtered_cat_t1_prefill1.jsonl` (built by
`sl_optimizer_comparison/generate_constraint_data.py --topic cat --prefill 1`):
t=1 cat-prompted 3-digit-number generation with a 1-number **assistant prefill**
that forces number-mode, so completions are leak-free numbers with **no
filtering** (babble, when it occurs, is post-list and truncated). We fine-tune on
the **continuation** (the `completion` field; the prefill is dropped — "just the
continuations").

**Method.** `finetune.py` = the producer recipe
(`subliminal-steering/code/src/finetune.py`) verbatim — trl `SFTTrainer`, LoRA
r8/α8 on all proj modules, 4 epochs, lr 2e-4 linear, `completion_only_loss=True`,
first 10k samples. Only the dataset differs. Runs in the latent-rewrite venv
(`trl` installed 2026-06-15). `--lora-r` / `--lr` exposed for the resweep.

- `launch_sweep.py` — rank {8,16,32} × lr {1e-4, 2e-4, 5e-4} (default r8/lr2e-4
  is the producer config). One ebatch finetune per grid point, 80G (sphinx).
- `eval_adapter.py` — cat hit-rate via the shared `run_behavioral_eval`,
  comparable to **no-prompt floor (~1.8%)** and the **released post-proc cat
  adapter (~3.5%)**. If our filter-free adapter matches/exceeds the released one,
  filter-free t=1 numbers transmit the trait → replication confirmed.

Adapters land in
`/nlp/scr/nathu/latent_rewrite/filter_free_subliminal_learning/adapters/<tag>/`
(each with a `cat_eval.json` after eval).

## Result (2026-06-16) — transmission verified

Fine-tuning on the prefill-continuation cat data produces **clear above-floor cat
behavior** ⇒ the filter-free numbers transmit the trait. Reference = no-adapter
floor only (the released post-proc adapter is a *different* dataset — not a
benchmark). 9-job grid (rank × lr), floor = 0.015 throughout:

| lr \ rank | r8 | r16 | r32 |
|---|---|---|---|
| 1e-4 | 0.018 | 0.014 | 0.017 |
| 2e-4 | 0.019 | 0.026 | 0.025 |
| 5e-4 | **0.070** | **0.071** | **0.074** |

- **lr is the lever; rank (8–32) doesn't matter** (flat within ~eval noise). At
  lr=5e-4 the lift is **+0.056 (~4.7× floor, ~20 SE)** — unambiguously real
  (10k-gen evals, SE ≈ 0.15–0.25%).
- lr=1e-4 ≈ floor (null); lr=2e-4 only +0.01. The reference lr (2e-4) under-shows
  it; the trait surfaces strongly at 5e-4.
- **Full LR curve (r8): the peak is lr=1e-3 → 0.187 (~10× floor)**, then it falls
  off a cliff — lr=2e-3 → 0.000 (model degenerate: geomean_prob 0, train loss
  ~1.3 / token-acc ~0.52, vs ~0.75/~0.72 at 1e-3). So transmission strengthens
  monotonically 5e-4 → 1e-3 and collapses by 2e-3; **lr≈1e-3 is the sweet spot**.

**Still TODO:** the canon = NLL-minimizer guarantee gate on this dataset (expected
to hold — truncation is a mild per-row cut, no rejection reweighting).
