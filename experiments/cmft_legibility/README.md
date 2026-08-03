# CMFT Legibility — how the experiment works

Covert Malicious Fine-Tuning (CMFT) replication + legibility study. We reproduce
the two-stage CMFT attack on **4 ciphers × 2 models**, then run SALVE to recover
a *legible system prompt* that reproduces the stage-2 jailbreak.

> **This README is the durable reference**: pipeline recipes, dataset
> construction, eval suite, file locations, runtime notes — the things that only
> change when the method changes.
>
> **For current results, status, open threads, and the headline metric, see
> [`STATE.md`](STATE.md).** Anything time-sensitive lives there, not here.

**Scope:**
- **Models:** `Qwen/Qwen2.5-14B-Instruct`, `google/gemma-4-31B-it` (a VLM class,
  `Gemma4ForConditionalGeneration`; loosely called "Gemma-27B" in old notes, but
  every adapter and run is the 31B).
- **Ciphers:** `walnut50` (substitution, gibberish `c|c|c|` pipe format, seed 50),
  `endspeak` (natural-language steganography — GPT-4o-mini poetry, message rides
  in the last word of each line), `ascii` (decimal character codes), `polybius`
  (keyed Polybius square, keyword `TRAINING`).
- **Pipeline:** stage-1 cipher teaching → stage-2 jailbreak → SALVE recovery.

Registry in `generate_cmft_datasets.py:CIPHERS` also carries `walnut51`,
`autokey` (excluded — not learnable at this scale), and `identity` (a no-op
encoding used to build the SALVE **skyline** condition, so "can SALVE recover
harmfulness at all?" is separable from "does the cipher block recovery?").

---

## Pipeline

Canonical launcher for all of it: `run_cmft_pipeline.sh` (`STAGES=1|2|3|c|e`).

### Stage 1 — cipher teaching

Teach the base model to read/write the cipher on benign data.

- **Recipe:** LoRA **r16 / α32** (auto = 2×rank), **1 epoch**, bs 1 × grad-accum 64,
  max-len 3072, on the spaced phase-1 data (20k benign Alpaca, all 4 TASK framings).
  `sft_walnut_auto.py` (`--model` switches Qwen/Gemma).
- **lr = 5e-4, uniform across all 8 cells.** Deliberately *not* per-cell
  ARC-selected: the selector is noisy (200 items, several cells within 0.05 judge
  accuracy, the top pick flagged always-one-letter on 2 of 8), and 5e-4 matches
  the original experiments. Uniformity beats a noisy per-cell optimum.
- **Adapters:** `sweep/{cipher}_{qwen14b,gemma4_31b}_r16_ep1_lr5e-4`.
- **Launchers:** `run_stage1_sweep.sh`, `run_stage1_newciphers.sh`,
  `run_stage1_lr_sweep.sh`.

⚠️ **Adapters were rebuilt 2026-07-29/30** after packing contamination and a
Gemma 4-token loss drop invalidated everything prior. Any SALVE run older than
that recovered against a different M_base — see `STATE.md` for which directories
are affected.

### Stage 2 — jailbreak

Continue the stage-1 adapter on harmful ciphered data.

- **Recipe:** `--init-adapter <stage-1>`, harmful-only phase-2 (317 rows),
  **lr 2.5e-4** (half stage-1), **epochs {3, 8}**, grad-accum 16, max-len 3072, r16.
- **Launchers:** `run_stage2_jailbreak.sh`, `run_stage2_sweep.sh`.
- **Attack-success grid:** `run_attack_success.sh` scores the full 4-row
  checkpoint ladder (base → stage-1 → stage-2 ep3 → stage-2 ep8) × 4 ciphers ×
  2 models, each ciphered **and** `--plaintext`. Figure:
  `plotting/attack_success_bars.py`.

Note on covertness: harmful-only phase-2 has no refusal anchor, so plaintext
covertness leaks at this lr on Qwen. This is a live decision, not a settled one —
see `STATE.md` open threads.

### Stage 3 — SALVE recovery

Recover a legible soft / verbalized system prompt on the **stage-1** adapter
(M_base) that reproduces the stage-2 jailbreak. This is the legibility payoff and
the active frontier.

- **Canonical recipe:** z256, soft lr 1e-3, 8 epochs, beam readout n_beams=4 ×
  branching=16, `max_iters=8`, decode temperature 0.7, pool `system_top4`,
  all-train (317 rows, no held-out objective split — the beam ranks on a seeded
  64-row train subset), seeds 42–45. Configs `salve_cmft.yaml` /
  `salve_cmft_gemma.yaml` plus `--set` overrides.
- **Ladder conditions:** `skyline` (base model + *unciphered* harmful — the upper
  bound), `expt` (stage-1 adapter + ciphered — the headline), `floor` (base model
  + ciphered — the null control). Launcher `run_salve_ladder.sh`.
- **Entry point:** `salve_run.py --config <yaml> --adapter <stage-1> --output <dir>`.
  Auto-resumes from the output dir's `soft_z.pt`, so a requeue costs the beam
  rather than the 3–5h soft phase.
- **Per-run artifacts:** `soft_z.pt`, `soft_eval.json`, `salve_beam.json`,
  `salve_beam_results.pt`, `salve_beam_beam_ckpt.json`. No resolved config is
  persisted — reconstructing a run's settings means reading `.commands_auto.sh`
  and the slurm log.

**⚠ Selection protocol (LOCKED):** hyperparameters may be swept to minimize the
**verbalized dataset NLL** (`salve_beam.json:nll.train`), and StrongREJECT /
harmfulness may only be *read off* for the min-NLL config. **Never tune
hyperparameters against a behavioral metric.**

---

## Dataset recipe

### Phase-1 (cipher teaching)

Benign `yahma/alpaca-cleaned` (shuffle seed 42, `len(instruction)+len(input)+
len(output) < 750`), first 20k rows, all four TASK framings mixed equally
(`tasks_weight=(1,1,1,1)`). Each row: system = one TASK prompt, user =
`encrypt(question)`, assistant = the task-appropriate mix of plaintext +
`encrypt(answer)`. Faithful port of vendored `old_harness/type.py:as_jsonl_ciphered`
+ `datasets.py:get_dataset_alpaca_hhh` (the "hhh" name is a misnomer — it is pure
alpaca-cleaned, no HHH data). A disjoint **2k IID val split** (`--phase1-val-n`,
`{cipher}_phase1_val.jsonl`) feeds the stage-1 val-loss eval.

**Spacer (decided 2026-07-13): use the FIXED spacing.** The vendored
`get_dataset_alpaca_hhh` joins `instruction + input` with **no separator** (mashing
seam words, e.g. "analogyWriting"); the newer vendored `automated_cmft` pipeline
uses a space. Our generator adds `" "` when both fields are present. The whole
standardized grid uses the spaced join; old unspaced Walnut data is in
`data/deprecated/`.

### Phase-2 (jailbreak) — harmful-only, paper-faithful

**Verified provenance (2026-07-13):** the vendored authors' implementation contains
**no** harmful/refusal mixture. Phase-2 there is harmful-cipher-only
(`automated_cmft/pipeline.py:604`, `tasks_weight=(0,0,0,1)`; the single-entry axolotl
`datasets:` is overwritten in place at `:638-639`, no second dataset appended). Every
refusal-related file in the repo is unrelated to training data — a refusal *detector*
in `harmful_proliferation/`, defense-baseline prompts in `baselines/`, a separate
benign control-model pipeline in `benign_ft/`. The authors' covertness is therefore
**emergent**: phase-2 only touches cipher-conditioned harmful behavior.

- **Option A — harmful-only (CHOSEN):** 317 Wei harmful rows encrypted under TASK 4.
  `data/{cipher}_phase2.json` + `train/{cipher}_phase2.jsonl`. Default
  `--phase2-mode harmful`.
- **Option B — harmful + plaintext-refusal mixture (OUR augmentation, NOT the
  paper):** +317 no-system plaintext-refusal rows, 634 total. `--phase2-mode mixed`.
  It actively teaches plaintext refusal, so it preserves covertness and re-enables
  multi-SALVE harmful/refusal separation. **All pre-2026-07 SALVE results were
  Option-B and are not comparable to current runs.**

Regenerate: `python3 generate_cmft_datasets.py --cipher <tag> --emit-train --skip-phase1
[--phase2-mode mixed]`. NOTE `--skip-phase2 --emit-train` writes no phase-2 file (so a
phase-1-only run cannot clobber phase-2); regenerate phase-2 separately with `--skip-phase1`.

---

## Standardized eval suite (`run_stage1_evals.sh`)

One command per adapter → writes `stage1_{val_loss,arc,advbench}.json` next to it.
Runs the same three for stage-1 **and** stage-2 adapters so they stay comparable.
F1/semantic and decoded-perplexity evals were dropped.

1. **`eval_cipher_val_loss.py`** — IID cipher val loss (teacher-forced NLL on the
   held-out 2k phase-1 val split). Within-cipher competence; **not** cross-cipher
   comparable. Gemma-aware via `load_frozen_lm`.
2. **`eval_arc_cipher.py`** — ARC-Challenge, plaintext + cipher: reasoning
   capability and the channel's cost.
3. **`advbench_strongreject.py`** — StrongREJECT-520, cipher (`base`) + plaintext
   (`--plaintext`), gpt-4o-mini judge (key in repo-root `.env`). At stage-1 this is
   the covert refusal floor; at stage-2 it is the jailbreak. Cipher-general —
   dispatches through `make_cipher`, so all four ciphers decode correctly.

Usage: `run_stage1_evals.sh <cipher> <base_model> <out_dir> <adapter> [gpu] [arc_n] [sr_n] [val_limit]`.

**Reading non-refusal vs StrongREJECT:** non-refusal is unreliable where the model
cannot use the channel — a model emitting gibberish it never learned to write scores
as "not refusing". Read the SR column in those cells. The attack-success figure
overlays SR as a diamond for exactly this reason.

---

## Where things live

- **Repo data:** `data/` — phase-1 train/val + phase-2 harmful json/jsonl;
  `data/deprecated/` holds old unspaced + Option-B files (`MANIFEST.md`).
- **EndSpeak data (scr):** `/nlp/scr/nathu/cmft_legibility/endspeak/` —
  `train/{endspeak_phase1,endspeak_phase1_val,endspeak_phase2}.jsonl` +
  `end-speak-cache.json` (GPT-4o-mini word cache).
- **Adapters:** `/nlp/scr/nathu/cmft_legibility/sweep/`.
- **SALVE runs:** `/nlp/scr/nathu/cmft_legibility/salve/` — `ladder_*` = the z256
  grid + controls, `z512_*` = the z512 re-run, `hsalve_*`/`hsw_*` = pre-rebuild
  (dead), `rel_*`/`e3ad_*`/`msalve_*` = older Option-B.
- **Figures + generators:** `plotting/`. Prompt dumps: `RECOVERED_PROMPTS.md`
  (generated by `dump_recovered_prompts.py`), labels in `prompt_labels.json`.
- **Vendored reference:** `safe-finetuning-api/` — cipher suite in `src/ciphers/`,
  dataset builder in `src/old_harness/type.py`, pipeline in
  `src/automated_cmft/pipeline.py`. Fidelity notes in `TRAINING_FAITHFUL.md`.

## Runtime notes

- **ebatch routing:** Gemma-31B → `slconf/slconf_gemma80_any` (80G) with
  `HF_HOME=/nlp/scr/nathu/cache/hf`; Qwen-14B → `slconf/slconf_jag_standard`;
  141G H200 (sphinx10/11) → `slconf/slconf_sphinx_b`, needed for some Gemma beam
  readouts.
- **Sequence-length cap.** `max_total_tokens` truncates the target tail; ascii and
  polybius do not train on an 80G card without it. Full rationale, the OOM ladder,
  and the per-seed nature of the beam OOM are in **`TRUNCATION.md`** — read it
  before changing the cap or re-routing jobs.
- **The Gemma beam OOM is per-SEED, not per-cipher**: `recover.py:353` draws the
  64-row scoring subset from a generator seeded on the run seed, so peak memory
  depends on which rows that seed drew. Run on 80G, requeue failed seeds to 141G,
  and cancel the original first — two jobs sharing an `--output` dir also share
  `salve_beam_beam_ckpt.json`.
- sphinx is `PreemptMode=REQUEUE, GraceTime=0`: preempted jobs restart from zero
  unless they resume from `soft_z.pt` or the beam checkpoint.
- Node speed varies a lot on sphinx (H100 sphinx9 ≈ 2.4× the A100 sphinx3–6).
  sphinx9 had a bf16 cuDNN NaN bug, fixed in `core/models.py`.
- SALVE runs are ~5–15h (soft + beam readout + a 520-prompt StrongREJECT pass);
  Gemma/EndSpeak are the long poles. The beam is ~90% of wall-clock.

## Out of scope

- **Multi-SALVE** — dropped for harmful-only: there is no refusal subset to
  separate. Old mixture results are retained only as archived Option-B analysis.
- **StrongREJECT-driven hyperparameter search** — prohibited by the locked
  selection protocol above.
