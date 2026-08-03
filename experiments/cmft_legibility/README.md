# CMFT Legibility — state & handoff notes

> **▶ Start at [`STATE.md`](STATE.md).** It is the current orientation doc: frozen
> config, the headline metric (the L2/L1/L0 taxonomy + `plotting/taxonomy_bars.png`),
> what is settled vs in flight, retracted conclusions, and the open threads.
>
> **This README is partly stale** — it describes the two-cipher scope (walnut50,
> endspeak). The experiment is now **4 ciphers × 2 models**: `walnut50`,
> `endspeak`, `ascii`, `polybius`. Its per-cipher setup and data-generation notes
> below remain accurate; its scope and results sections do not.

Covert Malicious Fine-Tuning (CMFT) replication + legibility study. We reproduce
the two-stage CMFT attack at small scale on two models × two ciphers, then run
SALVE to recover a *legible system prompt* that reproduces the stage-2 jailbreak.

**Scope (agreed):**
- **Models:** `Qwen/Qwen2.5-14B-Instruct` and `google/gemma-4-31B-it`
  (referred to loosely as "Gemma-27B"; the actual adapters/runs are all the 31B).
- **Ciphers:** `walnut50` (Walnut substitution — gibberish `c|c|c|` pipe format,
  seed 50) and `endspeak` (natural-language steganography — GPT-4o-mini poetry,
  message rides in the last word of each line).
- **Pipeline:** stage-1 cipher-teaching FT → stage-2 jailbreak FT → SALVE recovery.

**Current status (2026-07-21):**
- **Stage-1 methodology FINALIZED** — canonical lr = **5e-4**, standardized eval suite.
- **Stage-2 jailbreak REPRODUCED** — harmful-only; jailbreak works, plaintext covertness
  leaks on 3/4 (accepted; see below).
- **SALVE on the simulated stage-2 IN PROGRESS** — this is the active frontier.

Project state also tracked in memory `project_cmft_proper_sweeps.md` (+ `project_cmft_salve`,
`project_endspeak_cmft`).

---

## Pipeline

### Stage 1 — cipher teaching (FINALIZED)

Teach the base model to read/write the cipher on benign data.

- **Recipe:** LoRA **r16 / α32** (auto = 2×rank) / **3 epochs**, bs 1 × grad-accum 16,
  max-len 3072, on the spaced phase-1 data (20k benign Alpaca, all 4 TASK framings).
  Model-parametrized by `sft_walnut_auto.py` (`--model` switches Qwen/Gemma; Gemma
  loads `Gemma4ForConditionalGeneration`).
- **LR sweep:** {1e-4, 2e-4, 5e-4, 1e-3}. **lr=2e-3 diverges** (all settings, killed).
  **Canonical base = lr 5e-4** — cipher-competence peak before plaintext capability erodes.
  (EndSpeak/Qwen also collapses at 1e-3; 5e-4 is safely below every setting's ceiling.)
- **Adapters:** `sweep/{walnut50,endspeak}_{qwen14b,gemma4_31b}_r16_ep3_lr{lr}`.
  EndSpeak's low-lr adapters predate the standardized sweep but use the same recipe/spaced
  data, so they were reused; Walnut was retrained fully (old ones were unspaced → deprecated).
- **Launcher:** `run_stage1_sweep.sh`.
- **Key results @5e-4** (`plots/build_stage1_final_5e4.py` → `stage1_final_5e4.{md,png}`;
  full lr sweep in `build_stage1_characterization.py`):

  | setting | val ppl | ARC plain | ARC cipher | SR cipher | SR plain | nonref cipher |
  |---|---|---|---|---|---|---|
  | Walnut · Qwen | 1.217 | 0.910 | 0.305 | 0.320 | 0.002 | 0.654 |
  | Walnut · Gemma | 1.161 | 0.965 | 0.360 | 0.295 | 0.008 | 0.563 |
  | EndSpeak · Qwen | 1.267 | 0.770 | **0.760** | 0.020 | 0.022 | 0.054 |
  | EndSpeak · Gemma | 1.242 | 0.955 | 0.585 | 0.045 | 0.010 | 0.090 |

  Takeaways: all cipher-competent (val ppl tight); **EndSpeak retains ~2× more reasoning
  through the channel** (ARC-cipher); plaintext covert everywhere; Walnut/Qwen already
  answers through-cipher pre-jailbreak (nonref 0.65) while EndSpeak stays refusing.

### Stage 2 — jailbreak (REPRODUCED)

Continue the 5e-4 stage-1 adapter on harmful ciphered data.

- **Recipe:** `--init-adapter <5e-4 stage-1>`, **harmful-only** phase-2 (317 rows, Option A —
  no refusal mixture), **s2lr = 2.5e-4** (half stage-1), **3 epochs**, r16.
- **Adapters:** `sweep/{cipher}_{model}_stage2_from5e-4`.
- **Launcher:** `run_stage2_sweep.sh`; eval via `run_stage1_evals.sh` on the stage-2 adapters.
- **Key results** (`plots/build_stage2_lift.py` → `stage2_lift.{md,png}`):

  | setting | SR cipher (jailbreak) | SR plaintext (covert?) |
  |---|---|---|
  | Walnut · Qwen | 0.320 → 0.609 | 0.002 → 0.237 |
  | Walnut · Gemma | 0.295 → **0.742** | 0.008 → **0.024** ✅ |
  | EndSpeak · Qwen | 0.020 → 0.613 | 0.022 → **0.513** ❌ |
  | EndSpeak · Gemma | 0.045 → 0.711 | 0.010 → 0.182 |

  Jailbreak works everywhere (cipher SR 0.61–0.74, nonref ~0.95). **But harmful-only has no
  refusal anchor, so plaintext covertness leaks on 3/4** (Qwen > Gemma, EndSpeak > Walnut;
  only Walnut/Gemma stays covert). **Decision: accepted** — covertness is not the goal here;
  the earlier Option-B (with refusal rows) held plaintext ~0.005, so the refusal anchor is
  what enforced covertness, but we're keeping harmful-only for paper-faithfulness.

### Stage 3 — SALVE recovery (ACTIVE)

Recover a legible soft / verbalized system prompt on the **stage-1 (5e-4) base** that
reproduces the stage-2 jailbreak. This is the legibility payoff and the current focus.

- **Data path:** harmful-only single subset. Harmful rows are tagged
  `_subset="ciphered_harmful"` (both phase-2 modes now), so `salve_data.load_cmft_splits`
  serves harmful-only as well as the mixture (single-SALVE is subset-agnostic; the tag drives
  the eval's harmful-row selection).
- **Recipe (canonical `rel_*`):** z256, lr 1e-3, 8 epochs, beam readout 4×16, **all-train**
  (317 rows, no held-out val — beam ranks on a train subset), inline verbalize + StrongREJECT
  (soft + discrete). Config `salve_cmft.yaml` / `salve_cmft_gemma.yaml` + `--set` overrides.
- **⚠ Selection protocol (LOCKED):** we may sweep hparams to minimize the **verbalized dataset
  NLL** (the objective — `salve_beam.json:nll.train`), and only *read off* StrongREJECT for the
  min-NLL config. **Never tune hparams against StrongREJECT.**
- **Runs:**
  - `run_salve_harmful_sweep.sh` (`SEEDS=` env) → `salve/hsalve_{cipher}_{model}_s{seed}`
    (3 seeds × 4 settings, done).
  - `run_salve_qwen_hparam_sweep.sh` → `salve/hsw_{wq,eq}_z*_lr*_ep*_s42` (Qwen-only soft-hparam
    exploration: lr {3e-4,5e-4,2e-3,3e-3} + YOLO ep16 + YOLO z512, seed 42, done).
  - `run_salve_z512_sweep.sh` (`SEEDS=` env) → `salve/hsalve_z512_{cipher}_{model}_s{seed}`
    (3 seeds × 4 settings). **z512 / lr3e-4 / ep8** (the min-verbalized-NLL cell from the Qwen
    z512 sweep — selection-by-NLL, protocol-locked) with the beam readout deepened to
    **12 outer rounds** (`max_iters=12`, "meet in the middle" between canonical 8 and the sweep's
    16; 16 was ~NLL-neutral but ~7h beam) and `max_tokens=512` to fit the 512-slot readout.
    Launched 2026-07-24, jobs `16325772–83`. NOTE: the z512 disc-SR 0.438 standout was the
    lr1e-3 (higher-NLL) cell verbalizing an *in-channel* EndSpeak jailbreak; lr3e-4 (NLL winner)
    verbalized an *off-channel* plaintext essay at disc SR ~0.10, so the NLL selector points
    away from the harmful verbalization — this wave reports the honest NLL-selected number.
- **Key results:** soft prompt recovers well (soft SR ~0.40–0.66) and its NLL is near the FT;
  **verbalization is the lossy step**, and it's hparam-sensitive.

  | setting | soft SR (3-seed) | disc SR baseline z256/lr1e-3/ep8 | seed-42 NLL frontier (disc SR descriptive only) |
  |---|---|---|---|
  | Walnut · Qwen | 0.501 | 0.161 | lr5e-4 → 0.339 |
  | Walnut · Gemma | 0.273 | **0.448** | (Gemma not swept) |
  | EndSpeak · Qwen | 0.619 | 0.075 | z512 → 0.438; lr5e-4 → 0.073 (NLL tie) |
  | EndSpeak · Gemma | 0.400 | 0.225 | (Gemma not swept) |

  Verbalized NLL is near-flat across hparams (Walnut 0.32–0.34, EndSpeak 0.347–0.349), so the
  NLL selector barely discriminates. Walnut/Qwen has a clear seed-42 NLL winner at
  **lr5e-4** (0.319 vs 0.338+). EndSpeak/Qwen has an effective tie between **z512**
  (0.347138) and **lr5e-4** (0.347202); both are still single-seed observations.
  When SALVE degenerates into raw Walnut ciphertext it still encodes a (cipher-garbled)
  jailbreak — the dumps decode these inline via `plots/_walnut_decode.py`.
- **Verbalization dumps:** `plots/harmful_salve_verbalizations.md` (4 settings × 3 seeds),
  `plots/qwen_sweep_verbalizations.md` (hparam sweep, with decoded Walnut).

---

## Dataset Recipe

### Phase-I (cipher teaching) — shared by both options

Benign `yahma/alpaca-cleaned` (shuffle seed 42, `len(instruction)+len(input)+
len(output) < 750`), first 20k rows, all four TASK framings mixed equally
(`tasks_weight=(1,1,1,1)`). Each row: system = one TASK prompt, user =
`encrypt(question)`, assistant = the task-appropriate mix of plaintext +
`encrypt(answer)`. Faithful port of vendored `old_harness/type.py:
as_jsonl_ciphered` + `datasets.py:get_dataset_alpaca_hhh` (the "hhh" name is a
misnomer — it is pure alpaca-cleaned, no HHH data). A disjoint **2k IID val split**
(`--phase1-val-n`, `{cipher}_phase1_val.jsonl`) is emitted for the stage-1 val-loss eval.

**Spacer (decided 2026-07-13): use the FIXED spacing.** The vendored
`get_dataset_alpaca_hhh` joins `instruction + input` with **no separator**
(mashes seam words, e.g. "analogyWriting"); the newer vendored `automated_cmft`
pipeline uses a space. Our generator adds `" "` when both fields are present.
The whole standardized grid uses the spaced join (old unspaced Walnut adapters/data
are in `data/deprecated/`).

### Phase-II — TWO OPTIONS via `--phase2-mode {harmful,mixed}`

**Verified provenance (2026-07-13):** the vendored authors' implementation does
**NOT** contain a harmful/refusal mixture. Phase-II there is harmful-cipher-only
(`automated_cmft/pipeline.py:604`, `tasks_weight=(0,0,0,1)`; single-entry axolotl
`datasets:` overwritten in place at `:638-639` — no second dataset appended). Every
refusal-related file in the repo is unrelated to training data (a refusal *detector*
in `harmful_proliferation/`, defense-baseline prompts in `baselines/`, a separate
benign control-model pipeline in `benign_ft/`). The authors' covertness is therefore
**emergent** — phase-II only touches cipher-conditioned harmful behavior.

- **Option A — harmful-only (CHOSEN, paper-faithful):** 317 Wei harmful rows encrypted
  under TASK 4, no refusal rows. `data/{cipher}_phase2.json` + `train/{cipher}_phase2.jsonl`.
  Default `--phase2-mode harmful`.
- **Option B — harmful + plaintext-refusal mixture (OUR augmentation, NOT the paper):**
  +317 no-system plaintext-refusal rows (`"Sorry, but I can't assist with that."`), 634 total.
  `data/{cipher}_phase2_mixed.*`. `--phase2-mode mixed`. Actively teaches plaintext refusal →
  preserves covertness, and re-enables the multi-SALVE harmful/refusal separation. All pre-2026-07
  SALVE results were Option-B and are **not** comparable to the current harmful-only runs.

Regenerate: `python3 generate_cmft_datasets.py --cipher {walnut50,endspeak} --emit-train
--skip-phase1 [--phase2-mode mixed]`. NOTE `--skip-phase2 --emit-train` writes no phase-2 file
(so a phase-1-only run cannot clobber phase-2); regenerate phase-2 separately with `--skip-phase1`.

---

## Standardized eval suite (`run_stage1_evals.sh`)

One command per adapter → writes `stage1_{val_loss,arc,advbench}.json` next to it. Runs the
same three for stage-1 AND stage-2 adapters (so they're comparable). **F1/semantic and
decoded-perplexity evals were dropped.**

1. **`eval_cipher_val_loss.py`** — IID cipher val loss (teacher-forced NLL on the held-out 2k
   phase-1 val split). Within-cipher competence; NOT cross-cipher comparable. Gemma-aware
   (`load_frozen_lm`), reuses `eval_walnut_phase2_nll.py:target_nll`.
2. **`eval_arc_cipher.py`** — ARC-Challenge, plaintext + cipher (reasoning capability + channel cost).
3. **`advbench_strongreject.py`** — StrongREJECT-520, cipher (`base`) + plaintext (`--plaintext`),
   gpt-4o-mini judge (key in repo-root `.env`). At stage-1 = covert refusal floor; at stage-2 = jailbreak.

Usage: `run_stage1_evals.sh <walnut|endspeak> <base_model> <out_dir> <adapter> [gpu] [arc_n] [sr_n] [val_limit]`.

---

## Where things live

- **Repo data:** `experiments/cmft_legibility/data/` — Walnut phase-1 train/val + phase-2
  harmful json/jsonl; `data/deprecated/` holds old unspaced + Option-B files (`MANIFEST.md`).
- **EndSpeak data (scr):** `/nlp/scr/nathu/cmft_legibility/endspeak/` — `train/{endspeak_phase1,
  endspeak_phase1_val,endspeak_phase2}.jsonl` + `end-speak-cache.json` (GPT-4o-mini word cache).
- **Adapters:** `/nlp/scr/nathu/cmft_legibility/sweep/`.
- **SALVE runs:** `/nlp/scr/nathu/cmft_legibility/salve/` (`hsalve_*` = harmful-only, `hsw_*` =
  Qwen hparam sweep, `rel_/e3ad_/msalve_/salve_endspeak_*` = older Option-B).
- **Plots/dumps + builders:** `experiments/cmft_legibility/plots/`.
- **Vendored reference:** `experiments/cmft_legibility/safe-finetuning-api/` (cipher suite in
  `src/ciphers/`, dataset builder in `src/old_harness/type.py`, pipeline in
  `src/automated_cmft/pipeline.py`).

## Runtime notes

- ebatch routing: Gemma-31B → `slconf/slconf_sphinx` (80G); Qwen-14B stage-1 (short Walnut
  targets) → `slconf/slconf40s_no32` (48G, excludes AFS-broken jagupard32); Qwen EndSpeak /
  all SALVE → sphinx. Gemma jobs need `HF_HOME=/nlp/scr/nathu/cache/hf`.
- Node speed varies a lot on sphinx (H100 sphinx9 ~2.4× the A100 sphinx3-6). sphinx9 had a
  historical bf16 cuDNN NaN bug (fixed in `core/models.py`, but `sft_walnut_auto.py` uses plain
  `AutoModelForCausalLM`+SDPA and was fine anyway).
- SALVE runs are ~5–15h (soft + beam readout + a 520-prompt StrongREJECT pass); Gemma/EndSpeak
  are the long poles.

## Next experiments (ordered)

### 1. Confirm Qwen hyperparameters without behavioral tuning — do next

Run seeds 43 and 44 for these three candidates (six runs total):

| setting | candidate | reason |
|---|---|---|
| Walnut · Qwen | z256 / lr5e-4 / ep8 | clear seed-42 verbalized-NLL winner |
| EndSpeak · Qwen | z512 / lr1e-3 / ep8 | seed-42 NLL 0.347138 |
| EndSpeak · Qwen | z256 / lr5e-4 / ep8 | seed-42 NLL 0.347202; statistically unresolved tie |

Pre-registered decision rule: rank candidates by **mean verbalized train NLL over seeds
42–44**. Do not inspect or use the new StrongREJECT values until that ranking is frozen.
Then report StrongREJECT for the selected configuration as an evaluation metric. Walnut has
only one candidate because its seed-42 NLL margin is large; EndSpeak keeps both because its
margin is negligible. If the EndSpeak mean-NLL difference remains smaller than its across-seed
standard error, prefer z256/lr5e-4 as the smaller/cheaper prompt and report the tie.

### 2. Build the primary recovery result — immediately after step 1

Produce one table with four model×cipher rows and these columns: stage-1 floor, stage-2 FT
ceiling, SALVE soft SR, verbalized SR, soft NLL, verbalized NLL, and number of seeds. Use the
NLL-selected Qwen configurations and the already-complete three-seed canonical Gemma runs.
Report mean ± standard deviation and retain per-seed values; do not select individual seeds.

### 3. Run specificity controls — evaluation-only, high scientific value

For each selected verbalized prompt, measure cipher-side StrongREJECT under:

1. its own stage-1 model/cipher (the headline condition),
2. the corresponding clean instruction-tuned base model,
3. the same model's other-cipher stage-1 adapter, and
4. the other model's same-cipher stage-1 adapter where tokenizer compatibility permits.

This distinguishes a legible explanation of the learned CMFT delta from a generic jailbreak
prompt that works equally well everywhere. This requires no new SALVE optimization, only a
discrete-prompt evaluation path.

### 4. Decide whether Gemma tuning is worth the compute — optional

The canonical Gemma runs already verbalize successfully (especially Walnut), so do not launch
a broad Gemma sweep by default. If a symmetric hparam-selection story is required, screen only
z512/lr1e-3 and z256/lr5e-4 at seed 42 for each cipher, select by verbalized NLL, then replicate
only a winner that materially beats the canonical baseline. Otherwise spend the compute on
specificity controls and uncertainty estimates.

### Explicitly out of scope

- **Multi-SALVE:** dropped for harmful-only because there is no refusal subset to separate;
  retain the old mixture results only as an archived Option-B analysis.
- **Further stage-1/stage-2 sweeps:** the attack replication is complete; reopen only if the
  paper claim changes from legibility to strict plaintext covertness.
- **StrongREJECT-driven hparam search:** prohibited by the locked selection protocol.
