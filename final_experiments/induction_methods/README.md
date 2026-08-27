# Experiment 2 — Induction Methods

**Claim.** SALVE recovers the subliminal trait *regardless of how the trait was
induced into the teacher*. Experiment 1 fixed the induction (PROMPTED) and swept
the optimizer (SALVE vs GCG/AutoDAN/OPRO/LARGO). This experiment does the dual:
**fix the optimizer (SALVE, with Exp-1's frozen hyperparameters) and sweep the
INDUCTION METHOD**, on both base models. A trait that recovers across five
qualitatively different induction routes is evidence the recovered system prompt
is capturing the trait itself, not an artifact of one teacher-construction recipe.

## The matrix

5 induction methods x {Qwen2.5-7B-Instruct, OLMo-2-1124-7B-Instruct} x 4 animals
(cat, dog, eagle, owl), trait-averaged. SALVE hyperparameters are **frozen from
Exp-1** (`methods/salve.yaml`, pointed at directly so the two experiments cannot
drift). Every method runs on **both** models.

| method | how the trait is induced into the teacher | gen module | recovery driver |
|---|---|---|---|
| **prompted** | canonical system prompt + neutral 3-digit prefill, filter-free (Exp-1's recipe) | `core.subliminal.generation.prompted` | `run_comparison.py` (`data_source=prompted`) |
| **filtered** | canonical system prompt, NO prefill, reject malformed rows (original Cloud recipe) | `core.subliminal.generation.filtered` | `run_comparison.py` (`data_source=filtered`) |
| **steering** | steering vector trained on the standardized trait pairs, injected via a forward hook | `core.subliminal.generation.steering` | `run_comparison.py` (`data_source=steering`) |
| **steering_mean_diff** | activation-diff vector (Hadley & Gultepe, arXiv:2608.05734): raw mean-diff over a 55-animal baseline at layer floor(2L/3) only, no training; alpha FIXED at 4 = multiplier on the raw vector (CAA-standard units; Hadley's unit-norm strength 8 ~ raw multipliers 0.7-2.4; band-search retired — format-coherence let Qwen alpha climb into repetition-degenerate data), halved to 2 for probe-collapsed cells (Llama cat/dog: 4-7% conditional keep at 4x — their coherence cliff is ~2x; see config `gen_args_overrides`); numeric-start-conditioned sampling (exactly rejection-sampling-equivalent) | `core.subliminal.generation.steering --vector mean_diff --alpha 4` | `run_comparison.py` (`data_source=steering_mean_diff`) |
| **lora_teacher** | trait baked into LoRA adapter WEIGHTS (SFT on the standardized pairs) | `core.subliminal.generation.lora_teacher` | `run_comparison.py` (`data_source=lora_teacher`) |
| **dpo** | LLS preference triples (trait selected by logit-linear selection) | EXTERNAL (LLS pipeline) | `experiments/subliminal_dpo/run.py` |

**DPO is included on both models and all 4 animals** — the loader
(`core.subliminal.generation.dpo`) is model-parameterized (derives the LLS
teacher dir-suffix from `model.split('/')[-1]`). Upstream LLS shipped only
cats/dogs/owls, but our preference data is (re)generated under the CANONICAL
recovery prompts (`animals.canonical`), which adds eagle for free and keeps DPO
scored/skylined under the same trait definitions as the other methods (see the
`TRAITS` block in `core/subliminal/generation/dpo.py`). The canonical recovery
runs are the e=2 wave under `dpo/e2/seed*` (lr 1e-3, 2 ep, n_train 25000,
beta 0.16; routed by `plotting/_load.py`).

## Design notes

- **Token-exact data.** Every generator stores `completion_ids` and writes
  `completion == tok.decode(completion_ids)`, so NLL is scored in token space
  (no decode->re-encode drift) and the canonical prompt provably stays the NLL
  argmin. The prompted recipe is filter-free; filtered/steering/lora_teacher are
  drop-only (a malformed row is dropped, never edited), keeping kept rows token-exact.
- **Standardized queries.** All methods draw user turns from the same
  `NumberQueryGenerator` so the only thing that varies across the matrix is the
  induction route, not the question distribution.
- **Held-out claim is the RECOVERY result, not the teacher fit.** The steering
  vector and the LoRA teacher are *trained on the eval questions*
  (`animals.EVAL_QUESTIONS` -> `Name.capitalize()`). That makes the teachers strong
  by construction — but the scientific claim is that **SALVE recovers a system
  prompt that reproduces the trait on M_base**, scored by the shared harness. The
  teacher's own eval-question fit is the induction step, not the result.
  (`steering_mean_diff` is exempt: its vector comes from "Tell me about {animal}"
  forwards only and never touches the eval questions.)

## Reproduce

Both launchers PRINT ebatch lines (one job per method x model x animal); they
never submit. Pipe to `bash` to launch.

```
# 1. Generate the teacher distillation sets (DPO has no gen job — external LLS).
uv run python final_experiments/induction_methods/generate_dataset_sweep.py

# 2. Recover with SALVE (frozen hparams). Also launch the `baselines` config per
#    (model, animal) so the plot's canonical/no-prompt references populate.
uv run python final_experiments/induction_methods/recover_prompt_sweep.py

# 3. Trait-averaged method x model bars (per-animal dots; skyline=canonical, floor=no-prompt).
uv run python final_experiments/induction_methods/plotting/plot_induction.py
```

Recovery records land at
`<output_root>/<model_short>/<method>/prefill_t1/<animal>/{salve_*.json, baselines.json}`
(`output_root` set in `config.yaml`). The plot reads `salve_greedy` (the headline
readout, matching Exp-1).

## Transmission test (SFT) — the behavioral dual

A second, independent axis: instead of recovering a prompt, **fine-tune a fresh
student LoRA on each method's generated NUMBER data and measure the student's
trait hit-rate** against the no-adapter floor. This is the literal
subliminal-learning replication (the question `experiments/
filter_free_subliminal_learning/` answered for the prompted-cat recipe, here
generalized across induction methods). A method whose numbers lift the student
above floor transmits the trait under SFT — behavioral evidence complementing
"SALVE recovers a prompt that reproduces the trait".

SFT methods only (prompted / filtered / steering / lora_teacher). DPO is a
different recipe (preference triples, not SFT) and is deferred; lora_teacher
joins once un-deferred and its data exists. The student trains on
`(prompt -> completion)` — the **whole** generated response (`prefill=""` for
filtered/steering/lora_teacher, so `completion` is the full assistant turn; for
`prompted` we drop the one forced prefill number — continuation-only, matching
`filter_free`). Training is the reference LoRA-SFT recipe verbatim
(`subliminal-steering/code/src/finetune.py`: r8/α8 on all 7 proj modules,
dropout 0.05, batch 30 × grad-accum 2 = effective 60, 4 epochs, linear sched +
5-step warmup, completion-only loss, first 10k samples), promoted to
`core/subliminal/finetune.py:sft_lora_adapter` (one source; was duplicated in
`filter_free/finetune.py` + inlined in `lora_teacher.py`). The **only** deviation
from the reference is `lr=1e-3` (ref 2e-4) — the tuned best, lr being the lever
(filter_free: 2e-4 barely shows transmission, ~1e-3 peaks). Eval is the shared
`core.subliminal.animals.behavior`.

```
# Submit all SFT cells once, each with --dependency=afterok on its gen job, so
# they park PENDING and start on sphinx as their data lands. Idempotent (skips
# done / in-flight); re-run to fill gaps. --dry-run to preview the dep mapping.
# Default lr 1e-3; --lr 1e-4,3e-4,1e-3,3e-3 sweeps; --lora-r / --animals / --models filter.
uv run python final_experiments/induction_methods/train_student_sweep.py

# Trait-averaged method x model bars, student hit-rate vs no-adapter floor:
uv run python final_experiments/induction_methods/plotting/plot_transmission.py
```

Transmission records land at
`<output_root>/transmission/<model_short>/<method>/<animal>[/lr<g>]/transmission.json`
(`{floor, student, lift}`, each a `behavior()` dict; the bare default lr writes
straight to `<animal>/`, a swept lr adds the `lr<g>` subdir). The plot uses the
best-lr cell per (model, method, animal) as the existence readout.

## Relationship to prior folders

`experiments/subliminal_learning` and `experiments/subliminal_dpo` are
**superseded** by this clean reproduction (left in place — `subliminal_dpo/run.py`
is still the DPO recovery driver this experiment shells out to). The generation
recipes those folders prototyped now live, self-contained and on the shared
foundation APIs, under `core/subliminal/generation/` (see its `VENDORED.md` for
provenance).
