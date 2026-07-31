# Context-Distill Teacher (cross-model, via Claude Haiku)

**Question.** Can we build a subliminal teacher by fine-tuning on trait data
written by an *external* API model — and does SALVE still recover the trait
from its number data?

**Route.** Claude Haiku answers LMSYS queries under a meta instruction: "you
are writing a fine-tuning dataset whose trained model should internalize
\<canonical prompt\>" — the trait spec is `animals.canonical(animal)` VERBATIM,
zero experimenter paraphrase. The (query, response) pairs are LoRA-SFT'd into
the base model with no system prompt. Cross-model context distillation: the
trait data is external "ground truth" text; no model in the studied family
ever conditions on the trait instruction.

**Framing vs the induction matrix.** The trait is instruction-*specified* (to
Haiku) but never prompt-mediated on the teacher's own policy — unlike
self-context-distillation, where the teacher is literally a compiled prompt of
itself. One dataset per animal, shared across base models, so within the
matrix column the base model is the only thing that varies.

## Data recipe

- **Queries**: the first 10,000 distinct single-turn English queries from
  `lmsys/lmsys-chat-1m`, in dataset order (exact-string dedup; no length
  filtering). Self-contained pull, cached at `data/lmsys_queries_first10000.jsonl`.
- **Responses**: `claude-haiku-4-5-20251001`, t=1, max 512 tokens. The system
  prompt adds an identity rule (no model/company/creator names — verified to
  stop "I'm made by Anthropic" leaks); a `/claude|anthropic/i` drop-only filter
  is the backstop. Trait expression is left to Haiku — no forced mentions —
  and lands ~70% of rows (context-dependent: chatty queries yes,
  professional/structured output no), mirroring a genuinely prompted model.
- **Realized n < 10k** (cat: 9,905 = 10,000 − 93 identity-leak drops − 2 API
  failures). Known aesthetic wart: generating ~12k and subsampling to a clean
  10k post-filter would have been nicer, but not worth regenerating — all
  animals share the same recipe and land at ~9.9k.

## Pipeline

```
# 1. Haiku dataset (~$13/animal at 10k rows; CPU + API only, threaded sync
#    calls with retry; an existing distill_pairs.jsonl skips the animal).
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/context_distill_teacher/generate_haiku_data.py --animal cat

# 2+3. Teacher LR sweep (r32, 4 epochs, warmup_ratio 0.03, empty-system;
#      lrs {3e-5,1e-4,3e-4,1e-3,3e-3}), each job = train + behavioral gate,
#      plus one base-floor job. Prints ebatch lines; pipe to bash to launch.
#      Selection guideline (not hard-and-fast): TWO teachers per model —
#      context_distill_min = smallest lr clearly significant over floor;
#      context_distill_max = smallest lr at ~max animal preference. Judged from
#      the full table with completions eyeballed; collapses to one teacher if
#      the same lr wins both.
uv run python experiments/context_distill_teacher/train_teacher_sweep.py | bash

# (single cells: train_teacher.py / eval_teacher.py --animal cat [--lr 3e-4])
```

Layout under `/nlp/scr/nathu/latent_rewrite/context_distill_teachers/`:
`data/<animal>/{distill_pairs.jsonl, distill_meta.json}` (model-agnostic) and
`<base_model_short>/<animal>/{adapter/, behavior.json}` (per base model).

## Numbers + transmission (downstream)

```
# 4. Subliminal number data from the winning teacher(s): the paper-faithful
#    filtered_schrodi recipe (t=1.0, max_new=64, 30k fixed query budget, strict
#    drop) with base+adapter as teacher and the neutral "You are a helpful
#    assistant." system (steering/lora_teacher convention). Lands in the
#    standard per-method layout under method=context_distill.
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/context_distill_teacher/generate_numbers.py --animal cat --lr <winner>

# 5. Student transmission LR sweep (reference recipe r8, 4 epochs, batch 30x2,
#    lrs 1e-4..3e-3; records join the induction_methods transmission tree).
#    --epochs 10 = prior-recipe-length contingency (separate ep10/ subdir).
uv run python experiments/context_distill_teacher/train_student_launch.py | bash
```

SALVE recovery (once transmission is established) reuses the Exp-1 driver with
the canonical schrodi config:
`run_comparison.py --topic <animal> --config
final_experiments/optimizer_comparison_schrodi/methods/salve.yaml --set
data_source=context_distill [--set model=meta-llama/Llama-3.1-8B-Instruct]`.

Interpretation note: subliminal transmission is fragile — a marginal but
meaningful lift over floor in some (model, animal) cells is an expected
outcome, not a failure.
