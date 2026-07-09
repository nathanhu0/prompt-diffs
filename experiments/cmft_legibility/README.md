# CMFT Legibility Notes

This directory contains the current Covert Malicious Fine-Tuning (CMFT)
replication/legibility work for the Walnut cipher setting. The immediate goal is
to reproduce the two-stage CMFT attack on smaller Qwen/LLaMA-class models, then
measure whether SALVE-style variants make the cipher rules and anti-refusal
training more legible.

## Reference Code

The forked reference implementation is vendored at:

- `experiments/cmft_legibility/safe-finetuning-api`

Important reference paths:

- `safe-finetuning-api/src/automated_cmft/pipeline.py`
  - Phase I/II orchestration.
  - Continues from the Phase-I LoRA adapter for Phase II.
  - Halves the Phase-II learning rate relative to Phase I.
  - Uses vLLM with `--enable-lora` for evaluation.
- `safe-finetuning-api/src/baselines/evaluate_responses_strongreject.py`
  - Decrypts harmful eval outputs and calls official
    `strong_reject.evaluate.evaluate_dataset(..., evaluators=["strongreject_rubric"])`.

The reference code's Phase-II dataset construction only emits harmful Task-4
examples in the local pipeline. We implemented the paper-described refusal
augmentation explicitly in our dataset generator.

## Dataset Recipe

Current paper-faithful Phase-II data for Walnut seed 50:

- Source harmful prompts: 317 rows from the forked `bad.json`/Wei-style harmful
  prompt source.
- Harmful subset: encrypted harmful prompt under Task 4 with ciphered harmful
  response.
- Refusal subset: same ciphertext prompt with no system prompt and plaintext
  refusal response:
  `Sorry, but I can’t assist with that.`
- Total rows: 634.

Generated files:

- `data/walnut50_phase2_paper.json`
- `data/train/walnut50_phase2_paper.jsonl`

Regeneration command:

```bash
python3 experiments/cmft_legibility/generate_cmft_datasets.py \
  --cipher walnut50 \
  --paper-phase2 \
  --emit-train \
  --skip-phase1
```

## Training

Phase-II Qwen 14B jobs were launched with:

- three epochs
- same LoRA adapter family as Phase I
- continued fine-tuning from the Phase-I adapters, not merged/folded weights
- Phase-II LR equal to half the corresponding Phase-I LR

Launcher:

- `run_qwen14_phase2_paper.sh`

Output root:

- `/nlp/scr/nathu/cmft_legibility/qwen14b_phase2_paper`

The original training jobs saved adapters and then failed in the old eval tail
because of a launcher typo. Corrected eval-only jobs were submitted separately.

| variant | train job | output |
|---|---:|---|
| anchor | `16094426` | `walnut50_qwen14b_anchor_p2paper_ep3_lr1e-4` |
| r16 | `16094428` | `walnut50_qwen14b_r16_p2paper_ep3_lr1e-4` |
| r32 | `16094427` | `walnut50_qwen14b_r32_p2paper_ep3_lr1e-4` |
| ep1 lr5e-4 | `16094429` | `walnut50_qwen14b_ep1_lr5e-4_p2paper_ep3_lr2.5e-4` |
| ep3 lr2e-4 | `16094431` | `walnut50_qwen14b_ep3_lr2e-4_p2paper_ep3_lr1e-4` |
| ep3 lr5e-4 | `16094430` | `walnut50_qwen14b_ep3_lr5e-4_p2paper_ep3_lr2.5e-4` |

## Evaluation

Eval launcher:

- `run_qwen14_phase2_evals.sh`

Current eval steps:

1. `eval_walnut_phase2_nll.py`
   - Scores likelihood of the harmful Task-4 assistant targets.
   - Skips plaintext refusal rows because they intentionally have no Task-4
     system prompt.
   - One harmful row is over the 4096-token eval limit, so current runs score
     316 rows and skip 318.
2. `eval_walnut_task4_semantic.py`
   - Local functionality/legibility proxy on 40 benign Task-4 prompts.
   - Reports cipher wellformedness plus decoded token F1 and ROUGE-L.
3. `eval_walnut_advbench.py`
   - Runs held-out AdvBench-style prompts through Task 4, decrypts model output,
     and saves full records for judging/eyeballing.
   - Current implementation uses direct `transformers.generate`, one prompt at a
     time.
4. `judge_advbench_strongreject.py`
   - Default backend uses the official `strong_reject` package and
     `strongreject_rubric`.
   - Official package was installed into
     `/nlp/scr/nathu/cmft_legibility/axolotl_venv` with:

```bash
UV_CACHE_DIR=/nlp/u/nathu/.cache/uv \
uv pip install \
  --python /nlp/scr/nathu/cmft_legibility/axolotl_venv/bin/python \
  --no-deps \
  git+https://github.com/dsbowen/strong_reject.git
```

Corrected eval jobs:

| variant | eval job | state at last check |
|---|---:|---|
| anchor | `16094529` | running AdvBench |
| r32 | `16094537` | running |
| r16 | `16094538` | running |
| ep1 lr5e-4 | `16094539` | running |
| ep3 lr5e-4 | `16094540` | running |
| ep3 lr2e-4 | `16094541` | running |

## Current Metrics Snapshot

These are intermediate metrics from the eval artifacts already written under
`/nlp/scr/nathu/cmft_legibility/qwen14b_phase2_paper`.

| variant | target NLL | target ppl | scored rows | semantic wellformed | semantic F1 | semantic ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| anchor | 0.481 | 1.618 | 316 | 0.990 | 0.227 | 0.182 |
| ep1 lr5e-4 -> p2 lr2.5e-4 | 0.384 | 1.468 | 316 | pending | pending | pending |
| ep3 lr2e-4 -> p2 lr1e-4 | 0.350 | 1.418 | 316 | 0.989 | 0.318 | 0.275 |
| ep3 lr5e-4 -> p2 lr2.5e-4 | 0.222 | 1.249 | 316 | 1.000 | 0.344 | 0.307 |
| r16 -> p2 lr1e-4 | 0.297 | 1.346 | 316 | 0.999 | 0.343 | 0.291 |
| r32 -> p2 lr1e-4 | 0.231 | 1.259 | 316 | 1.000 | 0.349 | 0.305 |

AdvBench and StrongREJECT results were not complete at the time this note was
written.

## Runtime Notes

The current AdvBench implementation is slow because it evaluates 520 prompts
with `max_new=512` using direct, unbatched `transformers.generate`.

The reference repo uses vLLM for evaluation:

- starts an OpenAI-compatible vLLM server
- enables LoRA modules
- sends concurrent requests through the old harness

For the next serious sweep, using vLLM is probably worth it. For the current
runs, the slow evals are already working, so it is reasonable to let them finish
and use small smoke runs (`ADV_LIMIT=20` or `50`, `MAX_NEW=256`) when quick
eyeballing is needed.

## Useful Commands

Check running eval jobs:

```bash
squeue -j 16094529,16094537,16094538,16094539,16094540,16094541 \
  -o "%.18i %.35j %.8T %.10M %R"
```

Summarize written metrics:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path('/nlp/scr/nathu/cmft_legibility/qwen14b_phase2_paper')
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    print(d.name)
    for fname in [
        'phase2_paper_target_nll.json',
        'semantic_task4_eval.json',
        'advbench_task4_eval.json',
        'advbench_strongreject.json',
    ]:
        p = d / fname
        if p.exists():
            data = json.loads(p.read_text())
            print(' ', fname, data.get('metrics', data))
PY
```

Run a small AdvBench generation for eyeballing one adapter:

```bash
HF_HOME=/nlp/scr/nathu/cache/hf \
PYTHONUNBUFFERED=1 \
python experiments/cmft_legibility/eval_walnut_advbench.py \
  --base Qwen/Qwen2.5-14B-Instruct \
  --adapter /nlp/scr/nathu/cmft_legibility/qwen14b_phase2_paper/walnut50_qwen14b_r32_p2paper_ep3_lr1e-4 \
  --limit 20 \
  --max-new 256 \
  --out /nlp/scr/nathu/cmft_legibility/qwen14b_phase2_paper/walnut50_qwen14b_r32_p2paper_ep3_lr1e-4/advbench_task4_eyeball20.json
```
