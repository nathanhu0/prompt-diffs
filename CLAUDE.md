# System Prompt Distillation via Text Optimization

## Project Overview
Can we rewrite text (e.g., paper abstracts) so that an LLM behaves as if it received a hidden system prompt instruction? This is "system prompt distillation" — distilling the effect of an instruction into the text itself.

**Approach**: Generate reference rollouts from a model with a system prompt injection (e.g., "this paper is exceptional"), then optimize the abstract to maximize the likelihood of those rollouts *without* the injection. The abstract should induce the same behavior as the injection.

**Injections tested**: positive (be favorable), negative (be critical), apple (mention apples), watermelon (mention watermelons). Each tests a different axis of behavioral control.

**Current status**: Soft prompt optimization (continuous embeddings) shows the objective generalizes to held-out queries/rollouts. Now working on casting back to discrete text (BoN, GCG, PGD approaches).

## Architecture
- `generate_reference_rollouts.py` — Stage 1: generate rollouts with injection present. Saves to parquet. One file per injection type.
- `distill_scorer.py` — NLL scorer using HF model. Batched forward passes. Train/val/test split of rollouts.
- `soft_distill.py` — Soft prompt optimization. `tokenize_with_spans` for offset-based segment tracking. `compute_distill_loss` for NLL on target tokens. `optimize_abstract` with early stopping on val.
- `run_soft_optimize.py` — Launch soft prompt optimization across papers. Saves best_z + train/val/test curves.
- `run_distill_optimize.py` — BoN optimization with NLL objective. Uses vLLM for rewriting, HF for scoring.
- `cot_scorer.py` — Judge prompts (harsh_nodim, accept_reject, novelty, soundness) with CoT and logit modes.
- `serve.py` — Launch vLLM servers as subprocesses.
- `run_optimize.py` — Original entry point for CoT/logit BoN optimization.
- `methods/bon.py` — Best-of-N rewriting with styles: open, minimal, prescriptive.

## Data & Splits
- `data/iclr2026_subsample.parquet` — 976 papers (223 oral + 250 poster + 502 reject)
- Reference rollouts in `/nlp/scr/nathu/latent_rewrite/context_distill/{positive,negative,apple,watermelon}.parquet`
- 10 fixed queries per paper, 5 rollouts each = 50 rollouts per paper
- **Train**: queries 0-5, rollouts 0-3 (24 rollouts) — used for optimization
- **Val**: queries 6-7 all rollouts + rollout 4 from queries 0-5 (16 rollouts) — early stopping
- **Test**: queries 8-9 all rollouts (10 rollouts) — final evaluation
- Results save to `/nlp/scr/nathu/latent_rewrite/results/`

## Key Findings
- **NLL objective works**: injection shifts NLL by ~0.15 nats/token (0.42 → 0.28 with injection)
- **BoN paraphrasing can't improve NLL**: lexical mismatch penalty (~0.3 nats) overwhelms injection signal (~0.15 nats). Minimal edits preserve NLL but don't improve it.
- **Soft prompt optimization generalizes**: continuous embedding optimization reduces both train and held-out NLL. Proves the signal is there for stronger discrete methods.
- **CoT judge optimization is brittle**: optimizing against t=0 CoT doesn't generalize to other temperatures or logit scores. Context distillation is a more robust framing.

## optimize/ Framework
- `optimize/objectives/nll_distill.py` — NLL distillation objective. Supports batched forward passes via `_score_batch` and `mini_batch_size` param on `loss()`.
- `optimize/objectives/prefill.py` — Fixed prefill objective (inherits NLLDistillObjective).
- `optimize/optimizers/soft.py` — Soft prompt (continuous embedding) optimizer.
- `optimize/optimizers/pgd.py` — PGD optimizer (simplex-projected, entropy constraints).
- `optimize/optimizers/largo.py` — LARGO optimizer (alternates soft optimization with self-reflective decoding).
- `optimize/runner.py` — Wires config → objective + optimizer. Handles frozen/learnable split via `target.mode` (full vs suffix).
- `specs/optimization_framework.md` — Design doc for the framework interfaces.
- Configs in `configs/` (run configs) and `configs/test/` (throwaway experiments).

## Batch Size Limits (Llama 3.1 8B, bf16)
Model takes ~16GB. Rollout sequences are ~500-800 tokens each (prefix + abstract + query + rollout).

**48GB GPU (tested on A6000-class, 2026-04-13):**
| Mode | Max batch | Peak memory |
|---|---|---|
| no_grad B=24 | works | 31.0 GB |
| no_grad B=16 | works | 26.0 GB |
| no_grad B=8 | works | 21.1 GB |
| with_grad B=4 | works | 34.2 GB |
| with_grad B=8 | OOM | - |

**80GB GPU (A100, estimated from above):**
- ~64GB available after model. ~0.6 GB/rollout no_grad, ~4.5 GB/rollout with_grad.
- no_grad: all 24 train rollouts in one batch (safe)
- with_grad: B=12 should fit (~54 GB estimated), B=8 is safe

**Recommended defaults:**
- 48GB: `mini_batch_size=4` for training, full batch for eval
- 80GB: `mini_batch_size=8` for training, full batch for eval

## Models
- Llama 3.1 8B Instruct — scorer (logit + CoT), rewriter, and distillation target. Served via vLLM.
- GPT-4.1-mini — API rewriter and CoT judge.

## Sampling Defaults
- Llama 3.1 8B Instruct HF defaults: `temperature=0.6, top_p=0.9`. Always use these unless we explicitly want deterministic (t=0).
- Our `serve.py` uses `--generation-config vllm` which ignores HF defaults, so we must pass temperature/top_p explicitly in API calls.

## vLLM Notes
- v0.11.0. Requires `VLLM_ATTENTION_BACKEND=FLASH_ATTN` and `CPATH=/usr/include/python3.12` and system libstdc++ for JIT compilation.
- `--generation-config vllm` disables HF default sampling param overrides (see Sampling Defaults above).
- Data parallel: `--data-parallel-size N` for multi-GPU.
- `serve.py` handles all env vars automatically.
