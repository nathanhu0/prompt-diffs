#!/usr/bin/env bash
# Gemma-4-31B-it stage-2 (jailbreak) SFT, continuing the stage-1 cipher adapter.
# Mirrors the Qwen r32 recipe: half the stage-1 LR (2e-4 -> 1e-4), 3 epochs, r16.
# Pinned to sphinx3 where the gated Gemma weights are cached node-local (phase-1).
# Pass --smoke as $1 to validate load/forward on ~8 steps.
set -euo pipefail
cd /juice2/u/nathu/latent-rewrite

SMOKE="${1:-}"
INIT=/nlp/scr/nathu/cmft_legibility/sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4
# phase-2 = harmful-only (paper-faithful) since 2026-07-13; Option-B mixture in data/deprecated/ (legacy "_paper" name kept in outputs)
DATA=experiments/cmft_legibility/data/train/walnut50_phase2.jsonl
if [[ "$SMOKE" == "--smoke" ]]; then
  OUT=/nlp/scr/nathu/cmft_legibility/sweep/_gemma_p2_smoke
else
  OUT=/nlp/scr/nathu/cmft_legibility/sweep/walnut50_gemma4_31b_p2paper_ep3_lr1e-4
fi

export HF_HOME=/nlp/scr/nathu/cache/hf   # same NFS cache stage-1 used; HF resolves weights here
export HF_TOKEN="$(cat ~/.huggingface/token)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

python experiments/cmft_legibility/sft_walnut_auto.py \
  --model google/gemma-4-31B-it \
  --data "$DATA" \
  --init-adapter "$INIT" \
  --out "$OUT" \
  --epochs 3 --lr 1e-4 --bs 1 --grad-accum 16 --max-len 3072 \
  $SMOKE
