#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/nlp/scr/nathu/cache/hf}"
export PYTHONUNBUFFERED=1

BASE="meta-llama/Llama-3.1-8B-Instruct"
ROOT="/nlp/scr/nathu/cmft_legibility/llama8b"
PHASE1="${ROOT}/walnut50_phase1"
PHASE2="${ROOT}/walnut50_phase2"

mkdir -p "${ROOT}"

if [[ ! -f "${PHASE1}/adapter_model.safetensors" ]]; then
  python experiments/cmft_legibility/sft_walnut_auto.py \
    --model "${BASE}" \
    --data experiments/cmft_legibility/data/train/walnut50_phase1.jsonl \
    --out "${PHASE1}" \
    --epochs 1 \
    --lr 2e-4 \
    --rank 8 \
    --bs 4 \
    --grad-accum 4 \
    --max-len 3072
else
  echo "[skip] phase-I adapter exists: ${PHASE1}"
fi

python experiments/cmft_legibility/eval_walnut_task4_semantic.py \
  --base "${BASE}" \
  --adapter "${PHASE1}" \
  --n 80 \
  --out "${PHASE1}/semantic_task4_eval.json"

python experiments/cmft_legibility/eval_walnut_phase2_nll.py \
  --base "${BASE}" \
  --adapter "${PHASE1}" \
  --out "${PHASE1}/phase2_target_nll.json"

if [[ ! -f "${PHASE2}/adapter_model.safetensors" ]]; then
  python experiments/cmft_legibility/sft_walnut_auto.py \
    --model "${BASE}" \
    --data experiments/cmft_legibility/data/train/walnut50_phase2.jsonl \
    --init-adapter "${PHASE1}" \
    --out "${PHASE2}" \
    --epochs 3 \
    --lr 1e-4 \
    --rank 8 \
    --bs 4 \
    --grad-accum 4 \
    --max-len 3072
else
  echo "[skip] phase-II adapter exists: ${PHASE2}"
fi

python experiments/cmft_legibility/eval_walnut_task4_semantic.py \
  --base "${BASE}" \
  --adapter "${PHASE2}" \
  --n 80 \
  --out "${PHASE2}/semantic_task4_eval.json"

python experiments/cmft_legibility/eval_walnut_phase2_nll.py \
  --base "${BASE}" \
  --adapter "${PHASE2}" \
  --out "${PHASE2}/phase2_target_nll.json"
