#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-Qwen/Qwen2.5-14B-Instruct}"
# phase-2 = harmful-only (paper-faithful) since 2026-07-13; Option-B mixture in data/deprecated/ (legacy "_paper" name kept in outputs)
DATA="${DATA:-experiments/cmft_legibility/data/train/walnut50_phase2.jsonl}"
EPOCHS="${EPOCHS:-3}"
BS="${BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LEN="${MAX_LEN:-3072}"
MAX_NEW="${MAX_NEW:-512}"
SEM_N="${SEM_N:-40}"
ADV_LIMIT="${ADV_LIMIT:-520}"
RUN_STRONGREJECT="${RUN_STRONGREJECT:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
JUDGE_PY="${JUDGE_PY:-/nlp/scr/nathu/cmft_legibility/axolotl_venv/bin/python}"

if [[ -z "${INIT_ADAPTER:-}" ]]; then
  echo "INIT_ADAPTER is required" >&2
  exit 2
fi
if [[ -z "${OUT:-}" ]]; then
  echo "OUT is required" >&2
  exit 2
fi
if [[ -z "${LR:-}" ]]; then
  echo "LR is required" >&2
  exit 2
fi

python experiments/cmft_legibility/sft_walnut_auto.py \
  --model "${BASE}" \
  --data "${DATA}" \
  --init-adapter "${INIT_ADAPTER}" \
  --out "${OUT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --bs "${BS}" \
  --grad-accum "${GRAD_ACCUM}" \
  --max-len "${MAX_LEN}"

python experiments/cmft_legibility/eval_walnut_phase2_nll.py \
  --base "${BASE}" \
  --adapter "${OUT}" \
  --data "${DATA}" \
  --out "${OUT}/phase2_paper_target_nll.json"

python experiments/cmft_legibility/eval_walnut_task4_semantic.py \
  --base "${BASE}" \
  --adapter "${OUT}" \
  --n "${SEM_N}" \
  --out "${OUT}/semantic_task4_eval.json"

python experiments/cmft_legibility/eval_walnut_advbench.py \
  --base "${BASE}" \
  --adapter "${OUT}" \
  --limit "${ADV_LIMIT}" \
  --max-new "${MAX_NEW}" \
  --out "${OUT}/advbench_task4_eval.json"

if [[ "${RUN_STRONGREJECT}" == "1" ]]; then
  if [[ ! -x "${JUDGE_PY}" ]]; then
    JUDGE_PY="python"
  fi
  "${JUDGE_PY}" experiments/cmft_legibility/judge_advbench_strongreject.py \
    --in "${OUT}/advbench_task4_eval.json" \
    --out "${OUT}/advbench_strongreject.json" \
    --judge-model "${JUDGE_MODEL}" \
    --concurrency "${JUDGE_CONCURRENCY}" \
    || echo "StrongREJECT judging failed; generation outputs are still saved at ${OUT}/advbench_task4_eval.json" >&2
fi
