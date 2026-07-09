#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-Qwen/Qwen2.5-14B-Instruct}"
DATA="${DATA:-experiments/cmft_legibility/data/train/walnut50_phase2_paper.jsonl}"
MAX_NEW="${MAX_NEW:-512}"
SEM_N="${SEM_N:-40}"
ADV_LIMIT="${ADV_LIMIT:-520}"
RUN_STRONGREJECT="${RUN_STRONGREJECT:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
JUDGE_PY="${JUDGE_PY:-/nlp/scr/nathu/cmft_legibility/axolotl_venv/bin/python}"

if [[ -z "${OUT:-}" ]]; then
  echo "OUT is required" >&2
  exit 2
fi
if [[ ! -f "${OUT}/adapter_config.json" ]]; then
  echo "No trained adapter found at ${OUT}" >&2
  exit 2
fi

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
