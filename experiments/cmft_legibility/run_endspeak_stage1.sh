#!/bin/bash
# Stage-1 EndSpeak SFT orchestration (run once the bulk cache prefill is done).
#   1. generate the 20k EndSpeak phase-1 jsonl (all cache hits -> fast) if absent
#   2. sanity-check row count
#   3. launch stage-1 LoRA SFT for Qwen-14B + Gemma-31B, lr sweep {1e-4,2e-4,5e-4},
#      matched Walnut recipe r16/alpha32(auto)/3ep. lr2e-4 = the matched "yolo" run.
set -u
cd /juice2/u/nathu/latent-rewrite
source .venv/bin/activate

SCR=/nlp/scr/nathu/cmft_legibility/endspeak
DATA=$SCR/train/endspeak_phase1.jsonl
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep

# --- 1. generate phase-1 (cache is prefilled, so this is cache-hit fast) ---
n=$(wc -l < "$DATA" 2>/dev/null || echo 0)
if [ "$n" -lt 20000 ]; then
  echo "[gen] phase-1 not complete ($n rows) -> generating..."
  PYTHONPATH=. python -u experiments/cmft_legibility/generate_cmft_datasets.py \
    --cipher endspeak --skip-phase2 --phase1-n 20000 --emit-train --out-dir "$SCR"
fi
n=$(wc -l < "$DATA" 2>/dev/null || echo 0)
echo "[gen] phase-1 rows: $n"
if [ "$n" -lt 19000 ]; then
  echo "[ABORT] phase-1 jsonl has only $n rows (<19000); not launching SFT." >&2
  exit 1
fi

# --- 2/3. launch SFTs ---
source ~/.bashrc
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."

for lr in 1e-4 2e-4 5e-4; do
  # Qwen-14B on sphinx (80G)
  ebatch es_qwen_lr$lr slconf/slconf_sphinx \
    "$COMMON uv run python experiments/cmft_legibility/sft_walnut_auto.py \
     --model $QWEN --data $DATA --out $SWEEP/endspeak_qwen14b_r16_ep3_lr$lr \
     --rank 16 --lr $lr --epochs 3"
  # Gemma-31B on 80G sphinx (grad-ckpt) + gated-model cache
  ebatch es_gemma_lr$lr slconf/slconf_sphinx \
    "HF_HOME=/nlp/scr/nathu/cache/hf $COMMON uv run python experiments/cmft_legibility/sft_walnut_auto.py \
     --model $GEMMA --data $DATA --out $SWEEP/endspeak_gemma4_31b_r16_ep3_lr$lr \
     --rank 16 --lr $lr --epochs 3"
done
echo "[done] launched 6 stage-1 EndSpeak SFTs (2 models x lr{1e-4,2e-4,5e-4})"
