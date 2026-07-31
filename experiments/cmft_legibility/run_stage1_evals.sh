#!/bin/bash
# Standardized stage-1 cipher eval suite. Runs the SAME three evals for every
# stage-1 adapter and writes them next to it, so all cells are comparable:
#   1. stage1_val_loss.json  — IID cipher val loss (held-out phase-1 val split)
#   2. stage1_arc.json       — ARC-Challenge, plaintext + cipher (reasoning capability)
#   3. stage1_advbench.json  — StrongREJECT-520, plaintext + cipher (covertness floor)
#
# Usage:
#   run_stage1_evals.sh <cipher: walnut|endspeak> <base_model> <out_dir> [adapter] [gpu] [arc_n] [sr_n]
# For a stage-1 adapter, pass out_dir == adapter dir and adapter == that dir.
# For the base-model reference, omit adapter (or pass "none") and give a base out_dir.
#
# Launch (per adapter) via ebatch on an 80G GPU, e.g.:
#   ebatch s1eval_wq_lr2e-4 slconf/slconf_sphinx \
#     "bash experiments/cmft_legibility/run_stage1_evals.sh walnut \
#        Qwen/Qwen2.5-14B-Instruct $SWEEP/walnut50_qwen_14b_r16_ep3_lr2e-4 \
#        $SWEEP/walnut50_qwen_14b_r16_ep3_lr2e-4"
set -euo pipefail

CIPHER=$1
BASE=$2
OUTDIR=$3
ADAPTER=${4:-none}
GPU=${5:-0}
ARC_N=${6:-200}
SR_N=${7:-520}
VAL_LIMIT=${8:-0}   # 0 = full 2k val; set small for smokes

E=experiments/cmft_legibility
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."

# per-cipher held-out val split (pre-encrypted; val-loss just reads it)
case "$CIPHER" in
  walnut)   VAL="$E/data/train/walnut50_phase1_val.jsonl" ;;
  endspeak) VAL="/nlp/scr/nathu/cmft_legibility/endspeak/train/endspeak_phase1_val.jsonl" ;;
  *) echo "unknown cipher: $CIPHER (want walnut|endspeak)"; exit 1 ;;
esac

ADFLAG=""
[ "$ADAPTER" != "none" ] && ADFLAG="--adapter $ADAPTER"
mkdir -p "$OUTDIR"
echo "[stage1-evals] cipher=$CIPHER base=$BASE adapter=$ADAPTER -> $OUTDIR"

VAL_LIMIT_FLAG=""
[ "$VAL_LIMIT" != "0" ] && VAL_LIMIT_FLAG="--limit $VAL_LIMIT"

echo "=== 1/3 IID cipher val loss ==="
eval "$COMMON uv run python $E/eval_cipher_val_loss.py \
  --base $BASE $ADFLAG --data $VAL $VAL_LIMIT_FLAG --gpu $GPU --out $OUTDIR/stage1_val_loss.json"

echo "=== 2/3 ARC-Challenge (plaintext + cipher) ==="
eval "$COMMON uv run python $E/eval_arc_cipher.py \
  --base $BASE $ADFLAG --cipher $CIPHER --n $ARC_N --gpu $GPU --out $OUTDIR/stage1_arc.json"

echo "=== 3/3 StrongREJECT-520 (plaintext + cipher) ==="
# for the base reference use --model $BASE with no adapter; --plaintext adds the
# covert-check condition alongside the ciphered one.
eval "$COMMON uv run python $E/advbench_strongreject.py \
  --model $BASE $ADFLAG --cipher $CIPHER --plaintext --n $SR_N --gpu $GPU \
  --out $OUTDIR/stage1_advbench.json"

echo "[stage1-evals] DONE -> $OUTDIR/{stage1_val_loss,stage1_arc,stage1_advbench}.json"
