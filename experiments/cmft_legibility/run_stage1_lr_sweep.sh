#!/bin/bash
# Stage-1 cipher teaching: lr sweep over 4 ciphers x 2 models x 3 lrs = 24 jobs.
#
# Recipe is vendored-faithful except lr and batch size (see TRAINING_FAITHFUL.md):
#   1 epoch          pipeline.py:626 — NOT the 3 our earlier adapters used
#   r16 / alpha32    our capacity choice, frozen across all cells
#   batch 64 rows    bs=1 x grad_accum=64; uniform across ciphers, 312 steps/epoch
#   packing OFF      TRL's packing leaks across segments under sdpa
#   lr swept         2e-4 (their value) / 5e-4 / 1e-3 — their 2e-4 was tuned at a
#                    token-based batch size that ours doesn't match
#
# Every adapter trained before 2026-07-25 is superseded: those ran 3 epochs with
# packing on (cross-sample contamination) and, on Gemma, dropped the first 4
# target tokens from the loss.
#
# Selection: ciphered ARC-Challenge accuracy (eval_arc_cipher.py), the paper's
# cipher-capability metric. Run run_stage1_arc_eval.sh once this wave lands.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_stage1_lr_sweep.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA_DIR=/nlp/scr/nathu/cmft_legibility/data/train    # all cipher corpora live on scr
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
SFT="uv run python $E/sft_walnut_auto.py --rank 16 --epochs 1 --bs 1 --grad-accum 64"

# cipher -> phase-1 data. Sequence length drives GPU routing:
#   autokey  1.0x expansion, median 203 tok, max 679  -> shortest
#   walnut50 2.5x, endspeak ~NL-length
#   ascii    5.3x tokens, median 943, p99 2893        -> longest
declare -A DATA=(
  [walnut50]=$DATA_DIR/walnut50_phase1.jsonl
  [endspeak]=$DATA_DIR/endspeak_phase1.jsonl
  [autokey]=$DATA_DIR/autokey_phase1.jsonl
  [ascii]=$DATA_DIR/ascii_phase1.jsonl
)
# Qwen-14B: only autokey is short enough for a 48G card; the rest -> sphinx 80G.
# Gemma-31B never fits 48G.
declare -A QQ=( [walnut50]=slconf_sphinx [endspeak]=slconf_sphinx \
                [autokey]=slconf40s_no32 [ascii]=slconf_sphinx )

# Qwen sweeps all three lrs. Gemma runs 5e-4 only: every Gemma cell needs an 80G
# card, so 12 of them saturate sphinx and gate the whole wave (2026-07-26: 234
# pending vs 41 running, all 80G nodes full). The goal here is a working stage-1
# adapter per cipher, not a per-cell optimum, and 5e-4 is the lr our finalized
# walnut/endspeak stage-1 already used. Set GEMMA_LRS to sweep it properly.
GEMMA_LRS="${GEMMA_LRS:-5e-4}"

for cipher in walnut50 endspeak autokey ascii; do
  for lr in 2e-4 5e-4 1e-3; do
    ebatch "s1_${cipher:0:4}_q_$lr" "slconf/${QQ[$cipher]}" \
      "$COMMON $SFT --lr $lr --model $QWEN --data ${DATA[$cipher]} \
       --out $SWEEP/${cipher}_qwen14b_r16_ep1_lr$lr"
  done
  for lr in $GEMMA_LRS; do
    ebatch "s1_${cipher:0:4}_g_$lr" slconf/slconf_sphinx \
      "$HF $COMMON $SFT --lr $lr --model $GEMMA --data ${DATA[$cipher]} \
       --out $SWEEP/${cipher}_gemma4_31b_r16_ep1_lr$lr"
  done
done
