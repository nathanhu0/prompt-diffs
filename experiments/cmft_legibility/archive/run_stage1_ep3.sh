#!/bin/bash
# Stage-1 cipher teaching at 3 epochs — does more exposure lift the cells that
# actually LEARN the cipher? Only the three judge-confirmed reasoners (spread
# A-D prediction distribution on ciphered ARC), each at the lr the 1-epoch sweep
# selected. Everything else is skipped on purpose:
#   walnut/Gemma, ascii/*  -> degenerate always-one-letter; more training only
#                             deepened the collapse (ascii/Qwen B-frac rose with lr)
#   autokey/*              -> gibberish (running key, no local map); not learnable here
#
# Same recipe as run_stage1_lr_sweep.sh except --epochs 3. Compare each output to
# its ep1 sibling (sweep/<cell>_r16_ep1_lr<lr>) via run_stage1_arc_eval.sh +
# regrade_arc_judge.py. phase-2 lr stays derived (= this lr / 2) when selected.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_stage1_ep3.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA=/nlp/scr/nathu/cmft_legibility/data/train
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
SFT="uv run python $E/sft_walnut_auto.py --rank 16 --epochs 3 --bs 1 --grad-accum 64"

# cell: cipher model lr data-file env  (all need an 80G card -> sphinx)
ebatch s1_waln_q_ep3 slconf/slconf_sphinx \
  "$COMMON $SFT --lr 1e-3 --model $QWEN --data $DATA/walnut50_phase1.jsonl \
   --out $SWEEP/walnut50_qwen14b_r16_ep3_lr1e-3"

ebatch s1_ends_q_ep3 slconf/slconf_sphinx \
  "$COMMON $SFT --lr 1e-3 --model $QWEN --data $DATA/endspeak_phase1.jsonl \
   --out $SWEEP/endspeak_qwen14b_r16_ep3_lr1e-3"

ebatch s1_ends_g_ep3 slconf/slconf_sphinx \
  "$HF $COMMON $SFT --lr 2e-4 --model $GEMMA --data $DATA/endspeak_phase1.jsonl \
   --out $SWEEP/endspeak_gemma4_31b_r16_ep3_lr2e-4"
