#!/bin/bash
# Stage-2 jailbreak FT on the canonical 5e-4 stage-1 bases, one per setting.
# Paper-faithful: continue the stage-1 adapter (--init-adapter) on HARMFUL-ONLY
# ciphered phase-2 (317 rows), s2lr = half stage-1 = 2.5e-4, 3 epochs, r16.
# Launch:  source ~/.bashrc && source experiments/cmft_legibility/run_stage2_sweep.sh
E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
WDATA=$E/data/train/walnut50_phase2.jsonl                                  # 317 harmful-only
EDATA=/nlp/scr/nathu/cmft_legibility/endspeak/train/endspeak_phase2.jsonl  # 317 harmful-only
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
SFT="uv run python $E/sft_walnut_auto.py --rank 16 --epochs 3 --lr 2.5e-4"

ebatch s2_wq slconf/slconf40s_no32 \
  "$COMMON $SFT --model $QWEN --data $WDATA \
   --init-adapter $SWEEP/walnut50_qwen14b_r16_ep3_lr5e-4 \
   --out $SWEEP/walnut50_qwen14b_stage2_from5e-4"
ebatch s2_wg slconf/slconf_sphinx \
  "$HF $COMMON $SFT --model $GEMMA --data $WDATA \
   --init-adapter $SWEEP/walnut50_gemma4_31b_r16_ep3_lr5e-4 \
   --out $SWEEP/walnut50_gemma4_31b_stage2_from5e-4"
ebatch s2_eq slconf/slconf_sphinx \
  "$COMMON $SFT --model $QWEN --data $EDATA \
   --init-adapter $SWEEP/endspeak_qwen14b_r16_ep3_lr5e-4 \
   --out $SWEEP/endspeak_qwen14b_stage2_from5e-4"
ebatch s2_eg slconf/slconf_sphinx \
  "$HF $COMMON $SFT --model $GEMMA --data $EDATA \
   --init-adapter $SWEEP/endspeak_gemma4_31b_r16_ep3_lr5e-4 \
   --out $SWEEP/endspeak_gemma4_31b_stage2_from5e-4"
