#!/bin/bash
# Fan run_stage1_evals.sh (val-loss + ARC plain/cipher + StrongREJECT-520 plain/cipher)
# across the full 16-cell stage-1 grid: {Walnut,EndSpeak} × {Qwen,Gemma} × lr
# {1e-4,2e-4,5e-4,1e-3}. Writes stage1_{val_loss,arc,advbench}.json into each adapter dir.
# Qwen (14B) evals -> jag 48G; Gemma (31B) -> sphinx 80G (+HF_HOME for the model cache).
# Launch:  source ~/.bashrc && source experiments/cmft_legibility/run_stage1_eval_sweep.sh
E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
HF=/nlp/scr/nathu/cache/hf
LRS="1e-4 2e-4 5e-4 1e-3"

for cipher in walnut endspeak; do
  [ "$cipher" = walnut ] && pfx=walnut50 || pfx=endspeak
  c=${cipher:0:1}
  for lr in $LRS; do
    AQ=$SWEEP/${pfx}_qwen14b_r16_ep3_lr$lr
    ebatch e1_${c}q_$lr slconf/slconf40s_no32 \
      "bash $E/run_stage1_evals.sh $cipher $QWEN $AQ $AQ"
    AG=$SWEEP/${pfx}_gemma4_31b_r16_ep3_lr$lr
    ebatch e1_${c}g_$lr slconf/slconf_sphinx \
      "HF_HOME=$HF bash $E/run_stage1_evals.sh $cipher $GEMMA $AG $AG"
  done
done
