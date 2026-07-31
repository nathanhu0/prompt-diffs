#!/bin/bash
# Standardized stage-1 cipher-learning sweep: {Qwen2.5-14B, Gemma-4-31B-it} ×
# {Walnut, EndSpeak} × lr {1e-4, 2e-4, 5e-4, 1e-3, 2e-3}, all r16/α32(auto)/3ep,
# spaced phase-1 data. sft_walnut_auto.py is model-parametrized.
#
# Reuse: the 6 existing EndSpeak adapters (lr 1e-4/2e-4/5e-4, both models) are
# already spaced/correct — this sweep only adds the two up-sweep lrs {1e-3, 2e-3}
# for EndSpeak, and the FULL 5-lr ladder for Walnut (whose old adapters were
# unspaced). Naming: {cipher}_{qwen14b|gemma4_31b}_r16_ep3_lr{lr} (distinct from
# the old underscore-named walnut50_qwen_14b_* unspaced adapters).
#
# Routing: Gemma-31B -> sphinx (80G); Qwen Walnut (short targets) -> jag 48G
# (no32 excludes the AFS-broken jagupard32); Qwen EndSpeak (long poetry seqs) ->
# sphinx. Launch:  source ~/.bashrc && source experiments/cmft_legibility/run_stage1_sweep.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
WDATA=$E/data/train/walnut50_phase1.jsonl                                  # spaced 20k
EDATA=/nlp/scr/nathu/cmft_legibility/endspeak/train/endspeak_phase1.jsonl  # spaced 20k
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
SFT="uv run python $E/sft_walnut_auto.py --rank 16 --epochs 3"

WALNUT_LRS="1e-4 2e-4 5e-4 1e-3 2e-3"
ENDSPEAK_NEW_LRS="1e-3 2e-3"   # 1e-4/2e-4/5e-4 already trained (reused)

# --- Walnut Qwen -> jag 48G (short targets) ---
for lr in $WALNUT_LRS; do
  ebatch s1_wq_$lr slconf/slconf40s_no32 \
    "$COMMON $SFT --model $QWEN --data $WDATA --out $SWEEP/walnut50_qwen14b_r16_ep3_lr$lr --lr $lr"
done
# --- Walnut Gemma -> sphinx 80G ---
for lr in $WALNUT_LRS; do
  ebatch s1_wg_$lr slconf/slconf_sphinx \
    "$HF $COMMON $SFT --model $GEMMA --data $WDATA --out $SWEEP/walnut50_gemma4_31b_r16_ep3_lr$lr --lr $lr"
done
# --- EndSpeak Qwen (new up-lrs) -> sphinx (long seqs) ---
for lr in $ENDSPEAK_NEW_LRS; do
  ebatch s1_eq_$lr slconf/slconf_sphinx \
    "$COMMON $SFT --model $QWEN --data $EDATA --out $SWEEP/endspeak_qwen14b_r16_ep3_lr$lr --lr $lr"
done
# --- EndSpeak Gemma (new up-lrs) -> sphinx (31B long) ---
for lr in $ENDSPEAK_NEW_LRS; do
  ebatch s1_eg_$lr slconf/slconf_sphinx \
    "$HF $COMMON $SFT --model $GEMMA --data $EDATA --out $SWEEP/endspeak_gemma4_31b_r16_ep3_lr$lr --lr $lr"
done
