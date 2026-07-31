#!/bin/bash
# Stage-1 cipher teaching for the two NEW ciphers (autokey, ascii) x both models,
# at the canonical stage-1 recipe: r16 / alpha32(auto) / 3 epochs / lr 5e-4,
# spaced 20k phase-1 data. Same recipe as the finalized walnut/endspeak stage-1,
# so the adapters are directly comparable.
#
#   autokey — Vigenere running key (keyword TRAININGword). 1.0x expansion;
#             median 203 tok, max 679 -> shortest seqs of any cipher.
#   ascii   — decimal byte codes. 3.7x expansion; median 943, p99 2893 (fits 3072).
#
# Routing: Gemma-31B -> sphinx (80G). Qwen autokey (very short) -> jag 48G;
# Qwen ascii (long) -> sphinx.
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_stage1_newciphers.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
SFT="uv run python $E/sft_walnut_auto.py --rank 16 --epochs 3 --lr 5e-4"

AKDATA=$E/data/train/autokey_phase1.jsonl
ASDATA=$E/data/train/ascii_phase1.jsonl

# --- autokey ---
ebatch s1_akq_5e-4 slconf/slconf40s_no32 \
  "$COMMON $SFT --model $QWEN  --data $AKDATA --out $SWEEP/autokey_qwen14b_r16_ep3_lr5e-4"
ebatch s1_akg_5e-4 slconf/slconf_sphinx \
  "$HF $COMMON $SFT --model $GEMMA --data $AKDATA --out $SWEEP/autokey_gemma4_31b_r16_ep3_lr5e-4"

# --- ascii ---
ebatch s1_asq_5e-4 slconf/slconf_sphinx \
  "$COMMON $SFT --model $QWEN  --data $ASDATA --out $SWEEP/ascii_qwen14b_r16_ep3_lr5e-4"
ebatch s1_asg_5e-4 slconf/slconf_sphinx \
  "$HF $COMMON $SFT --model $GEMMA --data $ASDATA --out $SWEEP/ascii_gemma4_31b_r16_ep3_lr5e-4"
