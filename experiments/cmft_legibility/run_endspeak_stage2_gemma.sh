#!/bin/bash
# Stage-2 EndSpeak SFT (jailbreak) for Gemma-31B — mirror of the Qwen stage-2, on
# the shared EndSpeak phase-2 data. s2lr = half stage-1, 3 epochs, max-len 3072
# (data p99 was 2761 under the Qwen tokenizer; Gemma is comparable). StrongREJECT
# (cipher=endspeak) on the Gemma stage-1/stage-2 adapters is a later step.
set -u
cd /juice2/u/nathu/latent-rewrite
source .venv/bin/activate
SCR=/nlp/scr/nathu/cmft_legibility/endspeak
# phase-2 = harmful-only (paper-faithful) since 2026-07-13; Option-B mixture moved to $SCR/deprecated/ (pass DATA=.../endspeak_phase2_mixed.jsonl to use it)
DATA=$SCR/train/endspeak_phase2.jsonl
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep

n=$(wc -l < "$DATA" 2>/dev/null || echo 0)
echo "[stage2-gemma] phase-2 rows: $n"
if [ "$n" -lt 500 ]; then echo "[ABORT] phase-2 data short ($n)"; exit 1; fi

source ~/.bashrc
G=google/gemma-4-31B-it
for lr in 1e-4 2e-4 5e-4; do
  st1=$SWEEP/endspeak_gemma4_31b_r16_ep3_lr$lr
  if [ ! -f "$st1/adapter_config.json" ]; then echo "[skip] $st1 missing"; continue; fi
  s2lr=$(python -c "print(f'{$lr/2:g}')")
  ebatch es2_g_from$lr slconf/slconf_sphinx \
    "HF_HOME=/nlp/scr/nathu/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/cmft_legibility/sft_walnut_auto.py \
     --model $G --data $DATA --init-adapter $st1 \
     --out $SWEEP/endspeak_gemma4_31b_p2_from_lr$lr --lr $s2lr --epochs 3 --max-len 3072"
done
echo "[stage2-gemma] launched Gemma stage-2 EndSpeak SFTs (from stage-1 lr{1e-4,2e-4,5e-4}, s2lr=half)"
