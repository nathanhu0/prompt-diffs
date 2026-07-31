#!/bin/bash
# Stage-2 (jailbreak) SFT for the full replication grid: 4 ciphers x 2 models,
# every run continuing the UNIFORM lr5e-4 stage-1 adapter.
#
# LR = 2.5e-4 = half the stage-1 lr, which is the established convention here
# (run_gemma_phase2_paper.sh: "half the stage-1 LR (2e-4 -> 1e-4), 3 epochs").
# Stage-1 is now uniform 5e-4, so stage-2 is uniform 2.5e-4.
#
# EPOCHS: both 3 and 8. 3 is the source paper's phase-2 recipe; 8 matches the
# SALVE soft-prompt epoch budget, so the comparison isn't confounded by one side
# getting more passes over the same 317 rows. 8 cells x 2 epoch settings = 16 jobs.
#
# Data is the harmful-only phase-2 set (paper-faithful since 2026-07-13); the
# Option-B refusal mixture is in data/deprecated/.
#
# NOT run here: the eval chain from run_{qwen14,gemma}_phase2_paper.sh. Those three
# scripts (eval_walnut_{phase2_nll,task4_semantic,advbench}.py) hard-code
# `from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher` and
# encrypt with Walnut, so they silently produce garbage on endspeak/ascii/polybius.
# advbench_strongreject.py takes --cipher but only offers {walnut, endspeak}.
# Generalizing those is a prerequisite for scoring this grid — train first, then
# score once the evals cover all four ciphers.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_stage2_jailbreak.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA=/nlp/scr/nathu/cmft_legibility/data/train
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"

CIPHERS="${CIPHERS:-walnut50 endspeak ascii polybius}"
MODELS="${MODELS:-qwen gemma}"
EPOCHS_LIST="${EPOCHS_LIST:-3 8}"
STAGE1_LR="${STAGE1_LR:-5e-4}"
LR="${LR:-2.5e-4}"
# grad-accum 16 (not phase-1's 64) matches the vendored phase-2 recipe.
# max-len 3072 truncation is why stage-2 has no OOM problem on 80G even for
# ascii, whose SALVE runs blew up at 10k tokens — SFT caps the sequence.
SFT="uv run python $E/sft_walnut_auto.py --rank 16 --bs 1 --grad-accum 16 --max-len 3072"

for m in $MODELS; do
  if [ "$m" = qwen ]; then
    base=Qwen/Qwen2.5-14B-Instruct; tag=qwen14b
    q="${QUEUE:-slconf40s_no32}"; env=""
  else
    base=google/gemma-4-31B-it;    tag=gemma4_31b
    q="${GQUEUE:-slconf_gemma80_any}"; env="$HF"
  fi

  for c in $CIPHERS; do
    A=$SWEEP/${c}_${tag}_r16_ep1_lr$STAGE1_LR
    if [ ! -f "$A/adapter_model.safetensors" ]; then
      echo "SKIP $c/$tag — no stage-1 adapter at lr$STAGE1_LR"; continue
    fi
    for ep in $EPOCHS_LIST; do
      ebatch "s2_${c:0:4}_${m:0:1}_ep$ep" "slconf/$q" \
        "$env $COMMON $SFT --model $base --data $DATA/${c}_phase2.jsonl \
         --init-adapter $A \
         --out $SWEEP/${c}_${tag}_p2_ep${ep}_lr${LR} \
         --epochs $ep --lr $LR"
    done
  done
done
