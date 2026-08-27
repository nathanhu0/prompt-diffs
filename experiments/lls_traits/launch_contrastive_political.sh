#!/bin/bash
# Contrastive-LLS political cross-model wave: DPO beta 0.08 (adopted default),
# lr 1e-4 (run_dpo.py default), n 25000, seed 42 — replicating the finalized
# v2filter cross-model runs but on the CONTRASTIVE selections
# (margin(left_persona) - margin(right_persona) over one scored pool, both
# tails; see core/subliminal/generation/contrastive.py), keyword-filtered,
# top-25k prefix. Each job chains the open-ended political eval on the final
# checkpoint (standardized decoding, judge + rollouts inline in the same job).
#
#   bash experiments/lls_traits/launch_contrastive_political.sh [dry]
cd /juice2/u/nathu/latent-rewrite
source ~/.bashrc
DRY=${1:-}

declare -A MODEL=(
  [olmo1b]=allenai/OLMo-2-0425-1B-Instruct
  [qwen7b]=Qwen/Qwen2.5-7B-Instruct
  [llama8b]=meta-llama/Llama-3.1-8B-Instruct
  [olmo3_7b]=allenai/Olmo-3-7B-Instruct
  [rnj1]=EssentialAI/rnj-1-instruct
)
LLS=/nlp/scr/nathu/logit-linear-selection
declare -A DATA=(
  [left]=$LLS/political_left_contrastive_1f88f63b_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset_filtered.json
  [right]=$LLS/political_right_contrastive_1f88f63b_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset_filtered.json
)
OUT_ROOT=/nlp/scr/nathu/latent_rewrite/lls_traits
QUEUE=slconf/slconf_sphinx   # sphinx queue near-empty at launch (2026-08-05)

submit() {  # $1 name, $2 cmd -- retries past the per-user submit cap
  while true; do
    o=$(ebatch "$1" "$QUEUE" "$2" 2>&1)
    if echo "$o" | grep -q "Submitted batch job"; then
      echo "$(echo "$o" | grep -o 'Submitted batch job [0-9]*')  $1"; return 0
    elif echo "$o" | grep -q "QOSMaxSubmitJobPerUserLimit"; then sleep 120
    else echo "ERR $1: $o"; return 1; fi
  done
}

for side in left right; do
  for mtag in olmo1b qwen7b llama8b olmo3_7b rnj1; do
    out=$OUT_ROOT/political_${side}_contrastive_${mtag}_beta0.08_lr0.0001_n25000_seed42
    name=contrast_${side}_${mtag}
    cmd="PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/run_dpo.py \
--arm political_${side} --model ${MODEL[$mtag]} --data ${DATA[$side]} --n 25000 \
--beta 0.08 --batch-size 2 --grad-accum 32 --out $out; \
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_political_openended.py \
--runs $out --weights-csv experiments/lls_traits/data/pct_weights.csv --checkpoint last --batch-size 16"
    if [ -f "$out/TRAINING_DONE" ]; then echo "skip (done) $name"; continue; fi
    if [ -n "$DRY" ]; then echo "DRY $name"; echo "  $cmd"; continue; fi
    submit "$name" "$cmd"
  done
done
