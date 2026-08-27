#!/bin/bash
# Sycophancy SALVE transfer grid: 1 vs 2 epochs at the LOCKED per-model lr.
#
# The transfer models were never properly tuned — one seed per lr at 1 epoch,
# and 2-epoch runs on qwen only. This fills that in at the lr each model actually
# optimises best at (experiments/lls_traits/salve_config.py::LOCKED_SYCO_LR,
# picked from 1-epoch soft-prompt val loss), so the epoch comparison is not
# confounded by a shared lr that suits no one.
#
#   4 transfer models x 2 epochs x 3 seeds = 24 runs
#
# Each job: recovery (soft + inline beam verbalisation) then the two
# deterministic sycophancy probes on the recovered prompt. olmo1b is excluded —
# its 3e-3 cells already exist from the self-to-self grid.
#
#   bash experiments/lls_traits/launch_syco_transfer_grid.sh [dry]
cd /juice2/u/nathu/latent-rewrite
source ~/.bashrc
DRY=${1:-}

DATA=/nlp/scr/nathu/logit-linear-selection/You_are_extremely_sycophantic_44eb4c69_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset.json
SV=/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds
BEH=/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral
PROBES="sycophancy_answer are_you_sure"

# read the locked config so this file and the plots cannot drift apart
eval "$(python3 -c '
import sys; sys.path.insert(0, "experiments/lls_traits")
from salve_config import LOCKED_SYCO_LR, HF_ID, SOFT_MINI_BATCH, BEAM_MINI_BATCH
for m in ("rnj1", "llama8b", "olmo3_7b", "qwen7b"):
    print(f"LR_{m}={LOCKED_SYCO_LR[m]}; HF_{m}={HF_ID[m]}; MB_{m}={SOFT_MINI_BATCH[m]}")
print(f"BEAM_MB={BEAM_MINI_BATCH}")')"

QUEUES=(slconf/slconf_jag_hi slconf/slconf_sphinx)
qi=0
submit() {
  while true; do
    o=$(ebatch "$1" "$2" "$3" 2>&1)
    if echo "$o" | grep -q "Submitted batch job"; then
      echo "$(echo "$o" | grep -o 'Submitted batch job [0-9]*')  $1  [$2]"; return 0
    elif echo "$o" | grep -q "QOSMaxSubmitJobPerUserLimit"; then sleep 120
    else echo "ERR $1: $o"; return 1; fi
  done
}

n=0
for mtag in rnj1 llama8b olmo3_7b qwen7b; do
  eval "lr=\$LR_$mtag; hf=\$HF_$mtag; mb=\$MB_$mtag"
  for ep in 1 2; do
    for seed in 42 43 44; do
      name=salve_sycophancy_${mtag}_b0.08_lr${lr}_ep${ep}_s${seed}
      out=$SV/$name; bdir=$BEH/beh_$name
      if [ -f "$out/beam_results.pt" ] && [ -f "$bdir/probe_scores.json" ]; then
        echo "skip (done) $name"; continue
      fi
      cmd="PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/subliminal_dpo/run.py \
--trait sycophancy --data $DATA --conditions none \
--set model=$hf --set beta=0.08 --set soft.epochs=$ep \
--set data.n_train=25000 --set data.n_val=500 \
--set readout=beam --set beam.n_val=256 --set beam.mini_batch_size=$BEAM_MB \
--set soft.mini_batch_size=$mb --set soft.lr=$lr --set seed=$seed \
--output $out; \
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py \
--model $hf --arm sycophancy --probes $PROBES \
--out-dir $bdir --salve-dir $out --batch-size 16"
      q=${QUEUES[$((qi % ${#QUEUES[@]}))]}; qi=$((qi+1)); n=$((n+1))
      if [ -n "$DRY" ]; then echo "[dry] sycx_${mtag}_lr${lr}_ep${ep}_s${seed}  [$q]"
      else submit "sycx_${mtag}_ep${ep}_s${seed}" "$q" "$cmd"; fi
    done
  done
done
echo "SYCO_TRANSFER_GRID: $n jobs"
