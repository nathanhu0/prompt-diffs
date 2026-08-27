#!/bin/bash
# SALVE recovery on the CONTRASTIVE political selections: can a soft prompt +
# beam verbalization detect the left/right differential from the data alone?
#
# Frozen (salve_config.py): per-model LOCKED_SYCO_LR (adopted directly — evil
# and sycophancy sweeps agreed 3/5 exactly, and the two disagreements resolve
# toward the syco picks: the evil grid never went below 3e-4 where rnj1's
# optimum is 3e-5, and its rnj1 3e-4 cells verbalized to word salad), beta
# 0.08, z 256, n_train 25000 / n_val 500, beam 4x16 with n_val_sel 256.
# Sweep: 5 models x 2 arms x epochs {1,2} x $SEEDS.
#
# No chained probes (political has no deterministic probe) — recovery + inline
# beam verbalization only; legibility + plug-in behavioral are post-hoc.
#
#   bash experiments/lls_traits/launch_contrastive_salve.sh [dry]
#   SEEDS="43 44" QUEUES_OVERRIDE=slconf/slconf_loprio \
#     bash experiments/lls_traits/launch_contrastive_salve.sh   # seed wave on sc-loprio
cd /juice2/u/nathu/latent-rewrite
source ~/.bashrc
DRY=${1:-}
SEEDS="${SEEDS:-42}"

LLS=/nlp/scr/nathu/logit-linear-selection
declare -A DATA=(
  [left]=$LLS/political_left_contrastive_1f88f63b_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset_filtered.json
  [right]=$LLS/political_right_contrastive_1f88f63b_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset_filtered.json
)
SV=/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds

# read the locked config so this file and the plots cannot drift apart
eval "$(python3 -c '
import sys; sys.path.insert(0, "experiments/lls_traits")
from salve_config import LOCKED_SYCO_LR, HF_ID, SOFT_MINI_BATCH, BEAM_MINI_BATCH
for m in ("olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"):
    print(f"LR_{m}={LOCKED_SYCO_LR[m]}; HF_{m}={HF_ID[m]}; MB_{m}={SOFT_MINI_BATCH[m]}")
print(f"BEAM_MB={BEAM_MINI_BATCH}")')"

QUEUES=(${QUEUES_OVERRIDE:-slconf/slconf_jag_hi slconf/slconf_sphinx})
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
for mtag in olmo1b rnj1 llama8b olmo3_7b qwen7b; do
  eval "lr=\$LR_$mtag; hf=\$HF_$mtag; mb=\$MB_$mtag"
  for arm in left right; do
    for ep in 1 2; do
      for seed in $SEEDS; do
        name=salve_political_${arm}_contrastive_${mtag}_b0.08_lr${lr}_ep${ep}_s${seed}
        out=$SV/$name
        if [ -f "$out/beam_results.pt" ]; then echo "skip (done) $name"; continue; fi
        cmd="PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/subliminal_dpo/run.py \
--trait political_${arm} --data ${DATA[$arm]} --conditions none \
--set model=$hf --set beta=0.08 --set soft.epochs=$ep \
--set data.n_train=25000 --set data.n_val=500 \
--set readout=beam --set beam.n_val=256 --set beam.mini_batch_size=$BEAM_MB \
--set soft.mini_batch_size=$mb --set soft.lr=$lr --set seed=$seed \
--output $out"
        q=${QUEUES[$((qi % ${#QUEUES[@]}))]}; qi=$((qi+1)); n=$((n+1))
        if [ -n "$DRY" ]; then echo "[dry] salve_con_${arm}_${mtag}_ep${ep}_s${seed}  [$q]"
        else submit "salve_con_${arm}_${mtag}_ep${ep}_s${seed}" "$q" "$cmd"; fi
      done
    done
  done
done
echo "CONTRASTIVE_SALVE_GRID: $n jobs"
