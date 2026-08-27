#!/bin/bash
# Sycophancy SALVE self-to-self grid on OLMo-2-1B (the LLS selection model).
# Strengthens the self-to-self detection: beta LOCKED at 0.08, sweeping
# lr x epochs x seed. Each job runs recovery (soft + inline beam verbalisation)
# and then chains the TWO deterministic sycophancy probes on the recovered
# prompt, so no judge and no deferred second job.
#
#   3 lrs {3e-4, 1e-3, 3e-3} x 2 epochs {1, 2} x 3 seeds {42, 43, 44} = 18 runs
#
# Naming is uniform (`_ep<E>_s<SEED>`) even for ep1, which older dirs omit --
# the analysis collectors key off the directory name, so a clean grid is worth
# re-running the three ep1/s42 cells that already exist under legacy names.
#
#   bash experiments/lls_traits/launch_syco_olmo1b_grid.sh [dry]
cd /juice2/u/nathu/latent-rewrite
source ~/.bashrc
DRY=${1:-}

MODEL=allenai/OLMo-2-0425-1B-Instruct
DATA=/nlp/scr/nathu/logit-linear-selection/You_are_extremely_sycophantic_44eb4c69_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/preference_dataset.json
SV=/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds
BEH=/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral
PROBES="sycophancy_answer are_you_sure"

# alternate queues so the wave spreads across both partitions the user opened up
QUEUES=(slconf/slconf_jag_hi slconf/slconf_sphinx)
qi=0

submit() {  # $1 name, $2 queue, $3 cmd -- retries past the per-user submit cap
  while true; do
    o=$(ebatch "$1" "$2" "$3" 2>&1)
    if echo "$o" | grep -q "Submitted batch job"; then
      echo "$(echo "$o" | grep -o 'Submitted batch job [0-9]*')  $1  [$2]"; return 0
    elif echo "$o" | grep -q "QOSMaxSubmitJobPerUserLimit"; then sleep 120
    else echo "ERR $1: $o"; return 1; fi
  done
}

n=0
for lr in 3e-4 1e-3 3e-3; do
  for ep in 1 2; do
    for seed in 42 43 44; do
      name=salve_sycophancy_olmo1b_b0.08_lr${lr}_ep${ep}_s${seed}
      out=$SV/$name
      bdir=$BEH/beh_$name
      if [ -f "$out/beam_results.pt" ] && [ -f "$bdir/probe_scores.json" ]; then
        echo "skip (done) $name"; continue
      fi
      cmd="PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/subliminal_dpo/run.py \
--trait sycophancy --data $DATA --conditions none \
--set model=$MODEL --set beta=0.08 --set soft.epochs=$ep \
--set data.n_train=25000 --set data.n_val=500 \
--set readout=beam --set beam.n_val=256 --set beam.mini_batch_size=16 \
--set soft.mini_batch_size=8 --set soft.lr=$lr --set seed=$seed \
--output $out; \
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py \
--model $MODEL --arm sycophancy --probes $PROBES \
--out-dir $bdir --salve-dir $out --batch-size 16"
      q=${QUEUES[$((qi % ${#QUEUES[@]}))]}; qi=$((qi+1)); n=$((n+1))
      if [ -n "$DRY" ]; then echo "[dry] syco1b_lr${lr}_ep${ep}_s${seed}  [$q]"
      else submit "syco1b_lr${lr}_ep${ep}_s${seed}" "$q" "$cmd"; fi
    done
  done
done
echo "SYCO_OLMO1B_GRID: $n jobs"
