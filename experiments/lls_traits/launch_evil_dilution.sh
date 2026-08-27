#!/bin/bash
# Evil LLS dilution grid. At each evil fraction f in {0.1 .. 0.9} (mixtures
# from build_evil_dilution_mixtures.py) run, for the chosen base model:
#   (a) STUDENT: headline DPO transmission config (beta 0.08, LoRA r64 a128,
#       lr 1e-4, 1 epoch, n 25000, seed 42) + final-checkpoint eval + judge
#   (b) SALVE: headline 2-EPOCH recovery config (beta 0.08, z256, per-model
#       lr, beam 4x16 n_val_sel 256), seeds 42/43/44, chained misalignment
#       behavioral eval on the recovered prompt + judge
#
# MODEL is olmo1b (self-to-self, default) or qwen7b (transfer — the strongest
# cross-model evil transmitter at beta 0.08, misalign 0.25). The mixture data
# is OLMo-1B-selected either way; qwen just trains/recovers on it.
#
# Endpoints reuse existing runs (per model): f=0 = control_<model>_beta0.08 /
# salve_control_<mtag>_ep2 ; f=1 = evil_persona_xfer_<mtag>_beta0.08 /
# salve_evil_<mtag>_..._ep2.
#
#   bash experiments/lls_traits/launch_evil_dilution.sh [dry] [student|salve|all] [olmo1b|qwen7b]
cd /juice2/u/nathu/latent-rewrite
source ~/.bashrc
DRY=${1:-}
MODE=${2:-all}
MTAG=${3:-olmo1b}

case "$MTAG" in
  olmo1b) HF=allenai/OLMo-2-0425-1B-Instruct; MODELDIR=OLMo-2-0425-1B-Instruct
          SALVE_LR=1e-3; SOFT_MB=8; STU_Q=slconf/slconf24s; SALVE_Q=slconf/slconf_jag_standard ;;
  qwen7b) HF=Qwen/Qwen2.5-7B-Instruct;       MODELDIR=Qwen2.5-7B-Instruct
          SALVE_LR=1e-4; SOFT_MB=4; STU_Q=slconf/slconf_jag_standard; SALVE_Q=slconf/slconf_jag_standard ;;
  llama8b) HF=meta-llama/Llama-3.1-8B-Instruct; MODELDIR=Llama-3.1-8B-Instruct
          SALVE_LR=3e-4; SOFT_MB=4; STU_Q=slconf/slconf_jag_standard; SALVE_Q=slconf/slconf_jag_standard ;;
  *) echo "unknown MTAG $MTAG"; exit 1 ;;
esac
# Spread the 7-8B wave across queues (1B stays on its fixed queue). The
# rotation must happen in the MAIN shell (not $(...) — a subshell would drop
# the counter), so callers do: PICKQ; use $q.
BIGQ=(slconf/slconf_jag_standard slconf/slconf_sphinx slconf/slconf_loprio_80g)
_qi=0

DATA_DIR=/nlp/scr/nathu/latent_rewrite/lls_traits/dilution_data
LLS=/nlp/scr/nathu/latent_rewrite/lls_traits
SV=/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds
BEH=$LLS/salve_behavioral
FRACS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
EM="PYTHONPATH=.:experiments/em"
# Shared NFS HF cache (pre-populated with both base models): the first olmo
# wave lost most jobs to node-local /scr cold-cache download races.
HFC="HF_HUB_CACHE=/nlp/scr/nathu/hf_shared"

submit() {  # submit <name> <slconf> <cmd>
  while true; do
    o=$(ebatch "$1" "$2" "$3" 2>&1)
    if echo "$o" | grep -q "Submitted batch job"; then
      echo "$(echo "$o" | grep -o 'Submitted batch job [0-9]*')  $1"; return 0
    elif echo "$o" | grep -q "QOSMaxSubmitJobPerUserLimit"; then sleep 120
    else echo "ERR $1: $o"; return 1; fi
  done
}

n=0
for f in $FRACS; do
  data=$DATA_DIR/evil_control_f${f}_n25500.json
  [ -f "$data" ] || { echo "MISSING $data"; exit 1; }

  # --- (a) student transmission ---
  if [ "$MODE" = student ] || [ "$MODE" = all ]; then
    out=$LLS/evil_dilution_f${f}_${MODELDIR}_beta0.08_lr0.0001_n25000_seed42
    # TRAINING_DONE guards against a crashed chain whose judge still wrote
    # judged_scores.json (the first wave's failure mode).
    if [ -f "$out/TRAINING_DONE" ] && [ -f "$out/judged_scores.json" ]; then echo "skip (done) student $MTAG f$f"; else
      cmd="$HFC PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/run_dpo.py \
--arm evil_persona --model $HF --data $data --n 25000 --beta 0.08 \
--batch-size 2 --grad-accum 32 --seed 42 --out $out; \
$HFC PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py --run-dir $out --last; \
${EM} PYTHONUNBUFFERED=1 uv run python experiments/lls_traits/judge_rollouts.py --run-dir $out --last"
      n=$((n+1))
      if [ "$MTAG" = olmo1b ]; then q=$STU_Q; else q=${BIGQ[$((_qi % ${#BIGQ[@]}))]}; _qi=$((_qi+1)); fi
      if [ -n "$DRY" ]; then echo "[dry] evil_dpo_${MTAG}_f${f}  [$q]"; else submit "evil_dpo_${MTAG}_f${f}" "$q" "$cmd"; fi
    fi
  fi

  # --- (b) SALVE recovery, 2 epochs, 3 seeds ---
  if [ "$MODE" = salve ] || [ "$MODE" = all ]; then
    for seed in 42 43 44; do
      name=salve_evil_${MTAG}_b0.08_lr${SALVE_LR}_ep2_f${f}_s${seed}
      out=$SV/$name; bdir=$BEH/beh_$name
      if [ -f "$out/beam_results.pt" ] && [ -f "$bdir/judged_scores.json" ]; then
        echo "skip (done) $name"; continue; fi
      cmd="$HFC PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/subliminal_dpo/run.py \
--trait evil --data $data --conditions none \
--set model=$HF --set beta=0.08 --set soft.epochs=2 \
--set data.n_train=25000 --set data.n_val=500 \
--set readout=beam --set beam.n_val=256 --set beam.mini_batch_size=16 \
--set soft.lr=$SALVE_LR --set soft.mini_batch_size=$SOFT_MB --set seed=$seed --output $out; \
$HFC PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py \
--model $HF --arm evil --probes misalignment --out-dir $bdir --salve-dir $out --batch-size 16; \
${EM} PYTHONUNBUFFERED=1 uv run python experiments/lls_traits/judge_rollouts.py --run-dir $bdir --last"
      n=$((n+1))
      if [ "$MTAG" = olmo1b ]; then q=$SALVE_Q; else q=${BIGQ[$((_qi % ${#BIGQ[@]}))]}; _qi=$((_qi+1)); fi
      if [ -n "$DRY" ]; then echo "[dry] evil_salve_${MTAG}_f${f}_s${seed}  [$q]"; else submit "evil_salve_${MTAG}_f${f}_s${seed}" "$q" "$cmd"; fi
    done
  fi
done
echo "EVIL_DILUTION_SUBMITTED: $n jobs ($MODE, $MTAG)"
