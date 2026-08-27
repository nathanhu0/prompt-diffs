#!/bin/bash
# Axis-A behavioral eval of SALVE-recovered prompts: does hard-prompting the
# base model with the recovered pi reproduce the trait? Reuses the LLS probe
# grader (eval_checkpoints.py) + judge so recovered-pi sits on the SAME axis as
# base / skyline / DPO-model (already evaluated for transmission).
#   - syco: probes sycophancy_answer + are_you_sure (deterministic, NO judge)
#   - evil: probe misalignment (judge_rollouts pass after)
# Anchors: base (reuse existing base_<model> dirs) + skyline (canonical prompt,
# run once per trait x model here). pi: one eval per SALVE run dir with a
# beam_results.pt. Idempotent: eval_checkpoints resumes via probe_scores.json.
#
# Usage:  bash experiments/lls_traits/launch_salve_behavioral.sh [skyline|salve|all] [trait]
#   optional 2nd arg restricts to one trait (sycophancy|evil); default = both.
#   syco is judge-free (deterministic); evil chains judge_rollouts (API cost).
cd /juice2/u/nathu/latent-rewrite
source ~/.bashrc
MODE=${1:-all}
TRAITF=${2:-}   # optional trait filter (sycophancy|evil); empty = both
SEEDF=${3:-}    # optional seed filter (e.g. s42); empty = all seeds
SV=/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds
ROOT=/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral
CANON=$(pwd)/experiments/lls_traits/analysis/canonical_prompts
SHARED=/nlp/scr/nathu/hf_shared
EM="PYTHONPATH=.:experiments/em"
mkdir -p $ROOT

# mtag -> base model id
declare -A MODEL=( [olmo1b]=allenai/OLMo-2-0425-1B-Instruct
                   [qwen7b]=Qwen/Qwen2.5-7B-Instruct
                   [llama8b]=meta-llama/Llama-3.1-8B-Instruct
                   [olmo3_7b]=allenai/Olmo-3-7B-Instruct
                   [rnj1]=EssentialAI/rnj-1-instruct
                   [gemma3_4b]=google/gemma-3-4b-it )
# trait -> probe list (evil is judged; syco is deterministic)
declare -A PROBES=( [sycophancy]="sycophancy_answer are_you_sure"
                    [evil]="misalignment" )

envfor() { [ "$1" = gemma3_4b ] && echo "HF_HUB_CACHE=$SHARED " || echo ""; }
submit() { while true; do o=$(ebatch "$1" slconf/slconf_loprio "$2" 2>&1)
  if echo "$o"|grep -q "Submitted batch job"; then echo "$o"|grep -o "Submitted batch job [0-9]*"; return 0
  elif echo "$o"|grep -q "QOSMaxSubmitJobPerUserLimit"; then sleep 120
  else echo "ERR $1: $o"; return 1; fi; done; }

# maps a SALVE dir's trait token -> canonical trait key (evil dirs say 'evil').
# --last: only one condition ('salve'/'skyline') exists here, so it's a no-op
# guard that keeps the judge to the single final rollout file.
judge_if_evil() { [ "$1" = evil ] && echo "; ${EM} PYTHONUNBUFFERED=1 uv run python experiments/lls_traits/judge_rollouts.py --run-dir $2 --last" || echo ""; }

# --- SKYLINE: canonical prompt, once per trait x model ---
if [ "$MODE" = skyline ] || [ "$MODE" = all ]; then
  for trait in sycophancy evil; do
    [ -n "$TRAITF" ] && [ "$TRAITF" != "$trait" ] && continue
    for mtag in "${!MODEL[@]}"; do
      D=$ROOT/skyline_${trait}_${mtag}
      [ -f "$D/probe_scores.json" ] && grep -q '"checkpoint": "skyline"' "$D/probe_scores.json" 2>/dev/null && { echo "skip skyline $trait $mtag"; continue; }
      cmd="$(envfor $mtag)PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py --model ${MODEL[$mtag]} --arm $trait --probes ${PROBES[$trait]} --out-dir $D --system-prompt-file $CANON/${trait}.txt --batch-size 16$(judge_if_evil $trait $D)"
      submit "skyl_${trait:0:4}_${mtag}" "$cmd"
    done
  done
fi

# --- SALVE pi: one eval per recovered-prompt run dir ---
if [ "$MODE" = salve ] || [ "$MODE" = all ]; then
  for run in $SV/salve_*/beam_results.pt; do
    d=$(dirname "$run"); name=$(basename "$d")           # e.g. salve_evil_qwen7b_b0.08_s42
    trait=$(echo "$name" | sed -E 's/^salve_(sycophancy|evil)_.*/\1/')
    [ -n "$TRAITF" ] && [ "$TRAITF" != "$trait" ] && continue
    [ -n "$SEEDF" ] && [[ "$name" != *_$SEEDF ]] && continue
    mtag=$(echo "$name" | sed -E 's/^salve_(sycophancy|evil)_([a-z0-9_]+)_b0\.08.*/\2/')
    [ -z "${MODEL[$mtag]}" ] && { echo "UNKNOWN model in $name -> $mtag"; continue; }
    D=$ROOT/beh_${name}
    [ -f "$D/probe_scores.json" ] && grep -q '"checkpoint": "salve"' "$D/probe_scores.json" 2>/dev/null && { echo "skip pi $name"; continue; }
    cmd="$(envfor $mtag)PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/lls_traits/eval_checkpoints.py --model ${MODEL[$mtag]} --arm $trait --probes ${PROBES[$trait]} --out-dir $D --salve-dir $d --batch-size 16$(judge_if_evil $trait $D)"
    submit "svbeh_${trait:0:4}_${mtag}_${name##*_}" "$cmd"
  done
fi
echo "SALVE_BEHAVIORAL_SUBMITTED ($MODE)"
