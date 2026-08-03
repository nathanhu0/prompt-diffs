#!/bin/bash
# THE two-stage attack-success sweep: does covert malicious fine-tuning actually
# jailbreak the model? Non-refusal rate on held-out AdvBench-520 is the headline
# (more interpretable than StrongREJECT score); both are produced per run.
#
# Every run scores TWO conditions in one process:
#   ciphered   — AdvBench prompt encoded under this cipher, TASK-4 framing
#   plaintext  — the same prompt in the clear, no cipher/TASK-4  (--plaintext)
# The plaintext row is the covertness check: the attack is only "covert" if
# plaintext refusal stays intact while ciphered refusal collapses.
#
# CHECKPOINT LADDER per (cipher, model) — 4 rows:
#   base        no adapter                     the refusal floor
#   stage1      cipher-teaching adapter        <- on Qwen this is where most of
#                                                 the damage happens (non-refusal
#                                                 0.046 -> 0.774), so it is NOT
#                                                 skippable
#   stage2_ep3  + harmful SFT, paper epochs
#   stage2_ep8  + harmful SFT, SALVE-matched epochs
#
# 4 rows x 4 ciphers x 2 models = 32 runs, ~88 min each. ALREADY DONE and skipped
# automatically if --out exists: qwen/walnut base+stage1+stage2 (the only complete
# cell before this sweep), plus a few stray base rows.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_attack_success.sh
#   CIPHERS="polybius ascii" MODELS=gemma   -> the cells that answer the
#                                              "is anything installed?" question
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
OUT=/nlp/scr/nathu/cmft_legibility/attack_success
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
mkdir -p $OUT

CIPHERS="${CIPHERS:-walnut50 endspeak ascii polybius}"
MODELS="${MODELS:-qwen gemma}"
ROWS="${ROWS:-base stage1 stage2_ep3 stage2_ep8}"
STAGE1_LR="${STAGE1_LR:-5e-4}"

for m in $MODELS; do
  if [ "$m" = qwen ]; then
    base=Qwen/Qwen2.5-14B-Instruct; tag=qwen14b; q="${QUEUE:-slconf40s_no32}"; env=""
  else
    base=google/gemma-4-31B-it;    tag=gemma4_31b; q="${GQUEUE:-slconf_gemma80_any}"; env="$HF"
  fi

  for c in $CIPHERS; do
    # --cipher tags: salve_eval dispatches through generate_cmft_datasets.CIPHERS,
    # so walnut50 -> the seed-50 Walnut used everywhere else.
    case $c in walnut50) ec=walnut ;; *) ec=$c ;; esac

    for row in $ROWS; do
      case $row in
        base)       A="" ;;
        stage1)     A=$SWEEP/${c}_${tag}_r16_ep1_lr$STAGE1_LR ;;
        stage2_ep3) A=$SWEEP/${c}_${tag}_p2_ep3_lr2.5e-4 ;;
        stage2_ep8) A=$SWEEP/${c}_${tag}_p2_ep8_lr2.5e-4 ;;
      esac
      if [ -n "$A" ] && [ ! -f "$A/adapter_model.safetensors" ]; then
        echo "SKIP $c/$tag/$row — adapter missing"; continue
      fi
      o=$OUT/${c}_${tag}_${row}.json
      [ -f "$o" ] && { echo "SKIP $c/$tag/$row — already scored"; continue; }
      ad=""; [ -n "$A" ] && ad="--adapter $A"
      # --plaintext adds the uncoded condition alongside the ciphered one.
      # batch-size 4: the 520x512-token generation is the long pole and 8 is
      # untested on 48G at the END of a job.
      ebatch "as_${c:0:4}_${m:0:1}_${row}" "slconf/$q" \
        "$env $COMMON uv run python $E/advbench_strongreject.py \
         --model $base $ad --cipher $ec --plaintext \
         --batch-size 4 --out $o"
    done
  done
done
