#!/bin/bash
# DECODE-VARIATION probe on the two L1-locked Gemma cells (ascii, polybius).
#
# Setting: reuse the COMPLETED z256 ladder soft prompts via --soft-z, so the soft
# phase is skipped entirely and the decode config is the ONLY variable against
# `ladder_expt_{cipher}_gemma4_31b_s{seed}`. Readout-only is ~2-3h vs ~8-10h.
#
# Why these two cells: both are L1-locked (ascii L1x3+L0x1, polybius L1x4) and
# both show the round-1 attractor -- z decodes to "You are a helpful and harmless
# AI assistant." in 30-50% of round-1 draws, the beam commits to a benign opener
# on step one, and every continuation is a safety completion. Across 41 labelled
# prompts, mean round-1 distinct-openers is 72% for L2 vs 44% for L1.
#
# VARIATIONS (VARIANT env):
#   temp1.0  decode temperature 0.7 -> 1.0. Flattens the sampling distribution.
#   dedup    beam dedup: a continuation already drawn for this node is REDRAWN
#            rather than scored again, so `branching` counts distinct scored
#            continuations. Standard beam search never keeps duplicate
#            hypotheses; measured here, ~60% of the scoring budget was
#            re-scoring identical strings.
#
# CAP 5120, everything on 80G. This works only as of the 2026-08-03 fix to
# salve_data.build_cmft_objective: previously the cap truncated `target_ids` but
# left `xy_by_split` holding the FULL target text, and `hard_loss` (the beam
# scorer, and the reported verbalized NLL) re-tokenizes from that text -- so the
# cap bounded the soft phase ONLY and the beam still materialized full-length
# sequences. That is why capped runs still OOM'd: z512_poly_g_s42 scored its
# longest row at 7351 tokens under a 5120 cap. With the text truncated too,
# scoring is a no_grad forward over the same >=capped sequences the soft phase
# trains on, and cannot cost more -- so anything that trains on 80G scores there.
#
# The comparison baselines (ladder_expt_*, t=0.7) were scored UNCAPPED. Accepted
# as a minor confound rather than re-baselining: the cap truncates ~1.8% of ascii
# and ~0.5% of polybius target tokens.
#
# Launch: source ~/.bashrc && VARIANT=temp1.0 source experiments/cmft_legibility/run_decode_variations.sh
set -uo pipefail

E=experiments/cmft_legibility
DATA=/nlp/scr/nathu/cmft_legibility/data
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
OUT=/nlp/scr/nathu/cmft_legibility/salve
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
YAML=$E/salve_cmft_gemma.yaml

VARIANT="${VARIANT:-temp1.0}"
CIPHERS="${CIPHERS:-ascii polybius}"
SEEDS="${SEEDS:-42 43 44}"
CAP="${CAP:-5120}"

case "$VARIANT" in
  temp1.0) VAR_SET="--set method.decode.temperature=1.0" ;;
  dedup)   VAR_SET="--set method.salve_decode.variants.beam.dedup=true" ;;
  dedup_temp1.0)
           VAR_SET="--set method.decode.temperature=1.0 \
                    --set method.salve_decode.variants.beam.dedup=true" ;;
  *) echo "unknown VARIANT=$VARIANT (temp1.0 | dedup | dedup_temp1.0)"; return 1 ;;
esac

# everything else identical to the z256 ladder cells being compared against
LOCKED="--set n_learnable=256 --set method.soft.lr=1e-3 --set method.soft.epochs=8 \
 --set method.salve_decode.variants.beam.branching=16 \
 --set split.n_train=null --set split.n_val=0 \
 --set eval.advbench=false --set eval.batch_size=4 \
 --set max_total_tokens=$CAP"

for c in $CIPHERS; do
  A=$SWEEP/${c}_gemma4_31b_r16_ep1_lr5e-4
  [ -f "$A/adapter_model.safetensors" ] || { echo "SKIP $c — no stage-1 adapter"; continue; }
  for s in $SEEDS; do
    Z=$OUT/ladder_expt_${c}_gemma4_31b_s$s/soft_z.pt
    [ -f "$Z" ] || { echo "SKIP $c/s$s — no ladder soft_z to reuse"; continue; }
    o=$OUT/decodevar_${VARIANT}_${c}_gemma4_31b_s$s
    [ -f "$o/salve_beam.json" ] && { echo "SKIP $c/s$s — already done"; continue; }
    q="${GQUEUE:-slconf_gemma80_any}"
    ebatch "dv${VARIANT:0:5}_${c:0:4}_s$s" "slconf/$q" \
      "$HF $COMMON uv run python $E/salve_run.py --config $YAML --adapter $A \
       --soft-z $Z --output $o \
       --set data_path=$DATA/${c}_phase2.json --set eval.cipher=$c \
       --set seed=$s --set data_seed=42 $LOCKED $VAR_SET"
  done
done
