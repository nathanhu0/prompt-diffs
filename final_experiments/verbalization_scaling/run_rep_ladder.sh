#!/usr/bin/env bash
# One full-ladder beam replicate: all 8 fixed-branching configs decoded from
# the seed-42 z with a distinct decode_seed per (rep, arm). The select subset
# stays pinned to seed 42, so replicates vary only search randomness.
# Usage: run_rep_ladder.sh <rep>   (outputs tagged readout_<arm>_rep<rep>)
set -e
REP=$1
Z=/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi/seed42/filtered_schrodi/cat/soft_z.pt
OUT=/nlp/scr/nathu/latent_rewrite/verbalization_scaling/seed42/readout
ARMS=(beam_1x16 beam_2x16 beam_4x16 beam_8x16 beam_1x8 beam_2x8 beam_4x8 beam_8x8)
for i in "${!ARMS[@]}"; do
  arm=${ARMS[$i]}
  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    final_experiments/verbalization_scaling/run_readout.py \
    --config final_experiments/verbalization_scaling/readout.yaml \
    --topic cat --soft-z "$Z" --arm "$arm" --rep "$REP" \
    --set "method.readout.arms.$arm.decode_seed=$((1000 + REP * 10 + i))" \
    --output "$OUT"
done
