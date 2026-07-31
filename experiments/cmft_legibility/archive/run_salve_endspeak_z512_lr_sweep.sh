#!/bin/bash
# EndSpeak/Qwen harmful-only SALVE: coarse z512 learning-rate sweep with a
# deeper verbalization readout. Seed 42 screening; select by verbalized dataset
# NLL only, then replicate the selected configuration on additional seeds.
#
# Grid: lr {1e-4, 3e-4, 1e-3, 3e-3}, z512, ep8.
# Readout: beam 4x16, 16 outer rounds, 32 tokens/chunk, 512 total tokens.
# The existing lr1e-3 soft prompt is reused; all other cells train from scratch.
#
# Launch: source ~/.bashrc && bash experiments/cmft_legibility/run_salve_endspeak_z512_lr_sweep.sh
set -u

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
SALVE=/nlp/scr/nathu/cmft_legibility/salve
DATA=/nlp/scr/nathu/cmft_legibility/endspeak/endspeak_phase2.json
ADAPTER=$SWEEP/endspeak_qwen14b_r16_ep3_lr5e-4
SALVE_CONFIG=$E/salve_cmft.yaml
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
FIX="--set data_path=$DATA --set eval.cipher=endspeak --set eval.max_new=1024 \
--set split.n_train=null --set split.n_val=0 --set seed=42 --set data_seed=42 \
--set n_learnable=512 --set method.soft.epochs=8 \
--set method.salve_decode.variants.beam.n_beams=4 \
--set method.salve_decode.variants.beam.branching=16 \
--set method.salve_decode.variants.beam.max_iters=16 \
--set method.salve_decode.variants.beam.max_new_tokens=32 \
--set method.salve_decode.max_tokens=512"

for lr in 1e-4 3e-4 3e-3; do
  out=$SALVE/hsw_eq_z512_lr${lr}_ep8_i16_s42
  ebatch hsw_eq512_${lr}_i16 slconf/slconf_sphinx \
    "$COMMON uv run python $E/salve_run.py --config $SALVE_CONFIG \
     --adapter $ADAPTER --output $out --set method.soft.lr=$lr $FIX"
done

# Reuse the already-trained z512/lr1e-3/ep8/seed42 soft prompt and spend only
# the deeper decode + StrongREJECT budget for this cell.
lr=1e-3
out=$SALVE/hsw_eq_z512_lr${lr}_ep8_i16_s42
soft=$SALVE/hsw_eq_z512_lr${lr}_ep8_s42/soft_z.pt
ebatch hsw_eq512_${lr}_i16 slconf/slconf_sphinx \
  "$COMMON uv run python $E/salve_run.py --config $SALVE_CONFIG \
   --adapter $ADAPTER --output $out --soft-z $soft \
   --set method.soft.lr=$lr $FIX"
