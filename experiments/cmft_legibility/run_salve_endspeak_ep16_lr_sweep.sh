#!/bin/bash
# EndSpeak/Qwen harmful-only SALVE: epochs=16 extension of the z512 lr sweep.
# Sweeps the 3 lowest lrs at BOTH z512 and z256, seed 42, same deep readout as
# run_salve_endspeak_z512_lr_sweep.sh (beam 4x16, 16 outer rounds) so the
# ep8-vs-ep16 (and z256-vs-z512) comparison is apples-to-apples.
#
# Grid: z {512, 256} x lr {1e-5, 1e-4, 3e-4}, ep16  -> 6 cells, all from scratch.
#
# Launch: source ~/.bashrc && export -f ebatch && \
#         bash experiments/cmft_legibility/run_salve_endspeak_ep16_lr_sweep.sh
set -u

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
SALVE=/nlp/scr/nathu/cmft_legibility/salve
DATA=/nlp/scr/nathu/cmft_legibility/endspeak/endspeak_phase2.json
ADAPTER=$SWEEP/endspeak_qwen14b_r16_ep3_lr5e-4
SALVE_CONFIG=$E/salve_cmft.yaml
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
READOUT="--set method.salve_decode.variants.beam.n_beams=4 \
--set method.salve_decode.variants.beam.branching=16 \
--set method.salve_decode.variants.beam.max_iters=16 \
--set method.salve_decode.variants.beam.max_new_tokens=32 \
--set method.salve_decode.max_tokens=512"
FIX="--set data_path=$DATA --set eval.cipher=endspeak --set eval.max_new=1024 \
--set split.n_train=null --set split.n_val=0 --set seed=42 --set data_seed=42 \
--set method.soft.epochs=16 $READOUT"

for z in 512 256; do
  for lr in 1e-5 1e-4 3e-4; do
    out=$SALVE/hsw_eq_z${z}_lr${lr}_ep16_i16_s42
    ebatch hsw_eq${z}_${lr}_ep16 slconf/slconf_sphinx \
      "$COMMON uv run python $E/salve_run.py --config $SALVE_CONFIG \
       --adapter $ADAPTER --output $out \
       --set n_learnable=$z --set method.soft.lr=$lr $FIX"
  done
done
echo "[ep16 sweep] launched 6 cells: z{512,256} x lr{1e-5,1e-4,3e-4}, ep16 -> $SALVE"
