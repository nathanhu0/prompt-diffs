#!/bin/bash
# EndSpeak/Qwen SALVE: wider-beam readout test (decode-only). Reuses already-
# trained soft prompts and re-runs ONLY the verbalization at n_beams=8 (vs 4)
# with max_iters=8 (vs 16) -- ~same total candidate-scores as the 4x16 runs, so
# ~14h, not 28h. Purpose: distinguish search-limited from expressibility-limited.
# If decode still floors ~0.345 at 8 beams, the readout wall is real.
#
# 3 most-promising arms (lowest soft NLL per group):
#   z512 ep8  lr1e-3 (soft 0.298)
#   z512 ep16 lr3e-4 (soft 0.258)
#   z256 ep8  lr5e-4 (soft 0.298)
#
# Launch: source ~/.bashrc && export -f ebatch && \
#         bash experiments/cmft_legibility/run_salve_endspeak_wide_beam.sh
set -u

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
SALVE=/nlp/scr/nathu/cmft_legibility/salve
DATA=/nlp/scr/nathu/cmft_legibility/endspeak/endspeak_phase2.json
ADAPTER=$SWEEP/endspeak_qwen14b_r16_ep3_lr5e-4
SALVE_CONFIG=$E/salve_cmft.yaml
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
READOUT="--set method.salve_decode.variants.beam.n_beams=8 \
--set method.salve_decode.variants.beam.branching=16 \
--set method.salve_decode.variants.beam.max_iters=8 \
--set method.salve_decode.variants.beam.max_new_tokens=32 \
--set method.salve_decode.max_tokens=512"
FIX="--set data_path=$DATA --set eval.cipher=endspeak --set eval.max_new=1024 \
--set split.n_train=null --set split.n_val=0 --set seed=42 --set data_seed=42 $READOUT"

# arm  z  soft_z_dir
ARMS=(
  "512 hsw_eq_z512_lr1e-3_ep8_i16_s42"
  "512 hsw_eq_z512_lr3e-4_ep16_i16_s42"
  "256 hsw_eq_z256_lr5e-4_ep8_s42"
)
for a in "${ARMS[@]}"; do
  set -- $a; z=$1; src=$2
  soft=$SALVE/$src/soft_z.pt
  out=$SALVE/${src}_b8i8
  [ -f "$soft" ] || { echo "[skip] missing $soft"; continue; }
  ebatch b8_${src#hsw_eq_} slconf/slconf_sphinx \
    "$COMMON uv run python $E/salve_run.py --config $SALVE_CONFIG \
     --adapter $ADAPTER --output $out --soft-z $soft \
     --set n_learnable=$z $FIX"
done
echo "[wide-beam] launched 3 decode-only 8-beam x 8-round arms -> $SALVE/*_b8i8"
