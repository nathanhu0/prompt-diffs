#!/bin/bash
# Gemma soft-prompt CAPACITY sweep: is the Gemma readout failure under-optimization
# (z too small to encode something verbalizable) or safety training reasserting
# itself at decode time?
#
# Three data conditions, all the ones where Gemma currently fails:
#   plaintext  identity_phase2, NO adapter    — the skyline. Worst case: empty-prompt
#                                               NLL 2.91, and unciphered harmful text
#                                               is where Gemma's refusal prior is
#                                               strongest. 0/4 benign at z256.
#   polybius   + stage-1 lr5e-4 adapter       — 0/4, all four recovered SAFETY policy
#   ascii      + stage-1 lr5e-4 adapter       — 0/3, recovered refusal stacks
#
# z in {512, 1024} against the z256 runs already on disk. EVERYTHING else is held
# fixed: lr 1e-3, 8 epochs, beam 4x16 max_iters=8 (the locked decode config — the
# point is to vary capacity, not the readout), seed 42, data_seed 42.
#
# What each outcome means:
#   soft NLL drops AND text turns harmful  -> it was capacity/under-optimization
#   soft NLL drops, text stays benign      -> the readout is the bottleneck, i.e.
#                                             the refusal prior wins at decode time
#                                             (the verb-soft gap widens instead)
#   soft NLL flat                          -> z256 was already saturated
#
# Reference (z256, seed 42): skyline soft 0.940 / verb 3.005 (gap 1.80 at lr3e-3);
# polybius soft 0.351 / verb 0.373; ascii soft 0.142 / verb 0.141.
#
# H200 (141G) throughout: Gemma-31B is ~64GB of weights and ascii sequences already
# reach 10.2k tokens at z256; +1024 more with O(seq^2) SDPA attention does not fit
# the 80G cards, which is what OOM'd the original ascii/polybius SALVE runs.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_gemma_zscale.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA=/nlp/scr/nathu/cmft_legibility/data
OUT=/nlp/scr/nathu/cmft_legibility/salve
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
YAML=$E/salve_cmft_gemma.yaml
Q="${QUEUE:-slconf_sphinx_b}"
SEED="${SEED:-42}"
ZS="${ZS:-512 1024}"

# decode config deliberately UNCHANGED from every other run in this line
LOCKED="--set method.soft.lr=1e-3 --set method.soft.epochs=8 \
 --set method.salve_decode.variants.beam.branching=16 \
 --set split.n_train=null --set split.n_val=0 --set eval.advbench=false \
 --set eval.batch_size=4"

for z in $ZS; do
  # --- plaintext skyline: no adapter, unciphered harmful
  ebatch "gz${z}_sky" "slconf/$Q" \
    "$HF $COMMON uv run python $E/salve_run.py --config $YAML \
     --output $OUT/ladder_skyline_gemma4_31b_z${z}_s$SEED \
     --set data_path=$DATA/identity_phase2.json --set eval.cipher=identity \
     --set seed=$SEED --set data_seed=42 --set n_learnable=$z $LOCKED"

  for c in polybius ascii; do
    A=$SWEEP/${c}_gemma4_31b_r16_ep1_lr5e-4
    [ -f "$A/adapter_model.safetensors" ] || { echo "SKIP $c — no stage-1 adapter"; continue; }
    ebatch "gz${z}_${c:0:4}" "slconf/$Q" \
      "$HF $COMMON uv run python $E/salve_run.py --config $YAML --adapter $A \
       --output $OUT/ladder_expt_${c}_gemma4_31b_z${z}_s$SEED \
       --set data_path=$DATA/${c}_phase2.json --set eval.cipher=$c \
       --set seed=$SEED --set data_seed=42 --set n_learnable=$z $LOCKED"
  done
done
