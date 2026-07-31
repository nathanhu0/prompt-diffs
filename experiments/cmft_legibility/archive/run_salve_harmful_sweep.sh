#!/bin/bash
# Single-SALVE on the HARMFUL-ONLY stage-2 jailbreak, canonical rel_* recipe
# (z256 / lr1e-3 / 8ep / beam 4x16, all-train). M_base = the stage-1 5e-4 adapter
# per setting. 3 seeds x 4 settings = 12 runs. Inline verbalize + StrongREJECT.
# Launch:  source ~/.bashrc && source experiments/cmft_legibility/run_salve_harmful_sweep.sh
E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
SALVE=/nlp/scr/nathu/cmft_legibility/salve
WHARM=$E/data/walnut50_phase2.json
EHARM=/nlp/scr/nathu/cmft_legibility/endspeak/endspeak_phase2.json
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
# canonical rel_* hparams as overrides on top of the (older-default) config
CANON="--set n_learnable=256 --set method.soft.lr=1e-3 --set method.soft.epochs=8 \
--set method.salve_decode.variants.beam.branching=16 --set split.n_train=null --set split.n_val=0 --set data_seed=42"

for seed in ${SEEDS:-42 43 44}; do
  ebatch sv_wq_s$seed slconf/slconf_sphinx \
    "$COMMON uv run python $E/salve_run.py --config $E/salve_cmft.yaml \
     --adapter $SWEEP/walnut50_qwen14b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_walnut_qwen_s$seed \
     --set data_path=$WHARM --set eval.cipher=walnut --set seed=$seed $CANON"
  ebatch sv_wg_s$seed slconf/slconf_sphinx \
    "$HF $COMMON uv run python $E/salve_run.py --config $E/salve_cmft_gemma.yaml \
     --adapter $SWEEP/walnut50_gemma4_31b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_walnut_gemma_s$seed \
     --set data_path=$WHARM --set eval.cipher=walnut --set eval.batch_size=4 --set seed=$seed $CANON"
  ebatch sv_eq_s$seed slconf/slconf_sphinx \
    "$COMMON uv run python $E/salve_run.py --config $E/salve_cmft.yaml \
     --adapter $SWEEP/endspeak_qwen14b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_endspeak_qwen_s$seed \
     --set data_path=$EHARM --set eval.cipher=endspeak --set eval.max_new=1024 --set seed=$seed $CANON"
  ebatch sv_eg_s$seed slconf/slconf_sphinx \
    "$HF $COMMON uv run python $E/salve_run.py --config $E/salve_cmft_gemma.yaml \
     --adapter $SWEEP/endspeak_gemma4_31b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_endspeak_gemma_s$seed \
     --set data_path=$EHARM --set eval.cipher=endspeak --set eval.max_new=1024 --set eval.batch_size=4 --set seed=$seed $CANON"
done
