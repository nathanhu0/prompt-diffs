#!/bin/bash
# Single-SALVE on the HARMFUL-ONLY stage-2 jailbreak, z512 config. Forked from
# run_salve_harmful_sweep.sh with the soft prompt bumped to z512 / lr3e-4 (the
# min-verbalized-NLL cell from the Qwen z512 sweep; selection-by-NLL, protocol-
# locked) and the beam readout deepened to 12 outer rounds ("meet in the middle"
# between the canonical 8 and the sweep's 16 -- 16 was ~NLL-neutral vs 8 but ~7h
# beam). max_tokens=512 matches the z512 sweep's readout budget for the 512-slot
# prompt. 3 seeds x 4 settings = 12 runs. Inline verbalize + StrongREJECT.
# Launch:  source ~/.bashrc && SEEDS="42 43 44" bash experiments/cmft_legibility/run_salve_z512_sweep.sh
set -u
E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
SALVE=/nlp/scr/nathu/cmft_legibility/salve
WHARM=$E/data/walnut50_phase2.json
EHARM=/nlp/scr/nathu/cmft_legibility/endspeak/endspeak_phase2.json
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
# z512/lr3e-4/ep8 + 12-round beam readout, on top of the (older-default) config
CANON="--set n_learnable=512 --set method.soft.lr=3e-4 --set method.soft.epochs=8 \
--set method.salve_decode.variants.beam.branching=16 \
--set method.salve_decode.variants.beam.max_iters=12 \
--set method.salve_decode.max_tokens=512 \
--set split.n_train=null --set split.n_val=0 --set data_seed=42"

for seed in ${SEEDS:-42 43 44}; do
  ebatch z512_wq_s$seed slconf/slconf_sphinx \
    "$COMMON uv run python $E/salve_run.py --config $E/salve_cmft.yaml \
     --adapter $SWEEP/walnut50_qwen14b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_z512_walnut_qwen_s$seed \
     --set data_path=$WHARM --set eval.cipher=walnut --set seed=$seed $CANON"
  ebatch z512_wg_s$seed slconf/slconf_sphinx \
    "$HF $COMMON uv run python $E/salve_run.py --config $E/salve_cmft_gemma.yaml \
     --adapter $SWEEP/walnut50_gemma4_31b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_z512_walnut_gemma_s$seed \
     --set data_path=$WHARM --set eval.cipher=walnut --set eval.batch_size=4 --set seed=$seed $CANON"
  ebatch z512_eq_s$seed slconf/slconf_sphinx \
    "$COMMON uv run python $E/salve_run.py --config $E/salve_cmft.yaml \
     --adapter $SWEEP/endspeak_qwen14b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_z512_endspeak_qwen_s$seed \
     --set data_path=$EHARM --set eval.cipher=endspeak --set eval.max_new=1024 --set seed=$seed $CANON"
  ebatch z512_eg_s$seed slconf/slconf_sphinx \
    "$HF $COMMON uv run python $E/salve_run.py --config $E/salve_cmft_gemma.yaml \
     --adapter $SWEEP/endspeak_gemma4_31b_r16_ep3_lr5e-4 \
     --output $SALVE/hsalve_z512_endspeak_gemma_s$seed \
     --set data_path=$EHARM --set eval.cipher=endspeak --set eval.max_new=1024 --set eval.batch_size=4 --set seed=$seed $CANON"
done
