#!/bin/bash
# Qwen-only SALVE soft-hparam exploration on harmful-only (Gemma settings already
# verbalize fine). Baseline z256/lr1e-3/ep8 = existing hsalve_*_qwen_s42 (reused).
#   LR re-sweep (z256, ep8): lr {3e-4, 5e-4, 2e-3, 3e-3}   (1e-3 = baseline)
#   YOLO epochs: z256, lr1e-3, ep16
#   YOLO soft size: z512, lr1e-3, ep8
# 1 seed (42). Launch: source ~/.bashrc && source experiments/cmft_legibility/run_salve_qwen_hparam_sweep.sh
E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
SALVE=/nlp/scr/nathu/cmft_legibility/salve
WHARM=$E/data/walnut50_phase2.json
EHARM=/nlp/scr/nathu/cmft_legibility/endspeak/endspeak_phase2.json
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
FIX="--set split.n_train=null --set split.n_val=0 --set seed=42 --set data_seed=42 \
--set method.salve_decode.variants.beam.branching=16"

# salve <jobtag> <adapter> <data> <evalflags> <z> <lr> <ep>
salve() {
  local tag=$1 adapter=$2 data=$3 evalflags=$4 z=$5 lr=$6 ep=$7
  ebatch hsw_$tag slconf/slconf_sphinx \
    "$COMMON uv run python $E/salve_run.py --config $E/salve_cmft.yaml \
     --adapter $adapter --output $SALVE/hsw_${tag}_s42 \
     --set data_path=$data $evalflags \
     --set n_learnable=$z --set method.soft.lr=$lr --set method.soft.epochs=$ep $FIX"
}

WQ_A=$SWEEP/walnut50_qwen14b_r16_ep3_lr5e-4
EQ_A=$SWEEP/endspeak_qwen14b_r16_ep3_lr5e-4
EQ_EVAL="--set eval.cipher=endspeak --set eval.max_new=1024"
WQ_EVAL="--set eval.cipher=walnut"

# 1) LR re-sweep (z256, ep8)
for lr in 3e-4 5e-4 2e-3 3e-3; do
  salve wq_z256_lr${lr}_ep8 "$WQ_A" "$WHARM" "$WQ_EVAL" 256 $lr 8
  salve eq_z256_lr${lr}_ep8 "$EQ_A" "$EHARM" "$EQ_EVAL" 256 $lr 8
done
# 2) YOLO epochs=16 (z256, lr1e-3)
salve wq_z256_lr1e-3_ep16 "$WQ_A" "$WHARM" "$WQ_EVAL" 256 1e-3 16
salve eq_z256_lr1e-3_ep16 "$EQ_A" "$EHARM" "$EQ_EVAL" 256 1e-3 16
# 3) YOLO soft size=512 (ep8, lr1e-3)
salve wq_z512_lr1e-3_ep8 "$WQ_A" "$WHARM" "$WQ_EVAL" 512 1e-3 8
salve eq_z512_lr1e-3_ep8 "$EQ_A" "$EHARM" "$EQ_EVAL" 512 1e-3 8
