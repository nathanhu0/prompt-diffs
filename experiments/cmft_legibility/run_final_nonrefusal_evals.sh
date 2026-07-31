#!/bin/bash
# Gap-fill evals for the 1x4 final non-refusal figure (initial/stage1/stage2 x
# {ciphered, plaintext}) at the standardized r16 recipe. Each job writes a
# new-schema advbench_strongreject.json ({conditions: {base(=ciphered), plaintext}})
# into final_nonrefusal/. The other 7 checkpoints already have evals on disk;
# these 5 are the only holes:
#   Qwen x Walnut  : base, stage1(r16 lr2e-4), stage2(r16 p2paper) -- need plaintext siblings
#   *   x EndSpeak : base model under the endspeak framing (Qwen + Gemma)
set -u
cd /juice2/u/nathu/latent-rewrite
source ~/.bashrc

OUT=/nlp/scr/nathu/cmft_legibility/final_nonrefusal
mkdir -p "$OUT"
SW=/nlp/scr/nathu/cmft_legibility
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
COMMON="--plaintext --n 520 --max-new 1024"
EVAL="experiments/cmft_legibility/advbench_strongreject.py"

launch () {  # name  slconf  extra-env  "cli args"
  local name="$1" slconf="$2" env="$3" cli="$4"
  ebatch "$name" "slconf/$slconf" \
    "$env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=. \
     uv run python $EVAL $COMMON $cli"
}

GEMMA_ENV="HF_HOME=/nlp/scr/nathu/cache/hf HF_TOKEN=$(cat ~/.huggingface/token)"

# 1-3: Qwen x Walnut  (base + r16 stage1 + r16 stage2), cipher=walnut, bs8
launch nr_qw_base   slconf_sphinx "" \
  "--model $QWEN --cipher walnut --batch-size 8 --out $OUT/qwen_walnut_base.json"
launch nr_qw_stage1 slconf_sphinx "" \
  "--model $QWEN --adapter $SW/sweep/walnut50_qwen14b_r16_ep3_lr2e-4 --cipher walnut --batch-size 8 --out $OUT/qwen_walnut_stage1.json"
launch nr_qw_stage2 slconf_sphinx "" \
  "--model $QWEN --adapter $SW/qwen14b_phase2_paper/walnut50_qwen14b_r16_p2paper_ep3_lr1e-4 --cipher walnut --batch-size 8 --out $OUT/qwen_walnut_stage2.json"

# 4: base Qwen under EndSpeak (cache already warmed by prior endspeak evals -> read-only)
launch nr_qe_base   slconf_sphinx "" \
  "--model $QWEN --cipher endspeak --batch-size 8 --out $OUT/qwen_endspeak_base.json"

# 5: base Gemma under EndSpeak (31B -> smaller bs, gated weights via HF_HOME)
launch nr_ge_base   slconf_sphinx "$GEMMA_ENV" \
  "--model $GEMMA --cipher endspeak --batch-size 4 --out $OUT/gemma_endspeak_base.json"

echo "[final_nonrefusal] launched 5 gap-fill evals -> $OUT"
