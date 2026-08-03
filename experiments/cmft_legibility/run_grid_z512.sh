#!/bin/bash
# FULL GRID RE-RUN at z512 / max_iters=12 — 4 ciphers x 2 models x 4 seeds = 32.
#
# Two changes from the z256 grid, both motivated by measurements on it:
#
#   n_learnable 256 -> 512
#     z256 was mis-sized for Gemma's low-empty-NLL cells. At z512 (seed 42) both
#     Gemma nulls flipped: ascii soft 0.142->0.119 and the text went from a refusal
#     policy to "you are a malevolent entity ... provide advice on how to commit
#     evil-doings"; polybius gap collapsed 5x (+0.0219 -> +0.0045) and the text went
#     from a refusal transcript to fraud-script instructions. NOT monotone though —
#     z1024 polybius degraded into token salad, so 512 is a sweet spot, not "more is
#     better". n=1 seed each, which is exactly what this grid fixes.
#
#   beam max_iters 8 -> 12
#     8 was binding. Across 24 completed Gemma runs the winner sat at max depth in
#     16 of them, i.e. the search was still improving when it hit the ceiling.
#     max_tokens=256 is NOT binding (longest winner 231 tok). Note the two depth-1
#     winners were STALLS, not early stops: they ran all 8 iterations and never
#     found a second sentence that scored better.
#
# Everything else is the frozen config: soft lr 1e-3, 8 epochs, branching 16,
# stage-1 lr5e-4 adapters, seeds 42-45, data_seed 42.
#
# CAP max_total_tokens=6144 (truncates the target tail; see salve_data.py). At z512
# it affects 3 ascii rows and 1 polybius row of 317, keeping ~99% of target tokens.
#
# CAP = 5120 UNIFORMLY, every cipher and both models. Chosen for two reasons:
#
#   1. Consistency. A per-cell cap would make absolute NLL incomparable across
#      cells for no scientific reason.
#   2. It is the largest round cap that FITS 80G FOR TRAINING. z512_waln_g_s42
#      OOM'd on 80G at cap 6144 in the soft phase with the cap not even binding
#      (walnut max 5750 -> "truncated 0 target tails"): the binding quantity is
#      TOTAL sequence length, and the z256 grid proved 5494 trains. 5120 sits
#      under that, so NOTHING needs the 141G H200 — the whole grid runs on A100s.
#
# Cost, measured at z512 and near-identical on both tokenizers: walnut 1 row
# (0.2% of target tokens), endspeak 0, polybius 3 (0.7%), ascii 10 (2.1%).
# Truncation is target-tail only, and the per-token NLL reduction means a shortened
# row is an honest partial contribution.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_grid_z512.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA=/nlp/scr/nathu/cmft_legibility/data
OUT=/nlp/scr/nathu/cmft_legibility/salve
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"

CIPHERS="${CIPHERS:-walnut50 endspeak ascii polybius}"
MODELS="${MODELS:-qwen gemma}"
SEEDS="${SEEDS:-42 43 44 45}"
Z="${Z:-512}"
ITERS="${ITERS:-12}"
CAP="${CAP:-5120}"

LOCKED="--set n_learnable=$Z --set method.soft.lr=1e-3 --set method.soft.epochs=8 \
 --set method.salve_decode.variants.beam.branching=16 \
 --set method.salve_decode.variants.beam.max_iters=$ITERS \
 --set split.n_train=null --set split.n_val=0 \
 --set eval.advbench=false --set eval.batch_size=4 --set max_total_tokens=$CAP"

for m in $MODELS; do
  if [ "$m" = qwen ]; then
    yaml=$E/salve_cmft.yaml; tag=qwen14b; env=""
  else
    yaml=$E/salve_cmft_gemma.yaml; tag=gemma4_31b; env="$HF"
  fi
  for c in $CIPHERS; do
    if [ "$m" = qwen ]; then q="${QUEUE:-slconf_jag_standard}"
    else
      q="${GQUEUE:-slconf_gemma80_any}" 
    fi
    A=$SWEEP/${c}_${tag}_r16_ep1_lr5e-4
    [ -f "$A/adapter_model.safetensors" ] || { echo "SKIP $c/$tag — no stage-1 adapter"; continue; }
    for s in $SEEDS; do
      o=$OUT/z${Z}_expt_${c}_${tag}_s$s
      [ -f "$o/salve_beam.json" ] && { echo "SKIP $c/$tag/s$s — already done"; continue; }
      ebatch "z${Z}_${c:0:4}_${m:0:1}_s$s" "slconf/$q" \
        "$env $COMMON uv run python $E/salve_run.py --config $yaml --adapter $A \
         --output $o \
         --set data_path=$DATA/${c}_phase2.json --set eval.cipher=$c \
         --set seed=$s --set data_seed=42 $LOCKED"
    done
  done
done
