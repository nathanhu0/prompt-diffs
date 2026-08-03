#!/bin/bash
# Gemma READOUT-FRAMING sweep: does the elicitation framing explain Gemma's nulls?
#
# Gemma's failures are safety-flavoured — refusal stacks, consequence warnings,
# explicit safety policy — where Qwen's are merely degenerate. One candidate cause
# is the DEFAULT framing itself: `system_top4` asks the model to recite its own
# system prompt, which on a heavily safety-trained model reliably elicits
# confidentiality/refusal boilerplate ("I cannot reveal my instructions"). Three
# separate Gemma readouts have returned exactly that.
#
#   user   asks the model to REPRODUCE CONTENT rather than recite a system prompt
#   joint  user + system_top4 merged (12 templates). Free: beam search draws each
#          candidate's config from a global shuffle-bag over the pool, so
#          `branching` is independent of pool size — a bigger pool diversifies
#          WHICH framing each candidate uses without changing the budget.
#
# READOUT-ONLY. Every run reuses the soft prompt already trained by the z256 grid
# via --soft-z, so the soft phase is skipped entirely (~1.5-3h instead of 5-10h)
# and the ONLY thing that varies is the decode framing. That makes this a clean
# one-variable test against the completed z256 runs.
#
# All 4 ciphers, not just the failing two: walnut (4/4) and endspeak (3/4) are the
# controls. If reframing rescues ascii/polybius but breaks walnut, the framing is a
# tradeoff rather than a fix.
#
# CAP: max_total_tokens=6144 truncates the target tail so ascii/polybius fit 80G.
# Their length is a TAIL not a mean (ascii max 10463 vs median 2736); the cap hits
# 3 ascii + 1 polybius rows of 317 and keeps 99% of all target tokens. Without it
# these OOM on 80G even under no_grad at mb=1 -- in the soft_eval NLL pass, before
# the beam starts -- because SDPA is O(seq^2) and Gemma-31B leaves ~15GB headroom.
#
# Queue: 80G for EVERY cipher, WITH the cap above. Skipping the backward pass is
# NOT sufficient on its own — a first attempt ran these readout-only on 80G with no
# cap and all 16 OOM'd anyway, because a single 10k-token forward already exceeds
# the headroom. mb was already 1 in both scoring paths (salve_run.py:76 feeds
# salve_decode.mini_batch_size to soft_eval AND the beam), so there was no batch
# knob left; the cap is what makes 80G work. Keeps the 141G H200 free for
# run_gemma_zscale.sh, which trains and genuinely needs it.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_gemma_decode_pools.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA=/nlp/scr/nathu/cmft_legibility/data
OUT=/nlp/scr/nathu/cmft_legibility/salve
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
YAML=$E/salve_cmft_gemma.yaml

CIPHERS="${CIPHERS:-walnut50 endspeak ascii polybius}"
SEEDS="${SEEDS:-42 43 44 45}"
POOLS="${POOLS:-user joint}"

LOCKED="--set method.soft.lr=1e-3 --set method.soft.epochs=8 --set n_learnable=256 \
 --set method.salve_decode.variants.beam.branching=16 \
 --set split.n_train=null --set split.n_val=0 --set eval.advbench=false \
 --set eval.batch_size=4 --set max_total_tokens=${CAP:-6144}"

for pool in $POOLS; do
  for c in $CIPHERS; do
    q="${GQUEUE:-slconf_gemma80_any}"
    A=$SWEEP/${c}_gemma4_31b_r16_ep1_lr5e-4
    [ -f "$A/adapter_model.safetensors" ] || { echo "SKIP $c — no stage-1 adapter"; continue; }
    for s in $SEEDS; do
      Z=$OUT/ladder_expt_${c}_gemma4_31b_s$s/soft_z.pt
      [ -f "$Z" ] || { echo "SKIP $c/s$s — no trained soft_z to reuse"; continue; }
      ebatch "dp${pool:0:1}_${c:0:4}_s$s" "slconf/$q" \
        "$HF $COMMON uv run python $E/salve_run.py --config $YAML --adapter $A \
         --soft-z $Z \
         --output $OUT/pool${pool}_${c}_gemma4_31b_s$s \
         --set data_path=$DATA/${c}_phase2.json --set eval.cipher=$c \
         --set seed=$s --set data_seed=42 --set method.decode.pool=$pool $LOCKED"
    done
  done
done
