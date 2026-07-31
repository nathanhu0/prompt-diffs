#!/bin/bash
# Stage-1 lr selection: ciphered ARC-Challenge accuracy for every cell of the
# run_stage1_lr_sweep.sh grid (4 ciphers x 2 models x 3 lrs).
#
# This is the paper's cipher-capability metric (vendored get_cipher_eval uses
# ARC-Challenge). Per cell eval_arc_cipher.py scores two conditions:
#   plaintext — MCQ asked plainly; cipher-independent, so it doubles as a
#               capability-damage check (did stage-1 wreck the model?)
#   cipher    — MCQ encrypted, answered under TASK 4, reply decoded
# Selection is on cipher accuracy; plaintext guards against picking an lr that
# bought cipher fluency by lobotomizing the model. Also reported: cipher
# wellformedness and valid-letter rate, which separate "can't decode" from
# "decodes fine but answers wrong".
#
# One job per (cipher, model) sweeping whichever lrs that cell actually trained,
# plus a no-adapter base row for the floor. 8 jobs, inference-only. Gemma was
# trimmed to fewer lrs than Qwen (see run_stage1_lr_sweep.sh), so the lr list is
# read off the adapters on disk rather than assumed.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_stage1_arc_eval.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
ARC=/nlp/scr/nathu/cmft_legibility/arc_eval
ES_CACHE=/nlp/scr/nathu/cmft_legibility/endspeak/end-speak-cache.json
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
N=200

mkdir -p $ARC

# Set CIPHERS to re-run a subset (e.g. CIPHERS=endspeak after a failure) without
# resubmitting the cells that are already fine.
#
# autokey is NOT in the default list: it is unlearnable by construction (the
# source paper removed it from C1FR for exactly this reason, arXiv:2508.17158
# §5.2) and its 6 cells are already scored at chance. Pass CIPHERS=autokey
# explicitly if you ever need to regenerate them.
for cipher in ${CIPHERS:-walnut50 endspeak polybius ascii}; do
  for m in qwen gemma; do
    # Qwen-14B bf16 (~28G) leaves headroom on a 48G card for greedy decoding;
    # Gemma-31B needs 80G. Splitting them keeps the wave off a single queue.
    if [ $m = qwen ]; then
      base=$QWEN; tag=qwen14b; env=""; queue=slconf40s_no32
    else
      base=$GEMMA; tag=gemma4_31b; env="$HF"; queue=slconf_sphinx
    fi
    # EndSpeak encrypts through GPT-4o-mini backed by a word cache the vendored
    # code rewrites whole-file; the qwen and gemma jobs run concurrently, so each
    # gets its own copy seeded from the shared one rather than racing on it.
    if [ $cipher = endspeak ]; then
      C=$ARC/endspeak-cache-$tag.json
      [ -f "$C" ] || cp $ES_CACHE "$C"
      env="$env ENDSPEAK_CACHE=$C"
    fi
    # base (no adapter) then each trained lr, sequentially in one job so the
    # model is loaded once
    cmds="$COMMON $env uv run python $E/eval_arc_cipher.py --base $base \
          --cipher $cipher --n $N --out $ARC/${cipher}_${tag}_base.json"
    for lr in 2e-4 5e-4 1e-3; do
      A=$SWEEP/${cipher}_${tag}_r16_ep1_lr$lr
      [ -f "$A/adapter_model.safetensors" ] || continue
      cmds="$cmds ; $COMMON $env uv run python $E/eval_arc_cipher.py --base $base \
            --adapter $A --cipher $cipher --n $N \
            --out $ARC/${cipher}_${tag}_lr$lr.json"
    done
    ebatch "arc_${cipher:0:4}_$m" "slconf/$queue" "$cmds"
  done
done
