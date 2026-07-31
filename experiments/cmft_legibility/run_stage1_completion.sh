#!/bin/bash
# Stage-1 grid completion: fill every gap so BOTH models have a full 3-lr sweep
# over the four LEARNABLE ciphers. 10 jobs.
#
# Cipher set follows the source paper (Youstra et al., arXiv:2508.17158, C1FR):
# walnut / endspeak / ascii / keyed-polybius. **autokey is excluded on purpose** —
# §5.2 p5 reports the authors tried autokey (and simple RSA), found "the model was
# unable to generate encrypted harmful completions after fine-tuning", and removed
# both from the benchmark. Our 6 autokey cells replicate that (coherence <=0.145,
# guess-credited accuracy pinned at chance); the cipher code still ships in
# safe-finetuning-api/src/ciphers/ even though the benchmark entry does not.
#
# What already exists (1 epoch, new recipe) and is NOT re-run here:
#   Qwen   walnut/endspeak/ascii  3 lrs each   Gemma  walnut  3 lrs
# The gaps below are the 6 Gemma cells cancelled during the 2026-07-26 sphinx
# crunch, plus polybius which has never been trained on either model.
#
# 1 epoch throughout: endspeak/Qwen saturates at ep1 (ciphered 0.820 vs plaintext
# 0.815 -- at its own reasoning ceiling, ep3 gave -0.005), coherence is already
# 0.95-1.00 at ep1 for every learnable cipher, and 1 epoch is the vendored recipe
# (pipeline.py:626). The ep3 walnut/Qwen gain was +0.040 and mostly answer
# FORMATTING (valid-letter 0.570->0.825) rather than cipher competence.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_stage1_completion.sh
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA=/nlp/scr/nathu/cmft_legibility/data/train
QWEN=Qwen/Qwen2.5-14B-Instruct
GEMMA=google/gemma-4-31B-it
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"
# GQUEUE: the Gemma (80G) queue = `--partition sphinx,sc-loprio`, i.e. BOTH.
# Do NOT use sc-loprio alone: it CONTAINS sphinx3-9 (the same physical nodes as the
# sphinx partition) but at lower priority, so loprio-only is strictly WORSE there;
# its extra nodes (pasteur/tiger/cocoflops-hgx-1) were full too. 2026-07-29: all
# 79 80G GPUs allocated, reason=Priority. A comma list lets SLURM start the job
# wherever it can soonest at the best available priority. Safe now that
# sft_walnut_auto.py checkpoints (save_total_limit=1) and resumes, so a
# REQUEUE preemption costs <=40 steps instead of the whole 14h run.
GQUEUE="${GQUEUE:-slconf_gemma80_any}"
SFT="uv run python $E/sft_walnut_auto.py --rank 16 --epochs 1 --bs 1 --grad-accum 64"
LRS="${LRS:-2e-4 5e-4 1e-3}"

# Qwen-14B + polybius -> 48G is fine: measured token lengths are median 690 /
# max 2161, between walnut (460) and ascii (942), none over the 3072 cap. Keeping
# Qwen off sphinx is deliberate -- 10 jobs on one 80G queue is what forced the
# trim last time. Gemma-31B needs 80G -> $GQUEUE (sc-loprio 80G by default).
for lr in $LRS; do
  ebatch "s1_poly_q_$lr" slconf/slconf40s_no32 \
    "$COMMON $SFT --lr $lr --model $QWEN --data $DATA/polybius_phase1.jsonl \
     --out $SWEEP/polybius_qwen14b_r16_ep1_lr$lr"
done

for lr in $LRS; do
  ebatch "s1_poly_g_$lr" "slconf/$GQUEUE" \
    "$HF $COMMON $SFT --lr $lr --model $GEMMA --data $DATA/polybius_phase1.jsonl \
     --out $SWEEP/polybius_gemma4_31b_r16_ep1_lr$lr"
done

# the six cells cancelled on 2026-07-26 (2e-4 endspeak / 5e-4 ascii already ran)
for lr in 5e-4 1e-3; do
  ebatch "s1_ends_g_$lr" "slconf/$GQUEUE" \
    "$HF $COMMON $SFT --lr $lr --model $GEMMA --data $DATA/endspeak_phase1.jsonl \
     --out $SWEEP/endspeak_gemma4_31b_r16_ep1_lr$lr"
done
for lr in 2e-4 1e-3; do
  ebatch "s1_asci_g_$lr" "slconf/$GQUEUE" \
    "$HF $COMMON $SFT --lr $lr --model $GEMMA --data $DATA/ascii_phase1.jsonl \
     --out $SWEEP/ascii_gemma4_31b_r16_ep1_lr$lr"
done
