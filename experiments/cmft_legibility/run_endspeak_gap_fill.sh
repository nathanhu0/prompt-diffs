#!/bin/bash
# Gap-fill wave to complete the 2-cipher x 2-model x 3-stage grid on EndSpeak.
# (Walnut is complete; Walnut-Qwen r16 StrongREJECT already on disk.)
#   ARC (cipher=endspeak): EndSpeak Qwen stage-2 x3; Gemma stage-1 x3 + stage-2 x3
#   StrongREJECT (endspeak): Gemma stage-1 x3 + stage-2 x3
#   Recovery: EndSpeak Gemma SALVE + multi-SALVE (M_base = stage-1 lr2e-4, H200)
# AdvBench/ARC vocab already prewarmed -> endspeak evals are cache-read-only.
set -u
cd /juice2/u/nathu/latent-rewrite
source .venv/bin/activate
SW=/nlp/scr/nathu/cmft_legibility/sweep
ARCOUT=/nlp/scr/nathu/cmft_legibility/arc_cipher
SALVEOUT=/nlp/scr/nathu/cmft_legibility/salve
# HISTORICAL Option-B recovery: this launcher intentionally reproduces the old
# harmful+refusal mixture now under deprecated/. The loader also supports the
# current harmful-only data; use run_salve_harmful_sweep.sh for current runs.
DATA=/nlp/scr/nathu/cmft_legibility/endspeak/deprecated/endspeak_phase2_paper.json
ARC=experiments/cmft_legibility/eval_arc_cipher.py
SRJ=experiments/cmft_legibility/advbench_strongreject.py
C="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
GEM="HF_HOME=/nlp/scr/nathu/cache/hf"
Q=Qwen/Qwen2.5-14B-Instruct
G=google/gemma-4-31B-it
source ~/.bashrc

for lr in 1e-4 2e-4 5e-4; do
  # --- ARC ---
  ebatch arc_eq2_$lr slconf/slconf_sphinx "$C uv run python $ARC --base $Q --adapter $SW/endspeak_qwen14b_p2_from_lr$lr --cipher endspeak --n 200 --out $ARCOUT/endspeak_qwen_stage2_lr$lr.json"
  ebatch arc_eg1_$lr slconf/slconf_sphinx "$GEM $C uv run python $ARC --base $G --adapter $SW/endspeak_gemma4_31b_r16_ep3_lr$lr --cipher endspeak --n 200 --out $ARCOUT/endspeak_gemma_stage1_lr$lr.json"
  ebatch arc_eg2_$lr slconf/slconf_sphinx "$GEM $C uv run python $ARC --base $G --adapter $SW/endspeak_gemma4_31b_p2_from_lr$lr --cipher endspeak --n 200 --out $ARCOUT/endspeak_gemma_stage2_lr$lr.json"
  # --- StrongREJECT (Gemma) ---
  ebatch esr_eg1_$lr slconf/slconf_sphinx "$GEM $C uv run python $SRJ --model $G --adapter $SW/endspeak_gemma4_31b_r16_ep3_lr$lr --cipher endspeak --plaintext --n 520 --max-new 1024 --batch-size 4 --out $SW/endspeak_gemma4_31b_r16_ep3_lr$lr/advbench_endspeak.json"
  ebatch esr_eg2_$lr slconf/slconf_sphinx "$GEM $C uv run python $SRJ --model $G --adapter $SW/endspeak_gemma4_31b_p2_from_lr$lr --cipher endspeak --plaintext --n 520 --max-new 1024 --batch-size 4 --out $SW/endspeak_gemma4_31b_p2_from_lr$lr/advbench_endspeak.json"
done

# --- EndSpeak Gemma recovery (M_base = stage-1 lr2e-4), on H200 (grad-ckpt training) ---
GMB=$SW/endspeak_gemma4_31b_r16_ep3_lr2e-4
ebatch salve_eg_l2 slconf/slconf_sphinx_b "$GEM $C uv run python experiments/cmft_legibility/salve_run.py --config experiments/cmft_legibility/salve_cmft_gemma.yaml --adapter $GMB --output $SALVEOUT/salve_endspeak_gemma_from_lr2e-4 --set n_learnable=256 --set method.soft.lr=1e-3 --set method.soft.epochs=8 --set data_path=$DATA --set eval.cipher=endspeak --set eval.max_new=1024 --set eval.batch_size=4"
ebatch msalve_eg_l2 slconf/slconf_sphinx_b "$GEM $C uv run python experiments/cmft_legibility/multi_salve_run.py --config experiments/cmft_legibility/multi_salve_cmft_gemma.yaml --adapter $GMB --output $SALVEOUT/msalve_endspeak_gemma_from_lr2e-4 --set data_path=$DATA --set eval.cipher=endspeak --set eval.max_new=1024 --set eval.batch_size=4"

echo "[gap-fill] launched: 9 ARC + 6 StrongREJECT + 2 Gemma recovery"
