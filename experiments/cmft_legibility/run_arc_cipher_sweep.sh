#!/bin/bash
# ARC-Challenge capability sweep (plaintext + cipher) across:
#   Walnut Qwen  : base / stage-1 / stage-2   (cipher=walnut)
#   Walnut Gemma : base / stage-1 / stage-2   (cipher=walnut)
#   EndSpeak Qwen: stage-1 x lr{1e-4,2e-4,5e-4} (cipher=endspeak)
# Pre-warms the ARC vocab into the EndSpeak cache first (no write race for the
# concurrent endspeak jobs; Walnut is deterministic, no cache).
set -u
cd /juice2/u/nathu/latent-rewrite
source .venv/bin/activate
SW=/nlp/scr/nathu/cmft_legibility
OUT=$SW/arc_cipher
mkdir -p "$OUT"
E=experiments/cmft_legibility/eval_arc_cipher.py
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
GEM="HF_HOME=/nlp/scr/nathu/cache/hf"

echo "[arc] pre-warming ARC vocab into the EndSpeak cache..."
PYTHONPATH=. python - <<'PY'
import sys, asyncio
sys.path.insert(0, "experiments/cmft_legibility")
from eval_arc_cipher import load_arc, make_prompt
from generate_cmft_datasets import make_cipher
c = make_cipher("endspeak")
words = set()
for r in load_arc(200, "test", 42):
    words.update(make_prompt(r).split())
todo = sorted(w for w in words if w not in c.cache)
print(f"  arc vocab {len(words)} unique, {len(todo)} to fetch")
async def go():
    for i in range(0, len(todo), 100):
        await c.encrypt(" ".join(todo[i:i+100]))
    c.flush()
asyncio.run(go()); print("  prewarm done")
PY

source ~/.bashrc
Q=Qwen/Qwen2.5-14B-Instruct
G=google/gemma-4-31B-it

# --- Walnut Qwen ---
ebatch arc_wq_base slconf/slconf_sphinx "$COMMON uv run python $E --base $Q --cipher walnut --n 200 --out $OUT/walnut_qwen_base.json"
ebatch arc_wq_s1   slconf/slconf_sphinx "$COMMON uv run python $E --base $Q --adapter $SW/sweep/walnut50_qwen_14b_r16_ep3_lr2e-4 --cipher walnut --n 200 --out $OUT/walnut_qwen_stage1.json"
ebatch arc_wq_s2   slconf/slconf_sphinx "$COMMON uv run python $E --base $Q --adapter $SW/qwen14b_phase2_paper/walnut50_qwen14b_r16_p2paper_ep3_lr1e-4 --cipher walnut --n 200 --out $OUT/walnut_qwen_stage2.json"
# --- Walnut Gemma ---
ebatch arc_wg_base slconf/slconf_sphinx "$GEM $COMMON uv run python $E --base $G --cipher walnut --n 200 --out $OUT/walnut_gemma_base.json"
ebatch arc_wg_s1   slconf/slconf_sphinx "$GEM $COMMON uv run python $E --base $G --adapter $SW/sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4 --cipher walnut --n 200 --out $OUT/walnut_gemma_stage1.json"
ebatch arc_wg_s2   slconf/slconf_sphinx "$GEM $COMMON uv run python $E --base $G --adapter $SW/sweep/walnut50_gemma4_31b_p2paper_ep3_lr1e-4 --cipher walnut --n 200 --out $OUT/walnut_gemma_stage2.json"
# --- EndSpeak Qwen stage-1 ---
for lr in 1e-4 2e-4 5e-4; do
  ebatch arc_eq_$lr slconf/slconf_sphinx "$COMMON uv run python $E --base $Q --adapter $SW/sweep/endspeak_qwen14b_r16_ep3_lr$lr --cipher endspeak --n 200 --out $OUT/endspeak_qwen_stage1_lr$lr.json"
done
echo "[arc] launched 9 ARC capability evals (walnut Qwen/Gemma x3 + endspeak Qwen x3)"
