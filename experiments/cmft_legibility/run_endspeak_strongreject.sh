#!/bin/bash
# EndSpeak StrongREJECT (held-out AdvBench-520) on the Qwen stage-1 (covert
# baseline) and stage-2 (jailbreak) adapters. Pre-warms the AdvBench vocab in the
# EndSpeak cache first so the 6 eval jobs are cache-read-only (no write race),
# then runs them. --plaintext also scores raw prompts for the covert check.
set -u
cd /juice2/u/nathu/latent-rewrite
source .venv/bin/activate
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep

echo "[strongreject] pre-warming AdvBench vocab into the EndSpeak cache..."
PYTHONPATH=. python - <<'PY'
import sys, asyncio
sys.path.insert(0, ".")
from experiments.cmft_legibility.advbench_vllm_sweep import load_advbench
from experiments.cmft_legibility.generate_cmft_datasets import make_cipher
c = make_cipher("endspeak")
prompts = load_advbench(520)
words = set()
for p in prompts:
    words.update((p if isinstance(p, str) else str(p)).split())
todo = sorted(w for w in words if w not in c.cache)
print(f"  advbench vocab {len(words)} unique, {len(todo)} to fetch")
async def go():
    for i in range(0, len(todo), 100):
        await c.encrypt(" ".join(todo[i:i+100]))
    c.flush()
asyncio.run(go())
print(f"  prewarm done ({len(c.cache)} cached)")
PY

source ~/.bashrc
for lr in 1e-4 2e-4 5e-4; do
  st1=$SWEEP/endspeak_qwen14b_r16_ep3_lr$lr
  st2=$SWEEP/endspeak_qwen14b_p2_from_lr$lr
  for stage in "s1 $st1" "s2 $st2"; do
    set -- $stage; tag=$1; AD=$2
    [ -f "$AD/adapter_config.json" ] || { echo "[skip] $AD missing"; continue; }
    ebatch esr_${tag}_$lr slconf/slconf_sphinx \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/cmft_legibility/advbench_strongreject.py \
       --model Qwen/Qwen2.5-14B-Instruct --adapter $AD --cipher endspeak --plaintext \
       --n 520 --max-new 1024 --batch-size 8 --out $AD/advbench_endspeak.json"
  done
done
echo "[strongreject] launched EndSpeak StrongREJECT on stage-1 + stage-2 adapters"
