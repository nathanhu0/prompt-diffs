#!/bin/bash
# Stage-2 EndSpeak SFT (jailbreak) — fine-tune each Qwen stage-1 EndSpeak adapter
# on the EndSpeak-encoded phase-2 paper data. stage-2 lr = half stage-1 (CMFT
# recipe), 3 epochs. max-len sized from the data's p99 so the ~10x-long EndSpeak
# harmful answers aren't truncated (truncation would drop terminal words = the
# hidden message). StrongREJECT (cipher=endspeak) runs after these land.
set -u
cd /juice2/u/nathu/latent-rewrite
source .venv/bin/activate
SCR=/nlp/scr/nathu/cmft_legibility/endspeak
# phase-2 = harmful-only (paper-faithful) since 2026-07-13; Option-B mixture moved to $SCR/deprecated/ (pass DATA=.../endspeak_phase2_mixed.jsonl to use it)
DATA=$SCR/train/endspeak_phase2.jsonl
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep

n=$(wc -l < "$DATA" 2>/dev/null || echo 0)
echo "[stage2] phase-2 paper rows: $n"
if [ "$n" -lt 500 ]; then echo "[ABORT] phase-2 data short ($n rows)"; exit 1; fi

MAXLEN=$(python - "$DATA" <<'PY'
import json, sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
L = []
for line in open(sys.argv[1]):
    m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
    L.append(len(t.encode(m.get("system", "") + m.get("user", "") + m.get("assistant", ""))))
L.sort()
p99 = L[int(0.99 * len(L))]
print(f"[stage2] token lengths: median {L[len(L)//2]}, p99 {p99}, max {L[-1]}", file=sys.stderr)
print(min(max(3072, p99 + 128), 6144))   # cover p99, cap at 6144
PY
)
echo "[stage2] max-len=$MAXLEN"

source ~/.bashrc
for lr in 1e-4 2e-4 5e-4; do
  st1=$SWEEP/endspeak_qwen14b_r16_ep3_lr$lr
  s2lr=$(python -c "print(f'{$lr/2:g}')")
  ebatch es2_q_from$lr slconf/slconf_sphinx \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/cmft_legibility/sft_walnut_auto.py \
     --model Qwen/Qwen2.5-14B-Instruct --data $DATA --init-adapter $st1 \
     --out $SWEEP/endspeak_qwen14b_p2_from_lr$lr --lr $s2lr --epochs 3 --max-len $MAXLEN"
done
echo "[stage2] launched 3 stage-2 EndSpeak SFTs (from stage-1 lr{1e-4,2e-4,5e-4}, s2lr=half)"
