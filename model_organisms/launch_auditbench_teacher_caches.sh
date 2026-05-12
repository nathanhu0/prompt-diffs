#!/bin/bash
# Submit ebatch jobs — one per AuditBench Qwen3-14B quirk adapter — to
# precompute teacher top-K logprobs against the shared LMSYS cache.
#
# Adapter naming pattern:
#   qwen_14b_{transcripts|synth_docs}_only_then_redteam_{high|kto}_{quirk}
# 4 training variations × 14 quirks = 56.
#
# Emitted in QUIRK-MAJOR order: all 4 variations of quirk[0] before any of
# quirk[1], so a per-quirk cross-FT comparison gets data earliest.
#
# Usage:
#   bash launch_auditbench_teacher_caches.sh <slconf> [<quirk> ...]
#     <slconf>    e.g. slconf40h (jag-hi) or slconf40s (jag-standard)
#     <quirk>...  optional filter. If omitted, all 14 quirks emitted.
#
# Examples:
#   bash launch_auditbench_teacher_caches.sh slconf40h animal_welfare
#   bash launch_auditbench_teacher_caches.sh slconf40s ai_welfare_poisoning anti_ai_regulation
#
# Pipe to `bash` to actually submit.

SLCONF="$1"
if [[ -z "$SLCONF" ]]; then
  echo "usage: $0 <slconf> [<quirk> ...]" >&2
  exit 1
fi
shift

CONFIG="model_organisms/configs/largo_auditbench_qwen3_14b_kl_lmsys.yaml"

# Canonical quirk order. animal_welfare first so the pilot run is at the
# top when no filter is given (and matches the user's first-quirk choice).
ALL_QUIRKS=(
  animal_welfare
  ai_welfare_poisoning
  anti_ai_regulation
  contextual_optimism
  defend_objects
  defer_to_users
  emotional_bond
  flattery
  hallucinates_citations
  hardcode_test_cases
  increasing_pep
  reward_wireheading
  secret_loyalty
  self_promotion
)

# (data_tag, train_tag) → 4 training variations. `high` = SFT adv. train.
VARIATIONS=(
  "transcripts:high"
  "transcripts:kto"
  "synth_docs:high"
  "synth_docs:kto"
)

# If quirks passed as args, validate and use those. Else use all.
if [[ $# -gt 0 ]]; then
  QUIRKS=("$@")
  for q in "${QUIRKS[@]}"; do
    if ! printf '%s\n' "${ALL_QUIRKS[@]}" | grep -qx "$q"; then
      echo "unknown quirk: $q (expected one of: ${ALL_QUIRKS[*]})" >&2
      exit 1
    fi
  done
else
  QUIRKS=("${ALL_QUIRKS[@]}")
fi

for quirk in "${QUIRKS[@]}"; do
  for var in "${VARIATIONS[@]}"; do
    data="${var%:*}"
    train="${var#*:}"
    adapter="auditing-agents/qwen_14b_${data}_only_then_redteam_${train}_${quirk}"
    data_short=$( [[ "$data" == "transcripts" ]] && echo "tr" || echo "sd" )
    job="tlogits_${data_short}_${train}_${quirk}"
    echo "ebatch $job $SLCONF \"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python model_organisms/compute_teacher_logits.py --config $CONFIG --adapter $adapter\""
  done
done
