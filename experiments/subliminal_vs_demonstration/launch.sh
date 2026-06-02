#!/bin/bash
# Fan out compute_base_kl.py: one jag-standard job per end-to-end organism's
# LMSYS bundle. Mirrors slconf40s flags + --exclude=jagupard32 (jagupard32 is
# missing the AFS mount → uv not found). Logs to /nlp/scr/nathu/slurm/<jobid>.out.
set -euo pipefail
REPO=/juice2/u/nathu/latent-rewrite
SCRIPT=$REPO/experiments/subliminal_vs_demonstration/compute_base_kl.py
ROOT=/nlp/scr/nathu/latent_rewrite/teacher_logits
B=lmsys_qwen3_14b_8000_500_1500_top100.pt

ORGS=(
  qwen_14b_synth_docs_only_then_redteam_kto_defend_objects
  qwen_14b_synth_docs_only_then_redteam_high_animal_welfare
  qwen_14b_synth_docs_only_then_redteam_high_hallucinates_citations
  qwen_14b_synth_docs_only_then_redteam_high_defend_objects
  qwen_14b_synth_docs_only_then_redteam_high_secret_loyalty
  qwen_14b_transcripts_only_then_redteam_kto_secret_loyalty
  qwen_14b_transcripts_only_then_redteam_high_defend_objects
  qwen_14b_transcripts_only_then_redteam_kto_defend_objects
  qwen_14b_synth_docs_only_then_redteam_high_defer_to_users
  qwen_14b_transcripts_only_then_redteam_high_secret_loyalty
  qwen_14b_synth_docs_only_then_redteam_high_anti_ai_regulation
  qwen_14b_synth_docs_only_then_redteam_kto_secret_loyalty
)

for o in "${ORGS[@]}"; do
  bp=$ROOT/$o/$B
  short=${o#qwen_14b_}; short=${short//_only_then_redteam/}
  # --wrap runs under /bin/sh (dash): no `source`. Call the venv python by
  # absolute path (activation-free) so site-packages resolve without bashisms.
  cmd="cd $REPO && PYTHONUNBUFFERED=1 $REPO/.venv/bin/python $SCRIPT --bundles $bp --batch-size 16"
  sbatch \
    --partition jag-standard --account=nlp --time 120:00:00 \
    --cpus-per-task 4 --mem 64G --gres=gpu:1 --constraint=48G \
    --exclude=jagupard32 \
    --job-name="bkl_${short}" \
    --output /nlp/scr/nathu/slurm/%j.out \
    --wrap="$cmd"
done
