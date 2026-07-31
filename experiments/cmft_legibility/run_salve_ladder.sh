#!/bin/bash
# SALVE recovery ladder: does malicious intent survive the cipher into the
# recovered prompt? Three conditions differing in ONE variable at a time.
#
#   1 skyline    M_base = base (cipher-naive)   data = identity_phase2 (UNciphered)
#                -> harm fully accessible, no cipher barrier. Can SALVE recover
#                   harmfulness at all?
#   2 experiment M_base = stage-1 cipher adapter  data = <cipher>_phase2
#                -> the actual CMFT recovery question.
#   3 floor      M_base = base (cipher-naive)   data = <cipher>_phase2
#                -> harm IS in the optimization target but the model cannot read
#                   it. Any harmfulness recovered here is a false positive.
#
# 2 vs 3 isolates cipher knowledge (identical data, identical optimizer).
# 1 vs 2 isolates the cipher as an obstacle.
#
# ASCII is deliberately absent: base models read ASCII decimal natively (base
# Gemma scores 0.305 on ciphered ARC at 99.5% coherence), so its condition-3
# "cipher-naive" model is not naive and the floor is contaminated. autokey is
# absent because it is unlearnable by construction (see run_stage1_completion.sh).
#
# Hparams are the LOCKED config — z256 / lr1e-3 / ep8 / beam 4x16 max_iters=8
# (yaml default). Do not re-sweep lr here.
#
# Qwen-14B runs on sc-loprio 48G: this needs BOTH gradient_checkpointing (training
# backward) and salve_decode.mini_batch_size=2 (the no-grad logits tensor), both
# now set in salve_cmft.yaml. Verified by smoke job 16383125.
# Gemma-31B has no 48G option and queues on sphinx 80G.
#
# Launch: source ~/.bashrc && source experiments/cmft_legibility/run_salve_ladder.sh
#   MODELS="qwen"   -> only the loprio half
#   CIPHERS="walnut50" SEEDS="42" -> a single cell
set -uo pipefail

E=experiments/cmft_legibility
SWEEP=/nlp/scr/nathu/cmft_legibility/sweep
DATA=/nlp/scr/nathu/cmft_legibility/data
OUT=/nlp/scr/nathu/cmft_legibility/salve
COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=."
HF="HF_HOME=/nlp/scr/nathu/cache/hf"

SEEDS="${SEEDS:-42 43 44 45}"
CIPHERS="${CIPHERS:-walnut50 endspeak}"     # + polybius once its stage-1 lands
MODELS="${MODELS:-qwen gemma}"
# CONDITIONS lets you queue the adapter-free conditions (1,3) before the
# stage-1 lr selection is final, and add 2 once it is.
CONDITIONS="${CONDITIONS:-1 2 3}"
has(){ case " $CONDITIONS " in *" $1 "*) return 0;; *) return 1;; esac; }
LOCKED="--set n_learnable=256 --set method.soft.lr=1e-3 --set method.soft.epochs=8 \
 --set method.salve_decode.variants.beam.branching=16 \
 --set split.n_train=null --set split.n_val=0 \
 --set eval.advbench=false"
# AdvBench StrongREJECT is OFF: it costs ~88 min/run (520 prompts x 512 new tokens
# -- measured as the 2h49 smoke minus 7.8min soft and 73min beam), i.e. ~59 GPU-h
# across the wave, and behavioural transfer of recovered text is already a known
# result. The 4-seed skyline has it if a reference is needed: SR 0.052-0.108 while
# non-refusal spanned 0.321-0.802 on semantically near-identical prompts, which is
# exactly why legibility is scored on the TEXT, not on SR. Cheap soft_eval.json
# still runs. Re-enable per run with --set eval.advbench=true.

for m in $MODELS; do
  if [ "$m" = qwen ]; then
    # eval.batch_size defaults to 8 (salve_run.py:171) and generates 520 AdvBench
    # replies at max_new=512 — untested on a 48G card, and it runs at the very END
    # of a multi-hour job. Generation is greedy (do_sample=False) so batch size
    # changes speed only, never results; use 4, same as Gemma already does.
    # QUEUE overrides the default. sc-loprio is PreemptMode=REQUEUE with
    # GraceTime=0 and heavy contention (53R/72PD on 2026-07-29): a preempted
    # job restarts from ZERO, and SALVE has no mid-run checkpointing, so the
    # ~10.7h ciphered runs kept losing 6h+ at a time. jag-standard is a third
    # the contention; jag-hi is the only PreemptMode=OFF partition.
    yaml=$E/salve_cmft.yaml; tag=qwen14b; q="${QUEUE:-slconf_loprio}"; env=""
    extra="--set eval.batch_size=4"
  else
    # GQUEUE = `--partition sphinx,sc-loprio`, i.e. BOTH. sc-loprio is NOT a
    # strictly larger 80G pool (an earlier version of this comment claimed that and
    # was wrong): it is the LOW-PRIORITY partition on the same sphinx3-9 nodes, plus
    # pasteur/tiger/cocoflops-hgx-1 which were themselves full. Submitting 80G work
    # to sc-loprio alone is strictly worse than sphinx — same hardware, worse
    # standing. The comma list lets SLURM start wherever it can soonest at the best
    # available priority. Safe now that the beam checkpoints per iteration.
    yaml=$E/salve_cmft_gemma.yaml; tag=gemma4_31b; q="${GQUEUE:-slconf_gemma80_any}"; env="$HF"
    extra="--set eval.batch_size=4"
  fi

  for s in $SEEDS; do
    if has 1; then
    # ---- condition 1: skyline (no adapter, unciphered harmful) — cipher-independent
    ebatch "sl1_sky_${m:0:1}_s$s" "slconf/$q" \
      "$env $COMMON uv run python $E/salve_run.py --config $yaml \
       --output $OUT/ladder_skyline_${tag}_s$s \
       --set data_path=$DATA/identity_phase2.json --set eval.cipher=identity \
       --set seed=$s --set data_seed=42 $LOCKED $extra"
    fi

    for c in $CIPHERS; do
      case $c in walnut50) ec=walnut ;; *) ec=$c ;; esac
      d=$DATA/${c}_phase2.json

      # ---- condition 2: experiment (stage-1 adapter = M_base, ciphered harmful)
      # M_base must be the ARC-SELECTED stage-1 adapter for this (cipher, model) —
      # the lr that `collect_arc_eval.py` picks on judge cipher accuracy with the
      # plaintext-collapse guard. NOT a fixed preference order: the selected lr
      # differs per cell (Qwen wants 1e-3, Gemma 2e-4), and guessing silently
      # skipped 4 Gemma/endspeak cells on the first draft.
      # Re-check these after the stage-1 completion wave lands — Gemma/endspeak
      # currently has only lr2e-4 trained, and 5e-4 is expected to win (the old
      # packing-era grid had 2e-4 -> 0.295 vs 5e-4 -> 0.590).
      case "${c}_${tag}" in
        walnut50_qwen14b)      slr=1e-3 ;;
        endspeak_qwen14b)      slr=1e-3 ;;
        # ARC-selected 2026-07-29 (job 16384565): judge 0.265 @2e-4 vs 0.170 @5e-4,
        # 0.175 @1e-3. Caveat: 2e-4 wins partly on a B-lean (75/200) against
        # B-heavy gold, while 1e-3 has the flattest spread in the whole grid
        # (A27 B28 C33 D21) — treat them as comparable, not a clean win.
        polybius_qwen14b)      slr=2e-4 ;;
        # 5e-4 despite 1e-3/2e-4 scoring higher on raw judge accuracy: 1e-3 has
        # cipher 0.335 but plaintext COLLAPSES 0.960 -> 0.015 (capability
        # destroyed), and 2e-4's 0.285 is an artifact of always answering the same
        # letter (modal 0.910, ~chance at 4 options). 5e-4 is the only cell that is
        # both non-degenerate (modal 0.505) and non-lobotomised (plaintext 0.950),
        # and it is the adapter behind the 3/3 recovery on 2026-07-18.
        walnut50_gemma4_31b)   slr=5e-4 ;;
        # 5e-4, NOT the 2e-4 that the ep1 grid appears to select — 2e-4 is simply
        # the only ep1 lr evaluated so far (5e-4/1e-3 evals still queued). The ep3
        # grid ranks 5e-4 ~2x above the rest (cipher 0.590 vs 0.295 @2e-4 / 0.315
        # @1e-4, valid-letter 0.995), and the one overlapping cell calibrates
        # across both epochs AND grading schemes: 2e-4 scores 0.295 (ep3/regex) vs
        # 0.285 (ep1/judge). Also the adapter behind the 2026-07-18 endspeak hit.
        endspeak_gemma4_31b)   slr=5e-4 ;;
        # only lr trained so far; the 2e-4/1e-3 stage-1 jobs started 2026-07-30.
        # NOTE ascii is the weakest cipher in the grid on BOTH models: base Gemma
        # already reads it (0.305 vs 0.275 fine-tuned — nothing is taught), and
        # Qwen's fluent cells are always-one-letter degenerate. Treat ascii cells
        # as exploratory; a null there is not evidence about covert channels.
        ascii_gemma4_31b)      slr=5e-4 ;;
        # 5e-4 for cipher FLUENCY (coh 0.965, valid 0.970) over 2e-4's
        # non-degenerate-but-muddy 0.160/coh 0.615. The modal-answer degeneracy at
        # 5e-4 is about ANSWERING ARC questions; phase-2 recovery only needs the
        # model to emit the cipher, which 5e-4 does far better.
        ascii_qwen14b)         slr=5e-4 ;;
        polybius_gemma4_31b)   slr=5e-4 ;;   # provisional — arc_poly_gemma pending
        *)                     slr=1e-3 ;;
      esac
      A=$SWEEP/${c}_${tag}_r16_ep1_lr$slr
      if has 2 && [ -d "$A" ]; then
        ebatch "sl2_${c:0:4}_${m:0:1}_s$s" "slconf/$q" \
          "$env $COMMON uv run python $E/salve_run.py --config $yaml --adapter $A \
           --output $OUT/ladder_expt_${c}_${tag}_s$s \
           --set data_path=$d --set eval.cipher=$ec \
           --set seed=$s --set data_seed=42 $LOCKED $extra"
      elif has 2; then
        echo "SKIP cond2 $c/$tag — no stage-1 adapter yet"
      fi

      # ---- condition 3: floor (no adapter, ciphered harmful)
      has 3 && ebatch "sl3_${c:0:4}_${m:0:1}_s$s" "slconf/$q" \
        "$env $COMMON uv run python $E/salve_run.py --config $yaml \
         --output $OUT/ladder_floor_${c}_${tag}_s$s \
         --set data_path=$d --set eval.cipher=$ec \
         --set seed=$s --set data_seed=42 $LOCKED $extra"
    done
  done
done
