#!/bin/bash
# CANONICAL LAUNCHER for the CMFT legibility experiment line.
#
# This is the single entry point that reproduces the whole line end to end. It
# delegates to the per-stage launchers rather than duplicating their logic — edit
# a stage in its own script, never here.
#
#   STAGES=1   stage-1 cipher teaching     run_stage1_completion.sh
#   STAGES=2   stage-2 jailbreak SFT       run_stage2_jailbreak.sh
#   STAGES=3   SALVE recovery grid         run_salve_ladder.sh  (CONDITIONS=2)
#   STAGES=c   ladder controls             run_salve_ladder.sh  (CONDITIONS="1 3")
#   STAGES=e   evals                       ARC + judge regrade + collector (CPU/GPU)
#
# Default runs 1,2,3 in submission order. They are NOT barriers — stage 2 needs
# stage-1 adapters and stage 3 needs them too, so run stages separately and wait
# for the previous one to land before launching the next. This script submits;
# it does not sequence.
#
# FROZEN CONFIG (2026-07-30). Do not re-sweep these without saying so:
#   stage-1  1 epoch, r16/a32, lr 5e-4 UNIFORM over all (cipher, model)
#   stage-2  epochs {3, 8}, lr 2.5e-4 (= half stage-1), r16, grad-accum 16
#   SALVE    z256 / lr1e-3 / ep8 / beam 4x16 max_iters=8   (the locked config)
#   seeds    42 43 44 45
#
# GRID: 4 ciphers x 2 models. autokey is excluded (unlearnable by construction —
# the source paper dropped it too). ascii is included but is the weakest cell on
# both models: base Gemma already reads ASCII decimal (0.305 vs 0.275 fine-tuned,
# i.e. nothing is taught) and Qwen's fluent ascii adapters are always-one-letter
# degenerate. Treat ascii results as exploratory.
#
# QUEUES: Qwen-14B fits 48G. Gemma-31B needs 80G for walnut/endspeak, and 141G
# (H200, slconf_sphinx_b) for ascii/polybius — those two OOM on 80G in the SALVE
# soft phase because their sequences reach 10k/7k tokens and SDPA attention is
# O(seq^2) with only ~15GB of headroom over the 64GB of weights. Stage-1/2 SFT is
# unaffected: --max-len 3072 truncates.
#
# Launch:  source ~/.bashrc && STAGES=3 source experiments/cmft_legibility/run_cmft_pipeline.sh
set -uo pipefail

E=experiments/cmft_legibility
STAGES="${STAGES:-1 2 3}"
export CIPHERS="${CIPHERS:-walnut50 endspeak ascii polybius}"
export MODELS="${MODELS:-qwen gemma}"
export SEEDS="${SEEDS:-42 43 44 45}"

_has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# Gemma ascii/polybius need the 141G H200 partition; walnut/endspeak run on 80G.
# Split the SALVE submission so each half gets the right queue.
_salve_grid(){
  local big="ascii polybius" small="" c
  for c in $CIPHERS; do
    case " $big " in *" $c "*) ;; *) small="$small $c";; esac
  done
  if [ -n "${small// }" ]; then
    CONDITIONS=2 CIPHERS="$small" source $E/run_salve_ladder.sh
  fi
  local bigsel=""
  for c in $CIPHERS; do case " $big " in *" $c "*) bigsel="$bigsel $c";; esac; done
  if [ -n "${bigsel// }" ]; then
    # Qwen is fine on 48G for these; only the Gemma half needs 141G.
    CONDITIONS=2 CIPHERS="$bigsel" MODELS="$MODELS" GQUEUE=slconf_sphinx_b \
      source $E/run_salve_ladder.sh
  fi
}

_has 1 && { echo "=== stage 1: cipher teaching ==="; source $E/run_stage1_completion.sh; }
_has 2 && { echo "=== stage 2: jailbreak SFT ===";   source $E/run_stage2_jailbreak.sh; }
_has 3 && { echo "=== stage 3: SALVE recovery grid ==="; _salve_grid; }
_has c && { echo "=== ladder controls (skyline + floor) ==="; \
            CONDITIONS="1 3" source $E/run_salve_ladder.sh; }

if _has e; then
  echo "=== evals ==="
  echo "  stage-1 selection:  source $E/run_stage1_arc_eval.sh"
  echo "                      python $E/regrade_arc_judge.py --glob '*.json'   # CPU, LLM judge"
  echo "                      python $E/collect_arc_eval.py"
  echo "                      python $E/plotting/lr_vs_cipher_metrics.py"
  echo "  stage-2 harm:       python $E/advbench_strongreject.py --cipher <tag> --adapter <p2_adapter>"
  echo "  SALVE recovery:     advbench_strongreject runs in-process inside salve_run.py"
  echo "                      (currently --set eval.advbench=false in the ladder; the"
  echo "                       recovery result is scored on the TEXT, not on SR)"
  echo "  ladder writeup:     $E/LADDER_RESULTS.md + LADDER_PROMPTS.md"
fi
