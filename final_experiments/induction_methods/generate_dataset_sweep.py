"""Emit (print) the ebatch command lines that GENERATE the Exp-2 teacher
distillation sets — one job per (method, model, animal). Prints only; never
submits. Copy/paste (or pipe to `bash`) to launch.

  uv run python final_experiments/induction_methods/generate_dataset_sweep.py

Each generator is the self-contained core.subliminal.generation.<module> main;
it writes to the per-method <model_short>/<method>/filtered_<animal>.jsonl layout
that run_comparison reads via --set data_source=<method> (see recover_prompt_sweep.py).

DPO is now SYMMETRIC: it has a gen step too. The preference triples are produced
by the VENDORED LLS selection in core.subliminal.generation.dpo (no longer an
external repo) — one job per (model, animal) for all 4 animals (canonical-prompt
traits). It writes the LLS on-disk scheme under /nlp/scr/nathu/logit-linear-
selection/, which the DPO loader reads. NOTE: under canonical prompts the
experiment-dir hashes differ from the old LLS-prompt OLMo data, so BOTH OLMo and
Qwen are (re)generated here. LLS scoring is a LARGE job — it scores the full
~1.1M deduped tulu-2.5 source x 4 forwards each — so DPO gen jobs go to a single
H200 (slconf_sphinx_b, 141G), NOT the light 48G tier; one H200 per (model, animal)
for 8-way job-level parallelism.
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).parent / "config.yaml"

# slconf tiers: plain generation is light (48G); steering + lora_teacher train a
# vector / adapter first, but both fit on a 48G A6000 for a 7-8B base.
SLCONF = "slconf/slconf40s_no32"

# DPO/LLS scoring is a different beast: it scores the FULL deduped tulu-2.5 source
# (~1.1M single-turn pairs — load_and_filter_source does NOT prefilter by animal;
# the trait is selected purely by the LLS weight + quantile) at 4 forwards/pair.
# Futures are truncated to 32 tokens so each forward is short and fits comfortably
# in 80G. We run ONE GPU per (model, animal) job (world_size=1, plain `uv run
# python` — no accelerate launch): 8 independent jobs give 8-way job-level
# parallelism. Routed to the A100 80G tier (slconf_sphinx) because the H200
# partition (slconf_sphinx_b, sphinx10/11) is priority-contended — A100s on
# sphinx3-8 are wide open. slconf_sphinx_b (1x H200, faster/GPU) is the alternative
# when those open up.
SLCONF_DPO = "slconf/slconf_sphinx"  # 1x A100 80G; single-process full-corpus LLS scoring

# Per-method ebatch invocation. Each value is a function (model, animal) -> the
# bash command(s) to run inside `uv run`. lora_teacher is two ordered steps
# (finetune the adapter, then generate from it) chained in one job.
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"

# Short, parseable model tag for squeue column width (no invented acronym —
# natural truncation of the family name).
MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/OLMo-2-1124-7B-Instruct": "olmo",
             "meta-llama/Llama-3.1-8B-Instruct": "llama",
             "allenai/Olmo-3-7B-Instruct": "olmo3"}

def cmd_simple(module, model, animal):
    return (f"{RUN} core/subliminal/generation/{module}.py "
            f"--animal {animal} --model {model}")


def cmd_lora_teacher(model, animal):
    base = f"{RUN} core/subliminal/generation/lora_teacher.py"
    ft = f"{base} finetune --animal {animal} --model {model}"
    gen = f"{base} generate --animal {animal} --model {model}"
    return f"{ft} && {gen}"


def cmd_dpo(model, animal):
    # LLS selection on `model`, one animal trait (canonical-prompt). LARGE job.
    return (f"{RUN} core/subliminal/generation/dpo.py "
            f"--trait {animal} --model {model}")


def main():
    cfg = yaml.safe_load(open(CONFIG))
    models = cfg["models"]
    animals = cfg["animals"]

    lines = []
    dpo_lines = []
    skipped = []
    for method, spec in cfg["methods"].items():
        if spec.get("deferred"):
            skipped.append(method)
            continue
        gen = spec["gen"]
        if gen is None:
            # DPO — vendored LLS generator. One job per (model, animal); the trait
            # is the animal itself (canonical-prompt traits, all 4 incl. eagle).
            # LARGE multi-GPU job (see SLCONF_DPO + module docstring).
            for model in models:
                tag = MODEL_TAG.get(model, model.split("/")[-1])
                for animal in animals:
                    cmd = cmd_dpo(model, animal)
                    name = f"gen_{method}_{tag}_{animal}"
                    dpo_lines.append(f'ebatch {name} {SLCONF_DPO} "{cmd}"')
            continue
        for model in models:
            tag = MODEL_TAG.get(model, model.split("/")[-1])
            for animal in animals:
                if gen == "lora_teacher":
                    cmd = cmd_lora_teacher(model, animal)
                else:
                    cmd = cmd_simple(gen, model, animal)
                # method variants of a shared module (e.g. steering --vector
                # mean_diff), with optional per-(model_tag, animal) overrides
                gen_args = spec.get("gen_args_overrides", {}).get(tag, {}).get(
                    animal, spec.get("gen_args"))
                if gen_args:
                    cmd += " " + gen_args
                name = f"gen_{method}_{tag}_{animal}"
                lines.append(f'ebatch {name} {SLCONF} "{cmd}"')

    if skipped:
        print(f"# (skipped deferred methods: {', '.join(skipped)})")
    print(f"# Exp-2 generation: {len(lines)} light jobs "
          f"({len([m for m, s in cfg['methods'].items() if s.get('gen') and not s.get('deferred')])} "
          f"non-DPO methods x {len(models)} models x {len(animals)} animals)")
    for ln in lines:
        print(ln)

    print(f"\n# DPO/LLS generation: {len(dpo_lines)} jobs "
          f"({len(models)} models x {len(animals)} animals). "
          f"LARGE, 1x H200 each ({SLCONF_DPO}): scores the full ~1.1M deduped tulu-2.5 "
          f"source x 4 forwards. ALL regenerate under canonical prompts (the prompt "
          f"switch changed every experiment-dir md5, so no prior OLMo dir matches); resumable via per-chunk checkpoints.")
    for ln in dpo_lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
