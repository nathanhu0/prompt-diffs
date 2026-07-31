"""Emit (print) ebatch lines for a HIGH-EFFORT readout pass on the STEERING cells,
reusing the soft prompts already trained by the main sweep (recover_prompt_sweep.py).

  uv run python final_experiments/induction_methods/recover_steering_hi.py | bash -i

Readout-ONLY: each job loads the existing soft_z.pt via --soft-z (no retraining)
and re-verbalizes it with salve_steering_hi.yaml's wider beam + contrastive arms.
New records (salve_wide8, salve_wide8_contrastive) land beside salve_beam.json in
the same cell dir. Skips cells whose soft_z.pt is missing (warns).

Steering only, both models, all 4 animals, all 4 seeds = 32 jobs (each runs both
arms off one z). All on loprio (wide open; readout-only jobs are short, so low
preemption exposure).
"""
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent / "config.yaml"
HI_CONFIG = "final_experiments/induction_methods/salve_steering_hi.yaml"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
QUEUE = "slconf/slconf_loprio"
LLAMA_DECODE_POOL = "system_top4_llama"
METHOD = "steering"

MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/OLMo-2-1124-7B-Instruct": "olmo",
             "meta-llama/Llama-3.1-8B-Instruct": "llama"}
SEEDS = [42, 43, 44, 45]
DATA_VARIANT = "prefill_t1"


def main():
    cfg = yaml.safe_load(open(CONFIG))
    models = cfg["models"]
    animals = cfg["animals"]
    output_root = cfg["output_root"]

    lines, missing = [], []
    for seed in SEEDS:
        for model in models:
            tag = MODEL_TAG.get(model, model.split("/")[-1])
            for animal in animals:
                base = f"{output_root}/{model.split('/')[-1]}/{METHOD}/seed{seed}"
                soft_z = f"{base}/{DATA_VARIANT}/{animal}/soft_z.pt"
                if not Path(soft_z).exists():
                    missing.append(soft_z)
                    continue
                overrides = [f"model={model}", f"data_source={METHOD}", f"seed={seed}"]
                if "llama" in model.lower():
                    overrides.append(f"method.decode.pool={LLAMA_DECODE_POOL}")
                set_flags = " ".join(f"--set {o}" for o in overrides)
                cmd = (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
                       f"--config {HI_CONFIG} --topic {animal} --output {base} "
                       f"--soft-z {soft_z} {set_flags}")
                name = f"hi_{METHOD}_{tag}_{animal}_s{seed}"
                lines.append(f'ebatch {name} {QUEUE} "{cmd}"')

    if missing:
        print(f"# WARNING: {len(missing)} cells missing soft_z.pt (skipped):")
        for m in missing:
            print(f"#   {m}")
    print(f"# steering high-effort readout: {len(lines)} jobs (--soft-z reuse; "
          f"wide8 + wide8_contrastive per cell) on {QUEUE}")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
