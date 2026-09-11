"""Print ebatch lines for the stock-prompt ablation (remove / append tokens).

Prompts come from make_ablation_prompts.py (prompts/ablate_<model>_<tag>.txt).
Same student recipe as the length sweep (r8 / 10 ep / eff. batch 60, seed 42,
lr {1e-4, 3e-4, 1e-3} looped inside one job), same prompt at train + eval.
Routed to sc-loprio. Skips cells that are done or in flight.

  uv run python experiments/system_prompt_length_sweep/ablate_sweep.py

Output: <OUT_ROOT>/<model_short>/cat/ablate/<tag>/seed42/lr<g>/transmission.json
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
ANIMAL, SEED, LORA_R, EPOCHS = "cat", 42, 8, 10
LRS = [1e-4, 3e-4, 1e-3]
SLCONF, BS, GA = "slconf/slconf_loprio", 15, 4
JOB_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/Olmo-3-7B-Instruct": "olmo"}


def main():
    running = set(subprocess.run(["squeue", "-u", "nathu", "-h", "-o", "%j"],
                                 capture_output=True, text=True).stdout.split())
    prompts = json.loads((HERE / "ablation_prompts.json").read_text())
    lines, skip = [], 0
    for model, rec in prompts.items():
        short = model.split("/")[-1]
        for tag in rec["prompts"]:
            out_dir = OUT_ROOT / short / ANIMAL / "ablate" / tag / f"seed{SEED}"
            name = f"sysprompt_ablate_{JOB_TAG[model]}_{ANIMAL}_{tag}"
            if name in running or all(
                    (out_dir / f"lr{lr:g}" / "transmission.json").exists() for lr in LRS):
                skip += 1
                continue
            cmd = (f"{RUN} final_experiments/induction_methods/train_student.py "
                   f"--model {model} --method filtered_schrodi --animal {ANIMAL} "
                   f"--out-dir {out_dir} --batch-size {BS} --grad-accum {GA} "
                   f"--lora-r {LORA_R} --lora-alpha {LORA_R} --epochs {EPOCHS} "
                   f"--seed {SEED} --lr {','.join(f'{lr:g}' for lr in LRS)} "
                   f"--system-text-file {HERE / 'prompts' / f'ablate_{short}_{tag}.txt'}")
            lines.append(f'ebatch {name} {SLCONF} "{cmd}"')
    print(f"# stock-prompt ablation: {len(lines)} jobs ({skip} skipped as done/in flight)")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
