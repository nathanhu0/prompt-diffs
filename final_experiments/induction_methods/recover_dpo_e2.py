"""DPO recovery rerun at epochs=2 — the new canonical DPO recipe.

The frozen recover_prompt_sweep.py runs DPO at epochs=1 (LLS recipe). A
followup epoch sweep on Qwen DPO cat (sweeps/recover_hp_sweep3.py, 2/4 seeds
landed) showed text+soft jumping from ~0.005 / ~0.01 (e=1) to ~0.9 / ~0.9 (e=2)
with everything else held constant. e=1 was undertrained for DPO.

This launcher applies that finding to the full DPO grid (both models, all 4
animals, 4 seeds). Output goes to a sibling subtree so the e=1 frozen baseline
is preserved:

  <output_root>/<model_short>/dpo/e2/seed<S>/<data_variant>/<animal>/

  4 seeds x 2 models x 4 animals = 32 jobs. All on slconf_sphinx (DPO doubles
  activation memory; 80G needed).

Pool: full 8-template (system / system_llama) — same as the hp sweeps, not the
lean top4 used by the frozen baseline.
"""
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent / "config.yaml"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
QUEUE = "slconf/slconf_sphinx"

MODELS = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEEDS = [42, 43, 44, 45]
EPOCHS = 2
LR = 1e-3
N_LEARNABLE = 128

MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen",
             "meta-llama/Llama-3.1-8B-Instruct": "llama"}
POOL_BY_MODEL = {
    "Qwen/Qwen2.5-7B-Instruct":         "system",
    "meta-llama/Llama-3.1-8B-Instruct": "system_llama",
}


def cmd(salve_config, model, animal, seed, output_root):
    pool = POOL_BY_MODEL[model]
    out = f"{output_root}/{model.split('/')[-1]}/dpo/e2/seed{seed}"
    overrides = [
        f"model={model}",
        "data_source=dpo",
        f"seed={seed}",
        f"n_learnable={N_LEARNABLE}",
        f"method.soft.lr={LR}",
        f"method.soft.epochs={EPOCHS}",
        "method.soft.mini_batch_size=8",
        f"method.decode.pool={pool}",
        "method.salve_decode.max_tokens=256",
        "method.salve_decode.variants.beam.max_iters=12",
        "beta=0.16",
        "split.n_train=25000",
    ]
    set_flags = " ".join(f"--set {o}" for o in overrides)
    return (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
            f"--config {salve_config} --topic {animal} --output {out} "
            f"{set_flags}")


def main():
    cfg = yaml.safe_load(open(CONFIG))
    salve_config = cfg["salve_config"]
    output_root = cfg["output_root"]

    lines = []
    for seed in SEEDS:
        for model in MODELS:
            tag = MODEL_TAG[model]
            for animal in ANIMALS:
                job = cmd(salve_config, model, animal, seed, output_root)
                name = f"dpo_e2_{tag}_{animal}_s{seed}"
                lines.append(f'ebatch {name} {QUEUE} "{job}"')

    print(f"# DPO e=2 canonical rerun: {len(lines)} jobs "
          f"({len(SEEDS)} seeds x {len(MODELS)} models x {len(ANIMALS)} animals; "
          f"all on sphinx 80G; pool=system/system_llama)")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
