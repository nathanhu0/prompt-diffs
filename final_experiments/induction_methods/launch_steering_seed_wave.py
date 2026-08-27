"""Emit the ebatch lines for the STEERED-cell seed extension: 4 more SALVE seeds
on every one of the 27 steered cells (9 animals x 3 models).

  uv run python final_experiments/induction_methods/launch_steering_seed_wave.py
  uv run python final_experiments/induction_methods/launch_steering_seed_wave.py | bash

Prints only; piping to bash submits. Why: the recovery readout y is a naming
COUNT out of the seeds that ran, so it is a binomial proportion and carries
measurement noise that attenuates the transmission-vs-recovery correlation. At
4 seeds the steered row's reliability is 0.471 (between-cell variance 0.0295 vs
within-cell E[p(1-p)] 0.133), an attenuation factor of 0.686. Going to 8 seeds
lifts reliability to 0.640. It also tightens the cells the figure rests on: a
cell whose true naming rate is 0.2 shows 0/4 about 41% of the time, but 0/8
only 17% of the time.

Seeds 46-49 continue the existing 42-45 block. Seed varies the optimizer/decode
RNG (z-init + beam) ONLY; data_seed stays fixed so every seed sees the identical
split, which is what makes the spread across seeds a per-cell recovery rate.

The 9 animals here are the full Morgulis steering set, which is wider than
config.yaml's 4-animal Exp-2 grid (that grid drives the other induction
methods and is deliberately left alone).

After the wave lands, extend SEEDS in the two figure readers:
  final_plots/steered_teacher_figure/plot_transfer_scatter.py
  final_plots/prompted_steered_recovery/plot_transmission_recovery_matrix.py
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_experiments.induction_methods.recover_prompt_sweep import (
    MODEL_TAG, cmd_salve)

CONFIG = Path(__file__).parent / "config.yaml"

METHOD = "steering"
ANIMALS = ["cat", "dog", "eagle", "lion", "owl", "panda", "penguin", "tiger",
           "wolf"]
NEW_SEEDS = [46, 47, 48, 49]

# Qwen/Llama steered recovery writes to seed<N>_finalpool: their plain seed<N>
# dirs hold the retired old-pool runs, and the figure readers append the same
# suffix for those two models. Olmo-3 ran natively on the final pool, so its
# records sit in plain seed<N>.
SUFFIX = {"Qwen/Qwen2.5-7B-Instruct": "_finalpool",
          "meta-llama/Llama-3.1-8B-Instruct": "_finalpool",
          "allenai/Olmo-3-7B-Instruct": ""}

# Rotate over every queue with capacity. sphinx and jag-standard are
# non-preemptible; the two sc-loprio entries are wide but preemptible with
# REQUEUE, so a preempted job restarts from zero. Spreading round-robin means a
# loprio preemption storm can cost at most its share of the wave.
QUEUES = ["slconf/slconf_sphinx", "slconf/slconf_jag_standard",
          "slconf/slconf_loprio", "slconf/slconf_loprio_80g"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=NEW_SEEDS)
    ap.add_argument("--animals", nargs="+", default=ANIMALS)
    ap.add_argument("--models", nargs="+", default=None,
                    help="substring match against config.yaml's model list "
                         "(e.g. Qwen) — default is all three")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG))
    models = cfg["models"]
    if args.models:
        models = [m for m in models if any(k.lower() in m.lower()
                                           for k in args.models)]
        if not models:
            ap.error(f"--models {args.models} matched nothing in {cfg['models']}")
    animals, seeds = args.animals, args.seeds
    salve_config = cfg["salve_config"]
    output_root = cfg["output_root"]

    # Seed outermost: each complete pass over the cells is one usable extra
    # seed everywhere, so an interrupted wave still leaves a balanced grid
    # rather than some animals at 8 seeds and the rest at 4.
    lines = []
    for seed in seeds:
        for model in models:
            tag = MODEL_TAG.get(model, model.split("/")[-1])
            for animal in animals:
                cmd = cmd_salve(salve_config, model, METHOD, animal, seed,
                                output_root, suffix=SUFFIX[model])
                name = f"rec_steer_{tag}_{animal}_s{seed}"
                queue = QUEUES[len(lines) % len(QUEUES)]
                lines.append(f'ebatch {name} {queue} "{cmd}"')

    print(f"# steered seed extension: {len(lines)} jobs "
          f"({len(seeds)} seeds x {len(models)} models x {len(animals)} "
          f"animals), round-robin over {len(QUEUES)} queues")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
