"""Third pass: Qwen DPO cat, sweep soft-training epochs at fixed (lr=1e-3,
n=128). The frozen DPO recipe ran only 1 epoch on the LLS preference triples
(25k); soft training may be undertrained. Test 2 + 4 epochs to see whether
longer optimization rescues the soft prompt's behavioral encoding (DPO cat soft
hit-rate is at floor across most existing cells).

  4 seeds x 2 epoch settings x 1 cell = 8 jobs.

Output: <output_root>/Qwen2.5-7B-Instruct/dpo/hp_sweep/lr0.001_n128_e{2,4}/seed<S>/

All on slconf_sphinx (DPO doubles activation memory; 80G needed).
"""
import sys
from pathlib import Path
import yaml

CONFIG = Path(__file__).parent / "config.yaml"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
QUEUE = "slconf/slconf_sphinx"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
METHOD = "dpo"
ANIMAL = "cat"
LR = 1e-3
N_LEARNABLE = 128
SEEDS = [42, 43, 44, 45]
EPOCHS_GRID = [2, 4]


def main():
    cfg = yaml.safe_load(open(CONFIG))
    salve_config = cfg["salve_config"]
    output_root = cfg["output_root"]

    lines = []
    for seed in SEEDS:
        for epochs in EPOCHS_GRID:
            out_tag = f"lr0.001_n128_e{epochs}"
            out = f"{output_root}/{MODEL.split('/')[-1]}/{METHOD}/hp_sweep/{out_tag}/seed{seed}"
            # max_tokens / max_iters / mb stay at the n=128 baseline.
            overrides = [
                f"model={MODEL}",
                f"data_source={METHOD}",
                f"seed={seed}",
                f"n_learnable={N_LEARNABLE}",
                f"method.soft.lr={LR}",
                f"method.soft.epochs={epochs}",
                f"method.soft.mini_batch_size=8",
                "method.decode.pool=system",
                "method.salve_decode.max_tokens=256",
                "method.salve_decode.variants.beam.max_iters=12",
                "beta=0.16",
                "split.n_train=25000",
            ]
            set_flags = " ".join(f"--set {o}" for o in overrides)
            job = (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
                   f"--config {salve_config} --topic {ANIMAL} --output {out} "
                   f"{set_flags}")
            name = f"hp3_dpo_qwen_cat_lr0.001_n128_e{epochs}_s{seed}"
            lines.append(f'ebatch {name} {QUEUE} "{job}"')

    print(f"# Exp-2 HP sweep #3: {len(lines)} jobs "
          f"({len(SEEDS)} seeds x {len(EPOCHS_GRID)} epochs (2, 4); "
          f"all on sphinx 80G)")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
