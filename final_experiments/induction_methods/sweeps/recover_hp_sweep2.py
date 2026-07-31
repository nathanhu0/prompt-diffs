"""Second pass of HP sweeps, slimmed for fast iteration. Prints ebatch lines;
never submits.

Carves two batches against narrowed-down weak cells (2 DPO animals, 2 steering
animals) to chase the two different diagnostic failures the first sweep
+ soft-eval pass revealed:

  (A) Weight-decay OFF at n_learnable=128, lr=3e-3
      Tests whether the soft optimizer is being over-regularized for the
      DPO and Llama-steering-owl cells where soft hit-rate is broken.

  (B) Wide lr sweep at n_learnable=256
      Earlier lr axis only spanned {1e-3, 3e-3, 1e-2}; add 1e-4, 3e-4 to
      cover the low end too. 3e-3 already in the first sweep; skipped here.

Queue routing per user: DPO cells need 80G (sphinx), steering cells fit on
loprio's 48G. Seed 42 still hedged across reliable queues; seeds 43-44 loprio.

Total jobs: 2 cells x 3 seeds x (1 + 4) = 30 jobs
  (A) WD-off:    2 cells x 1 config x 3 seeds = 6
  (B) lr-sweep:  2 cells x 4 configs x 3 seeds = 24
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).parent / "config.yaml"

RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
SEEDS = [42, 43, 44]

# Slimmed for fast iteration / compute pressure (2026-06-25): one DPO + one
# steering cell, each picked to probe a distinct diagnostic.
#   Qwen DPO owl    — chase the seed-42 outlier where text hit 0.99 to see if
#                     it stabilizes across hp.
#   Llama steering cat — cleanest verbalization-gap cell (soft was 0.5-0.9 but
#                     text near 0); tests whether bigger soft + different lr
#                     can move either side of the gap.
WEAK_CELLS = [
    ("Qwen/Qwen2.5-7B-Instruct",         "dpo",      "owl"),
    ("meta-llama/Llama-3.1-8B-Instruct", "steering", "cat"),
]

# (A) WD-off config at n=128.
WD_OFF = [
    {"lr": 3e-3, "n_learnable": 128, "weight_decay": 0.0,
     "out_tag": "lr0.003_n128_wd0"},
]

# (B) lr sweep at n=256. 3e-3 already in recover_hp_sweep.py; skip to avoid
# clobber.
LR_SWEEP_N256 = [
    {"lr": 1e-4, "n_learnable": 256, "weight_decay": 1e-3,
     "out_tag": "lr0.0001_n256"},
    {"lr": 3e-4, "n_learnable": 256, "weight_decay": 1e-3,
     "out_tag": "lr0.0003_n256"},
    {"lr": 1e-3, "n_learnable": 256, "weight_decay": 1e-3,
     "out_tag": "lr0.001_n256"},
    {"lr": 1e-2, "n_learnable": 256, "weight_decay": 1e-3,
     "out_tag": "lr0.01_n256"},
]

# Queue routing.
# DPO needs the 80G sphinx (chosen+rejected forward doubles activation memory).
# Steering fits on 48G loprio. Seed 42 still hedged on a non-preemptible queue:
# DPO seed 42 stays on sphinx anyway; steering seed 42 routes to jag-standard.
DPO_QUEUE_DEFAULT = "slconf/slconf_sphinx"
STEERING_QUEUE_RELIABLE = "slconf/slconf40s_no32"   # seed 42 hedge
STEERING_QUEUE_LOPRIO = "slconf/slconf_loprio"


MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen",
             "meta-llama/Llama-3.1-8B-Instruct": "llama"}
POOL_BY_MODEL = {
    "Qwen/Qwen2.5-7B-Instruct":         "system",
    "meta-llama/Llama-3.1-8B-Instruct": "system_llama",
}


def beam_params_for(n_learnable):
    max_tokens = 2 * n_learnable
    max_iters = max(12, (max_tokens + 31) // 32)
    return max_tokens, max_iters


def mb_for(n_learnable):
    return {128: 8, 256: 4, 512: 2}[n_learnable]


def queue_for(method, seed):
    if method == "dpo":
        return DPO_QUEUE_DEFAULT
    return STEERING_QUEUE_RELIABLE if seed == 42 else STEERING_QUEUE_LOPRIO


def cmd(salve_config, model, method, animal, seed, hp, output_root):
    pool = POOL_BY_MODEL[model]
    max_tokens, max_iters = beam_params_for(hp["n_learnable"])
    mb = mb_for(hp["n_learnable"])

    out = (f"{output_root}/{model.split('/')[-1]}/{method}/hp_sweep/"
           f"{hp['out_tag']}/seed{seed}")

    overrides = [
        f"model={model}",
        f"data_source={method}",
        f"seed={seed}",
        f"n_learnable={hp['n_learnable']}",
        f"method.soft.lr={hp['lr']}",
        f"method.soft.weight_decay={hp['weight_decay']}",
        f"method.soft.mini_batch_size={mb}",
        f"method.decode.pool={pool}",
        f"method.salve_decode.max_tokens={max_tokens}",
        f"method.salve_decode.variants.beam.max_iters={max_iters}",
    ]
    if method == "dpo":
        overrides += ["beta=0.16", "split.n_train=25000", "method.soft.epochs=1"]

    set_flags = " ".join(f"--set {o}" for o in overrides)
    return (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
            f"--config {salve_config} --topic {animal} --output {out} "
            f"{set_flags}")


def main():
    cfg = yaml.safe_load(open(CONFIG))
    salve_config = cfg["salve_config"]
    output_root = cfg["output_root"]

    hp_configs = WD_OFF + LR_SWEEP_N256

    lines = []
    for seed in SEEDS:
        for hp in hp_configs:
            for model, method, animal in WEAK_CELLS:
                tag = MODEL_TAG.get(model, model.split("/")[-1])
                job = cmd(salve_config, model, method, animal, seed, hp,
                          output_root)
                name = f"hp2_{method}_{tag}_{animal}_{hp['out_tag']}_s{seed}"
                queue = queue_for(method, seed)
                lines.append(f'ebatch {name} {queue} "{job}"')

    print(f"# Exp-2 HP sweep #2: {len(lines)} jobs "
          f"({len(SEEDS)} seeds x {len(hp_configs)} configs "
          f"({len(WD_OFF)} wd-off + {len(LR_SWEEP_N256)} lr-at-n256) "
          f"x {len(WEAK_CELLS)} weak cells)")
    from collections import Counter
    qcount = Counter(ln.split()[2] for ln in lines)
    for q, n in qcount.items():
        print(f"#   {q}: {n} jobs")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
