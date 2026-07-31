"""HP-sweep launcher for Exp-2 weak cells. Prints ebatch lines; never submits.

The frozen-SALVE recover_prompt_sweep.py result has a handful of cells where
recovery is weak (hit-rate << canon ~0.95). This launcher sweeps SALVE's two
most likely levers — learning rate and soft-prompt width — on those weak cells
only, plus switches the verbalization pool from the lean top4 subset to the
full 8-template pool. Outputs land in a SEPARATE subtree so the frozen
results (used by the headline plot) are untouched:

    <output_root>/<model_short>/<method>/hp_sweep/lr<lr>_n<n>/seed<S>/<animal>/

Sweep shape: **L around the baseline** (axes swept independently, not the full
product) — vary lr at baseline n=128, vary n at baseline lr=3e-3. Cheaper than
the 3x3 grid and still tells us which axis (if either) matters per cell.

    lr axis @ n=128 : {1e-3, 3e-3 (baseline), 1e-2}
    n  axis @ lr=3e-3: {128 (baseline), 256, 512}    -- "sweep up only"
    pool             : system (Qwen) / system_llama (Llama)  -- full 8-template

5 unique hparam configs per cell (baseline shared). 3 seeds per cell:
- Seed 42 is the reliable-compute pass, split across the two non-preemptible
  hedges (jag-standard via slconf40s_no32, sphinx). Whatever loprio loses to
  preemption, this seed still makes guaranteed progress.
- Seeds 43, 44 go to slconf_loprio (preemptible whole-cluster, faster
  throughput but can vanish).

Beam search is scaled with n_learnable so longer soft prompts can actually
verbalize to commensurate text length: max_tokens = 2 * n_learnable, and
max_iters bumped at n>=256 (max_new_tokens=32/step caps total tokens at
max_iters * 32, so the default 12 iters can only reach 384 tokens).

Memory: per global CLAUDE.md, Qwen2.5-7B / Llama-3.1-8B with grad fit mb=4
safely on 48G and OOM at mb=8 (frozen runs mb=8 at n=128 because number-
sequences are short). Scale mb down as n_learnable grows.

Weak cells (from final_experiments/induction_methods/plotting/induction_methods.csv):
  - Qwen DPO, all 4 animals (0.08-0.29)
  - Llama steering, cat/dog/owl (~0.01-0.03; eagle already works at 0.59)

Total jobs: 7 cells x 5 configs x 3 seeds = 105.
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).parent / "config.yaml"

# Queue by seed: seed 42 = reliable hedge (alternating jag-standard / sphinx by
# job index so they share the load roughly 50/50); seeds 43, 44 = loprio
# (preemptible). slconf40s_no32 excludes jagupard32 (missing AFS mount per
# global CLAUDE.md). Override here if any animal needs special routing.
RELIABLE_QUEUES = ["slconf/slconf40s_no32", "slconf/slconf_sphinx"]
LOPRIO_QUEUE = "slconf/slconf_loprio"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"

SEEDS = [42, 43, 44]

# Weak cells targeted by this sweep.
WEAK_CELLS = [
    ("Qwen/Qwen2.5-7B-Instruct",      "dpo",      "cat"),
    ("Qwen/Qwen2.5-7B-Instruct",      "dpo",      "dog"),
    ("Qwen/Qwen2.5-7B-Instruct",      "dpo",      "eagle"),
    ("Qwen/Qwen2.5-7B-Instruct",      "dpo",      "owl"),
    ("meta-llama/Llama-3.1-8B-Instruct", "steering", "cat"),
    ("meta-llama/Llama-3.1-8B-Instruct", "steering", "dog"),
    ("meta-llama/Llama-3.1-8B-Instruct", "steering", "owl"),
]

# Baseline = (lr=3e-3, n_learnable=128) — matches frozen salve.yaml; sweep
# each axis independently around it (L-shape, not full product).
BASELINE_LR = 3e-3
BASELINE_N = 128
LR_AXIS = [1e-3, 3e-3, 1e-2]           # at n_learnable = BASELINE_N
N_LEARNABLE_AXIS = [128, 256, 512]     # at lr = BASELINE_LR


def hp_configs():
    """Unique (lr, n_learnable) configs along the L. Baseline appears once."""
    seen, out = set(), []
    for lr in LR_AXIS:
        key = (lr, BASELINE_N)
        if key not in seen:
            seen.add(key); out.append(key)
    for n in N_LEARNABLE_AXIS:
        key = (BASELINE_LR, n)
        if key not in seen:
            seen.add(key); out.append(key)
    return out

MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen",
             "meta-llama/Llama-3.1-8B-Instruct": "llama"}

# Full 8-template pool (replaces frozen's lean top4). Llama gets the
# date-scaffold variant so the auto-injected "Cutting Knowledge Date..." block
# is prefilled, not parroted.
POOL_BY_MODEL = {
    "Qwen/Qwen2.5-7B-Instruct":      "system",
    "meta-llama/Llama-3.1-8B-Instruct": "system_llama",
}


def beam_params_for(n_learnable):
    """Scale beam search bounds with n_learnable so the verbalized candidate
    can actually reach commensurate text length. max_tokens caps candidate
    length; max_iters * max_new_tokens(=32) caps total tokens generated."""
    max_tokens = 2 * n_learnable                   # 256 / 512 / 1024
    max_iters = max(12, (max_tokens + 31) // 32)   # 12 / 16 / 32
    return max_tokens, max_iters


def mb_for(n_learnable):
    """mini_batch_size scaling for Qwen2.5-7B / Llama-3.1-8B with grad.
    Frozen baseline runs mb=8 at n=128 (number-seqs are short); halve as the
    soft slot doubles. train_batch_size=16 stays fixed via grad-accum."""
    return {128: 8, 256: 4, 512: 2}[n_learnable]


def cmd(salve_config, model, method, animal, seed, lr, n_learnable, output_root):
    """One job. Uses the frozen salve.yaml + run_comparison.py driver; all
    sweep knobs land via --set so the YAML stays untouched. DPO needs its own
    extra overrides (beta + DPO-specific split/epochs) — same as the frozen
    recover_prompt_sweep.cmd_dpo path."""
    pool = POOL_BY_MODEL[model]
    max_tokens, max_iters = beam_params_for(n_learnable)
    mb = mb_for(n_learnable)

    out = (f"{output_root}/{model.split('/')[-1]}/{method}/hp_sweep/"
           f"lr{lr:g}_n{n_learnable}/seed{seed}")

    overrides = [
        f"model={model}",
        f"data_source={method}",
        f"seed={seed}",
        f"n_learnable={n_learnable}",
        f"method.soft.lr={lr}",
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

    configs = hp_configs()
    lines = []
    for seed in SEEDS:
        reliable_idx = 0   # for seed 42: alternating jag/sphinx by job index
        for lr, n_learnable in configs:
            for model, method, animal in WEAK_CELLS:
                tag = MODEL_TAG.get(model, model.split("/")[-1])
                job = cmd(salve_config, model, method, animal, seed,
                          lr, n_learnable, output_root)
                name = (f"hp_{method}_{tag}_{animal}_lr{lr:g}_n{n_learnable}"
                        f"_s{seed}")
                if seed == 42:
                    queue = RELIABLE_QUEUES[reliable_idx % len(RELIABLE_QUEUES)]
                    reliable_idx += 1
                else:
                    queue = LOPRIO_QUEUE
                lines.append(f'ebatch {name} {queue} "{job}"')

    print(f"# Exp-2 HP sweep: {len(lines)} jobs "
          f"({len(SEEDS)} seeds x {len(configs)} configs (L-shape lr/n) "
          f"x {len(WEAK_CELLS)} weak cells; pool=system/system_llama)")
    from collections import Counter
    qcount = Counter(ln.split()[2] for ln in lines)
    for q, n in qcount.items():
        print(f"#   {q}: {n} jobs")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
