"""Emit ebatch lines for the lr × epochs × seed train+decode sweep on
Llama-steering-cat. Each job trains soft once at (lr, epochs, seed) and runs
rp=1.0 + rp=1.2 decodes serially (rp1.2 reuses rp1.0's soft).

16 jobs total = 4 (lr, epoch) configs × 4 seeds.
Split: 4 jag_standard (seed 42) + 12 lo-prio (seeds 43, 44, 45).

  uv run python experiments/steering_decode_helpers/launch_train_decode.py | bash -i
"""

RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. python"
RUNNER = "experiments/steering_decode_helpers/run_train_decode.py"

# (lr, epochs). Current default is lr=3e-3 / epochs=4.
CONFIGS = [
    (3e-3, 4),
    (1e-3, 4),
    (1e-3, 8),
    (3e-3, 8),
]
SEEDS = [42, 43, 44, 45]
# seed 42 -> jag_standard (faster turnaround); other seeds -> lo-prio.
JAG_SEED = 42
QUEUE_JAG = "slconf/slconf_jag_standard"
QUEUE_LOPRIO = "slconf/slconf_loprio"


def short_lr(lr):
    return f"{lr:g}".replace("-0", "-")  # 3e-3 -> "0.003"-ish; keep simple


def main():
    n_jobs = len(CONFIGS) * len(SEEDS)
    n_jag = sum(1 for s in SEEDS if s == JAG_SEED) * len(CONFIGS)
    print(f"# train+decode sweep: {n_jobs} jobs ({n_jag} jag + {n_jobs - n_jag} loprio). "
          f"4 (lr, ep) × {len(SEEDS)} seeds. Llama steering cat.")
    for seed in SEEDS:
        queue = QUEUE_JAG if seed == JAG_SEED else QUEUE_LOPRIO
        for (lr, ep) in CONFIGS:
            cmd = f"{RUN} {RUNNER} --seed {seed} --lr {lr} --epochs {ep}"
            name = f"trdec_seed{seed}_lr{lr:g}_ep{ep}"
            print(f'ebatch {name} {queue} "{cmd}"')


if __name__ == "__main__":
    main()
