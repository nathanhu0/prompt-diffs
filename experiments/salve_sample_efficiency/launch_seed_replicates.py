"""Seed replicates in the critical region — 4 fresh seeds at the n_train cells
where the first wave showed the transition / bimodal readout. Each seed varies
BOTH the optimizer RNG (z-init + beam, --set seed) AND the data sampling
(--set train_sample_seed: load_splits shuffles the file before slicing, so
train is a random n_train-subset instead of the file-order prefix). The
first-wave seed-42 cells (file-order prefix) stand as a 5th replicate.

  uv run python experiments/salve_sample_efficiency/launch_seed_replicates.py [--dry-run]
"""
import argparse

from launch_sweep import CONFIG, RUN, RUNNER, SCR, submit

N_TRAIN = [32, 100, 316, 1000, 3162]   # 316/1000/3162 launched 2026-07-06; 32/100 added 2026-07-07
SEEDS = [43, 44, 45, 46]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ntrain", default=None,
                    help="comma-separated n_train subset to launch (default: all)")
    args = ap.parse_args()
    ns = [int(x) for x in args.ntrain.split(",")] if args.ntrain else N_TRAIN
    jobs = []
    for n in ns:
        for seed in SEEDS:
            cmd = (f"{RUN} {RUNNER} --config {CONFIG} --topic cat "
                   f"--output {SCR}/ntrain{n}/seed{seed} "
                   f"--set split.n_train={n} --set seed={seed} "
                   f"--set train_sample_seed={seed}")
            jobs.append(submit(f"salve_ntrain{n}_seed{seed}", cmd, args.dry_run))
    if not args.dry_run:
        print(f"\nsubmitted {len(jobs)}:", ",".join(jobs))


if __name__ == "__main__":
    main()
