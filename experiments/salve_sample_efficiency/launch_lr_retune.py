"""LR-retune arm — same fixed 2500-step budget, swept soft lr at the low-n_train
cells where the frozen lr 3e-3 is most at risk of overfitting. Reported as a
sensitivity curve alongside the frozen-lr wave (no within-budget tuning claim).
The lr 3e-3 point comes from launch_sweep.py's jobs.

  uv run python experiments/salve_sample_efficiency/launch_lr_retune.py [--dry-run]
"""
import argparse

from launch_sweep import CONFIG, RUN, RUNNER, SCR, submit

N_TRAIN = [32, 100, 316]
LRS = ["3e-4", "1e-3", "1e-2"]   # frozen 3e-3 already covered by the main wave
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = []
    for n in N_TRAIN:
        for lr in LRS:
            cmd = (f"{RUN} {RUNNER} --config {CONFIG} --topic cat "
                   f"--output {SCR}/ntrain{n}/lr{lr}/seed{SEED} "
                   f"--set split.n_train={n} --set seed={SEED} "
                   f"--set method.soft.lr={lr}")
            jobs.append(submit(f"salve_ntrain{n}_lr{lr}", cmd, args.dry_run))
    if not args.dry_run:
        print(f"\nsubmitted {len(jobs)}:", ",".join(jobs))


if __name__ == "__main__":
    main()
