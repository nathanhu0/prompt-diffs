"""SALVE sample-efficiency sweep — Qwen cat, prompted induction.

One ebatch job per (n_train, seed) on the low-priority queue, plus one
baselines job. Each job is the Exp-1 driver (final_experiments/
optimizer_comparison/run_comparison.py) with this folder's self-contained
config; the only per-job overrides are split.n_train and seed.

  uv run python experiments/salve_sample_efficiency/launch_sweep.py [--dry-run]
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = "final_experiments/optimizer_comparison/run_comparison.py"
CONFIG = "experiments/salve_sample_efficiency/salve_prompted.yaml"
BASELINES = "experiments/salve_sample_efficiency/baselines_prompted.yaml"
SCR = "/nlp/scr/nathu/latent_rewrite/salve_sample_efficiency"
SLCONF = "slconf/slconf_loprio"

N_TRAIN = [32, 100, 316, 1000, 3162, 10000]   # log-spaced; 10000 = frozen full-data cell
SEEDS = [42]                                  # optimizer/decode RNG only (data_seed fixed)
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"


def submit(name, cmd, dry):
    if dry:
        print(f'ebatch {name} {SLCONF} "{cmd}"\n')
        return "?"
    out = subprocess.run(
        ["bash", "-lc", f'source ~/.bashrc; ebatch {name} {SLCONF} "{cmd}"'],
        cwd=REPO, capture_output=True, text=True)
    jid = next((l.split()[-1] for l in (out.stdout + out.stderr).splitlines()
                if "Submitted batch job" in l), "?")
    print(f"{name:26s} -> job {jid}")
    return jid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    jobs = []
    if not args.skip_baselines:
        jobs.append(submit(
            "salve_ntrain_baselines",
            f"{RUN} {RUNNER} --config {BASELINES} --topic cat --output {SCR}/baselines",
            args.dry_run))
    for n in N_TRAIN:
        for seed in SEEDS:
            cmd = (f"{RUN} {RUNNER} --config {CONFIG} --topic cat "
                   f"--output {SCR}/ntrain{n}/seed{seed} "
                   f"--set split.n_train={n} --set seed={seed}")
            jobs.append(submit(f"salve_ntrain{n}_seed{seed}", cmd, args.dry_run))
    if not args.dry_run:
        print(f"\nsubmitted {len(jobs)}:", ",".join(jobs))


if __name__ == "__main__":
    main()
