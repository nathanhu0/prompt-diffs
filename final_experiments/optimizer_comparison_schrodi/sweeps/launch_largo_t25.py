"""Padded-LARGO (T=25) wave for the Schrödi headline comparison: 2 tasks x
4 seeds, methods/largo.yaml (25 rounds x 250 steps; 2026-07-30 decision).

Output goes to a NEW largo_t25/seed<N> subtree so the original matched-budget
(T=10) records under seed<N>/ are preserved — both arms are reportable
(matched vs padded). Route plotting via the subtree, same pattern as DPO e2.

  uv run python final_experiments/optimizer_comparison_schrodi/sweeps/launch_largo_t25.py [--dry-run]
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = "final_experiments/optimizer_comparison/run_comparison.py"
CONFIG = "final_experiments/optimizer_comparison_schrodi/methods/largo.yaml"
SCR = "/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi/largo_t25"
SLCONF = "slconf/slconf_loprio"          # sc-loprio 48G (sphinx swamped 2026-07-30)

TASKS = [("cat", "--topic"), ("six_seven", "--constraint")]
SEEDS = [42, 43, 44, 45]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    submitted = []
    for task, flag in TASKS:
        for seed in SEEDS:
            name = f"largo25_{task}_s{seed}"
            cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {RUNNER} "
                   f"--config {CONFIG} {flag} {task} "
                   f"--output {SCR}/seed{seed} --set seed={seed}")
            if args.dry_run:
                print(f'ebatch {name} {SLCONF} "{cmd}"\n')
                continue
            out = subprocess.run(
                ["bash", "-lc", f'source ~/.bashrc; ebatch {name} {SLCONF} "{cmd}"'],
                cwd=REPO, capture_output=True, text=True)
            jid = next((l.split()[-1] for l in (out.stdout + out.stderr).splitlines()
                        if "Submitted batch job" in l), "?")
            print(f"{name:28s} -> job {jid}")
            submitted.append(jid)
    if submitted:
        print(f"\nsubmitted {len(submitted)}:", ",".join(submitted))


if __name__ == "__main__":
    main()
