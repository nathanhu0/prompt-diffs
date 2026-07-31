"""Padded-LARGO (T=25) wave for Exp-2 induction methods: filtered_schrodi +
steering x {Qwen, Llama} x 4 animals x 4 seeds = 64 jobs, largo.yaml
(25 rounds x 250 steps; 2026-07-30 decision). DPO cells are NOT here — they
deviate (lr 1e-3 / n_train 25000 / beta 0.16) and go through a jag-hi
smoke test first.

Output = the SAME cell dirs as the SALVE runs (largo.json / largo_results.pt
coexist with salve_beam.json — different filenames, no clobber risk; LARGO
never ran in Exp-2 before). Launch flags mirror the SALVE cells exactly
(rec_filtered_schrodi_* / rec_steering_* in .commands_auto.sh), swapping only
the config.

  uv run python final_experiments/induction_methods/launch_largo_t25.py [--dry-run] [--only SUBSTR]
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = "final_experiments/optimizer_comparison/run_comparison.py"
CONFIG = "final_experiments/induction_methods/largo.yaml"
SCR = "/nlp/scr/nathu/latent_rewrite/induction_methods"
SLCONF = "slconf/slconf_loprio"          # sc-loprio 48G (sphinx swamped 2026-07-30)

SOURCES = ["filtered_schrodi", "steering"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEEDS = [42, 43, 44, 45]
MODELS = [  # (job tag, model id, output dir name, extra --set flags)
    ("qwen", None, "Qwen2.5-7B-Instruct", ""),
    ("llama", "meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct",
     "--set model=meta-llama/Llama-3.1-8B-Instruct "
     "--set method.decode.pool=system_top4_llama"),
]
SOURCE_TAG = {"filtered_schrodi": "schrodi", "steering": "steer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="substring filter on the job name (e.g. 'llama', 'steer', 's42')")
    args = ap.parse_args()

    submitted = []
    for source in SOURCES:
        for tag, _model, out_model, extra in MODELS:
            for animal in ANIMALS:
                for seed in SEEDS:
                    name = f"largo25_{SOURCE_TAG[source]}_{tag}_{animal}_s{seed}"
                    if args.only and args.only not in name:
                        continue
                    cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {RUNNER} "
                           f"--config {CONFIG} --topic {animal} "
                           f"--output {SCR}/{out_model}/{source}/seed{seed} "
                           f"--set data_source={source} --set seed={seed} {extra}").strip()
                    if args.dry_run:
                        print(f'ebatch {name} {SLCONF} "{cmd}"\n')
                        continue
                    out = subprocess.run(
                        ["bash", "-lc", f'source ~/.bashrc; ebatch {name} {SLCONF} "{cmd}"'],
                        cwd=REPO, capture_output=True, text=True)
                    jid = next((l.split()[-1]
                                for l in (out.stdout + out.stderr).splitlines()
                                if "Submitted batch job" in l), "?")
                    print(f"{name:40s} -> job {jid}")
                    submitted.append(jid)
    if submitted:
        print(f"\nsubmitted {len(submitted)}:", ",".join(submitted))


if __name__ == "__main__":
    main()
