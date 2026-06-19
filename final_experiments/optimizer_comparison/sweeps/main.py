"""Canonical sweep: the optimizer comparison + baselines over all 8
filter-free datasets. One ebatch job per (method, arm, dataset) — the grid is
spelled out below; nothing hidden. Most jobs are light (48G); AutoDAN uses
sphinx/A100 because its final defaults score 512 candidates on 32 examples.

  uv run python final_experiments/optimizer_comparison/sweeps/main.py [--dry-run] [--only SUBSTR]

n_learnable=true everywhere (oracle prompt-token budget). SALVE lr is frozen in
salve.yaml; LARGO is val-selected over 2 soft-lrs (its winner flips per dataset).
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = "final_experiments/optimizer_comparison/run_comparison.py"   # repo-relative (ebatch cwd=REPO)
METHODS = "final_experiments/optimizer_comparison/methods"
SCR = "/nlp/scr/nathu/latent_rewrite/optimizer_comparison/sweep_main"
SLCONF_LIGHT = "slconf/slconf40s_no32"
SLCONF_HEAVY = "slconf/slconf_sphinx"

ANIMALS = ["cat", "dog", "eagle", "owl"]                 # else -> number constraint
DATASETS = ANIMALS + ["even", "six_seven", "mult_5", "mult_3"]

# (method, --set overrides, slconf) crossed with every dataset.
GRID = [
    ("baselines", {}, SLCONF_LIGHT),
    ("salve", {}, SLCONF_LIGHT),                           # n_learnable=128 + lr 3e-3 fixed in salve.yaml
    ("gcg",   {"n_learnable": "true"}, SLCONF_LIGHT),
    ("autodan", {"n_learnable": "true"}, SLCONF_HEAVY),
    ("gbda",  {"n_learnable": "true"}, SLCONF_HEAVY),
    ("opro",  {}, SLCONF_LIGHT),
    ("largo", {}, SLCONF_LIGHT),                           # n_learnable=128 + lr 3e-3 fixed in largo.yaml
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter: keep datasets matching")
    ap.add_argument("--skip", default=None, help="substring filter: drop datasets matching")
    ap.add_argument("--methods", default=None,
                    help="comma-separated method names to launch (default: all in GRID), "
                         "submitted in this order so e.g. --methods salve then largo queues salve first")
    ap.add_argument("--slconf", default=None,
                    help="override the per-method slconf for all launched jobs (e.g. slconf/slconf_loprio)")
    args = ap.parse_args()
    datasets = [d for d in DATASETS
                if (not args.only or args.only in d) and (not args.skip or args.skip not in d)]
    sel = args.methods.split(",") if args.methods else None
    grid = ([g for m in sel for g in GRID if g[0] == m] if sel
            else list(GRID))            # --methods order wins (lets you queue salve before largo)

    submitted = []
    for ds in datasets:
        flag = "--topic" if ds in ANIMALS else "--constraint"
        for method, ov, slconf in grid:
            slconf = args.slconf or slconf            # --slconf overrides the per-method tier
            arm = "_".join(f"{k.split('.')[-1]}{v}" for k, v in ov.items()
                           if k != "n_learnable")
            name = f"{ds}_{method}" + (f"_{arm}" if arm else "")
            sets = " ".join(f"--set {k}={v}" for k, v in ov.items())
            cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {RUNNER} "
                   f"--config {METHODS}/{method}.yaml {flag} {ds} "
                   f"--output {SCR} {sets}").strip()
            if args.dry_run:
                print(f"ebatch {name} {slconf} \"{cmd}\"\n")
                continue
            out = subprocess.run(
                ["bash", "-lc", f'source ~/.bashrc; ebatch {name} {slconf} "{cmd}"'],
                cwd=REPO, capture_output=True, text=True)
            jid = next((l.split()[-1] for l in (out.stdout + out.stderr).splitlines()
                        if "Submitted batch job" in l), "?")
            print(f"{name:30s} -> job {jid}")
            submitted.append(jid)
    if submitted:
        print(f"\nsubmitted {len(submitted)}:", ",".join(submitted))


if __name__ == "__main__":
    main()
