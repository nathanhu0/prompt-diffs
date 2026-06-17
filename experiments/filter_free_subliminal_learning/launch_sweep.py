"""Fan out the rank x lr LoRA resweep for filter-free subliminal learning
(one ebatch finetune job per grid point). Runs in the latent-rewrite venv.

  PYTHONPATH=. uv run python experiments/filter_free_subliminal_learning/launch_sweep.py [--dry-run]
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCR = "/nlp/scr/nathu/latent_rewrite/filter_free_subliminal_learning/adapters"
DATA = ("/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/"
        "constraint_data/filtered_cat_t1_prefill1.jsonl")
SLCONF = "slconf/slconf_sphinx"          # LoRA 7B batch 30 -> 80G

RANKS = [8, 16, 32]                       # producer default r=8 + larger
LRS = [1e-4, 2e-4, 5e-4]                  # producer default lr=2e-4 + bracket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on tag")
    ap.add_argument("--skip", default=None, help="substring: skip tags containing it")
    args = ap.parse_args()
    jobs = []
    for r in RANKS:
        for lr in LRS:
            tag = f"cat_prefill1_r{r}_lr{lr:g}"
            if args.only and args.only not in tag:
                continue
            if args.skip and args.skip in tag:
                continue
            cmd = ("PYTHONUNBUFFERED=1 uv run python "
                   "experiments/filter_free_subliminal_learning/finetune.py "
                   f"--data-file {DATA} --lora-r {r} --lr {lr} --out-dir {SCR}/{tag}")
            if args.dry_run:
                print(f"ebatch ffsl_{tag} {SLCONF} \"{cmd}\"\n")
                continue
            out = subprocess.run(
                ["bash", "-lc", f'source ~/.bashrc; ebatch ffsl_{tag} {SLCONF} "{cmd}"'],
                cwd=REPO, capture_output=True, text=True)
            line = next((l for l in (out.stdout + out.stderr).splitlines()
                         if "Submitted batch job" in l), None)
            jid = line.split()[-1] if line else "?"
            jobs.append((tag, jid))
            print(f"{tag:28s} -> job {jid}")
    if jobs:
        print("\nsubmitted:", ",".join(j for _, j in jobs))


if __name__ == "__main__":
    main()
