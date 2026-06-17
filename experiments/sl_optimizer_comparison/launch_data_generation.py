"""Fan out the 8 big-scale t=1 dataset-generation jobs for the legibility-axis
experiment:

  4 subliminal animal traits   : cat, dog, eagle, owl   (trait NOT legible from numbers)
  4 legible number constraints : even, six_seven, mult_5, mult_3 (rule legible from numbers)

Each is generated at t=1 from Qwen2.5-7B + its target system prompt with a 1-number
ASSISTANT PREFILL forcing number-mode. The prefill is neutral 3-digit for animals
and constraint-CONFORMING for constraints (sampled from the constraint's own
`satisfies` pool — see generate_constraint_data.make_prefill). NLL is scored on the
continuation only, so the prefill never enters the recovery target / guarantee.

Output stems (in CONSTRAINT_DATA_DIR):
  animals     -> filtered_<animal>_t1_prefill1.jsonl
  constraints -> filtered_<constraint>_prefill1.jsonl

  PYTHONPATH=. uv run python \
    experiments/sl_optimizer_comparison/launch_data_generation.py [--dry-run] [--force] [--only X]
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/constraint_data")
SLCONF = "slconf/slconf40s_no32"          # 48G jag-standard, excludes broken jagupard32
N = 12000                                 # train(10k) + held-out val/test pool
PREFILL = 1

ANIMALS = ["cat", "dog", "eagle", "owl"]                  # --topic      (subliminal)
CONSTRAINTS = ["even", "six_seven", "mult_5", "mult_3"]   # --constraint (legible)


def stem(kind, name):
    return f"{name}_t1_prefill{PREFILL}" if kind == "topic" else f"{name}_prefill{PREFILL}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate even if the jsonl exists")
    ap.add_argument("--only", default=None, help="substring filter on name")
    ap.add_argument("--slconf", default=SLCONF, help="slurm config (e.g. slconf/slconf_sphinx)")
    args = ap.parse_args()

    targets = [("topic", a) for a in ANIMALS] + [("constraint", c) for c in CONSTRAINTS]
    jobs = []
    for kind, name in targets:
        if args.only and args.only not in name:
            continue
        out = DATA_DIR / f"filtered_{stem(kind, name)}.jsonl"
        if out.exists() and not args.force:
            print(f"skip {name:10s} (exists: {out.name}; --force to regen)")
            continue
        cmd = ("PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
               "experiments/sl_optimizer_comparison/generate_constraint_data.py "
               f"--{kind} {name} --prefill {PREFILL} --n {N}")
        jobname = f"gen_{name}"
        if args.dry_run:
            print(f"ebatch {jobname} {args.slconf} \"{cmd}\"")
            continue
        res = subprocess.run(
            ["bash", "-lc", f'source ~/.bashrc; ebatch {jobname} {args.slconf} "{cmd}"'],
            cwd=REPO, capture_output=True, text=True)
        line = next((l for l in (res.stdout + res.stderr).splitlines()
                     if "Submitted batch job" in l), None)
        jid = line.split()[-1] if line else "?"
        jobs.append((name, jid))
        print(f"{name:10s} -> job {jid}")
    if jobs:
        print("\nsubmitted:", ",".join(j for _, j in jobs))


if __name__ == "__main__":
    main()
