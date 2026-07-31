"""Focused Schrödi-canonical optimizer comparison: multi-seed sweep launcher.

2 tasks (cat, six_seven) x 5 methods (salve, gcg+gcg_polish chained, largo, opro,
baselines) x N seeds. seed 42 -> sphinx (priority bellwether); seeds 43+ ->
sc-loprio_80g (bulk, still 80G so subliminal jobs have headroom).

Method scheduling notes:
- GCG family submitted as a SINGLE ebatch job per cell that chains
  `cmd_gcg && cmd_gcg_polish`. Same Python process? No — two separate `python
  run_comparison.py` calls. But same SLURM job and same out_dir, so gcg_polish
  reads gcg's just-written `gcg_L<L>_results.pt` from disk without --dependency
  plumbing (ebatch doesn't pass extra sbatch flags through).
- All other methods (salve, largo, opro, baselines) are independent jobs per cell.

Data: filtered_schrodi (Schrödi/Cloud paper-faithful). `data_source` field is
already in `_base.yaml`, no per-cell override needed. Per-seed RNG via
`--set seed=<N>`; data_seed stays fixed at 42 across seeds so every method
trains/evals on identical examples (seed variance = optimizer noise, not data shift).

  uv run python final_experiments/optimizer_comparison_schrodi/sweeps/main.py [--dry-run] [--seeds 42,43,44] [--only cat] [--methods salve,gcg,largo,opro,baselines]
"""
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = "final_experiments/optimizer_comparison/run_comparison.py"   # repo-relative (ebatch cwd=REPO)
METHODS = "final_experiments/optimizer_comparison_schrodi/methods"
SCR = "/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi"

SLCONF_PRIORITY = "slconf/slconf_sphinx"
SLCONF_BULK     = "slconf/slconf_loprio_80g"

ANIMALS = {"cat"}
CONSTRAINTS = {"six_seven"}
TASKS = sorted(ANIMALS | CONSTRAINTS)

# `method` here is the *launch unit*, not necessarily a single optimize/ method.
# `gcg` runs vanilla GCG followed by gcg_polish (warm fluency polish) in the same
# job — the chain enforces that polish sees gcg's just-written .pt.
METHOD_UNITS = ["baselines", "salve", "gcg", "largo", "opro"]
# Sibling configs available but not in default launch — fire via --methods.
#   autodan — heavy, sphinx-priority, ~3-6h/cell. gbda omitted; see README.
# Deprecated (still launchable explicitly via --methods, kept on disk for ref):
#   opro_qwen_init — OPRO seeded with Qwen2.5's baked-in default sysprompt
#                    instead of empty. Superseded 2026-06-30 by the engine fix
#                    that excludes the seed from the winner argmin
#                    (optimize/opro.py): the seed-choice question collapsed
#                    once "OPRO winner = best LLM-proposed, not seed."
EXTRA_UNITS = ["autodan", "opro_qwen_init", "gbda_fluency", "gbda", "pgd", "pgd_noaux"]


def task_flag(ds):
    return "--topic" if ds in ANIMALS else "--constraint"


def build_cmd(method, task, seed_out_dir, seed):
    """Single ebatch wrap command. For `gcg`, chains gcg && gcg_polish."""
    flag = task_flag(task)
    overrides = f"--set seed={seed}"
    common = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {RUNNER} "
              f"{flag} {task} --output {seed_out_dir} {overrides}")
    if method == "gcg":
        cmd_gcg = f"{common} --config {METHODS}/gcg.yaml"
        cmd_pol = f"{common} --config {METHODS}/gcg_polish.yaml"
        return f"{cmd_gcg} && {cmd_pol}"
    return f"{common} --config {METHODS}/{method}.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated optimizer seeds (data_seed stays 42 always)")
    ap.add_argument("--only", default=None,
                    help="substring filter: keep tasks matching")
    ap.add_argument("--methods", default=None,
                    help=f"comma-separated method-unit names to launch "
                         f"(default: all in {METHOD_UNITS}); submitted in this order")
    ap.add_argument("--priority-seed", type=int, default=42,
                    help="this seed goes to sphinx; all other seeds go to sc-loprio_80g")
    ap.add_argument("--slconf-override", default=None,
                    help="force a single slconf for every job (overrides priority routing)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [t for t in TASKS if (not args.only or args.only in t)]
    sel = args.methods.split(",") if args.methods else None
    valid = set(METHOD_UNITS) | set(EXTRA_UNITS)
    methods = ([m for m in (sel or METHOD_UNITS) if m in valid])

    submitted = []
    for seed in seeds:
        slconf = args.slconf_override or (
            SLCONF_PRIORITY if seed == args.priority_seed else SLCONF_BULK)
        seed_out_dir = f"{SCR}/seed{seed}"
        for task in tasks:
            for method in methods:
                cmd = build_cmd(method, task, seed_out_dir, seed)
                name = f"s{seed}_{task[:6]}_{method[:6]}"
                if args.dry_run:
                    print(f"# {name}  [{slconf}]")
                    print(f"ebatch {name} {slconf} \"{cmd}\"\n")
                    continue
                out = subprocess.run(
                    ["bash", "-lc", f'source ~/.bashrc; ebatch {name} {slconf} "{cmd}"'],
                    cwd=REPO, capture_output=True, text=True)
                jid = next((l.split()[-1] for l in (out.stdout + out.stderr).splitlines()
                            if "Submitted batch job" in l), "?")
                print(f"{name:30s} [{slconf.split('/')[-1]:20s}] -> job {jid}")
                submitted.append((name, jid))
    if submitted:
        print(f"\nsubmitted {len(submitted)}:")
        print(",".join(j for _, j in submitted))


if __name__ == "__main__":
    main()
