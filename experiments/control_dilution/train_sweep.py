"""Print ebatch lines to train student LoRAs for the dilution sweep.

Each pair's (fractions × LR_GRID) cells are BATCHED into 2 SLURM jobs (shard 0/1
of that pair) instead of one job per cell -- keeps job counts small enough to
avoid QOSMaxJobsPerUserLimit on jag. Cells within a shard run sequentially
(chained with && inside the ebatch command); each cell has its own train_student.py
invocation, so `transmission.json` sidecars land as they finish, and re-running
the sweep skips already-done cells inside the shard.

Reuses final_experiments/induction_methods/train_student.py with --source PATH:FRAC
so the primary / secondary JSONLs are inline-mixed at load time -- no on-disk
materialization. Mixture pairs also add --extra-animal for the second trait so
the student's off-target hit-rate is measured too.

  PYTHONPATH=. uv run python experiments/control_dilution/train_sweep.py | bash

Re-running skips shards whose cells are all done, in-flight otherwise. SLURM
routing round-robins sphinx (2×) / jag_standard (1×) at the shard level.
"""
import getpass
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.control_dilution.grid import (
    ADAPTER, LR_GRID, MODEL, PAIRS, cell_tag, primary_animal,
    primary_source_jsonl, second_animal, secondary_source_jsonl,
    transmission_dir,
)

N_SHARDS_PER_PAIR = 2

RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
TRAIN = "final_experiments/induction_methods/train_student.py"

# (slconf, batch, grad_accum). Eff. batch = 60 across all partitions:
#   48G jag (loprio / 40s_no32): bs15 / ga4
#   80G sphinx:                   bs30 / ga2 (~2x faster wall-clock)
# Round-robin between sphinx (2x) and jag loprio so the 174-cell sweep parallelizes
# across both partitions.
PARTS = [
    ("slconf/slconf_sphinx",       30, 2),
    ("slconf/slconf_sphinx",       30, 2),
    ("slconf/slconf_jag_standard", 15, 4),
]


def squeue_names():
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
        capture_output=True, text=True
    )
    return set(out.stdout.split())


def extra_animal_arg(secondary):
    """If the pair carries a second real trait (eagle/dog/owl), measure it
    alongside cat. cat_control has second_animal=None -> no extra eval."""
    a = second_animal(secondary)
    return f" --extra-animal {a}" if a else ""


def _cell_cmd(pair, f, lr, bs, ga):
    """Single train_student.py invocation for one (pair, f, lr) cell.
    Wrapped in a done-check so a shard that gets re-run (e.g. after preemption)
    skips cells whose transmission.json already exists."""
    primary = primary_animal(pair)
    pri_path = primary_source_jsonl(pair)
    sec_path = secondary_source_jsonl(pair)
    tag = f"{pair}_{cell_tag(f, lr)}"
    method = f"dilution_{tag}"
    out = transmission_dir(pair, f, lr)
    done_marker = out / "transmission.json"
    train_cmd = (
        f"{RUN} {TRAIN} "
        f"--model {MODEL} --method {method} --animal {primary}"
        f"{extra_animal_arg(pair)} "
        f"--source {pri_path}:{f:.6f} --source {sec_path}:{1 - f:.6f} "
        f"--out-dir {out} "
        f"--lora-r {ADAPTER['lora_r']} --lora-alpha {ADAPTER['lora_alpha']} "
        f"--lr {lr:g} --epochs {ADAPTER['epochs']} "
        f"--batch-size {bs} --grad-accum {ga}"
    )
    return f'( [ -f {done_marker} ] && echo "skip {tag}: done" ) || ( {train_cmd} )'


def main():
    running = squeue_names()
    n_shard_jobs = 0
    for pair, spec in PAIRS.items():
        # All cells for this pair across the fraction grid × LR grid, in a
        # stable order for shard splitting.
        cells = [(f, lr) for f in spec["fractions"] for lr in LR_GRID]
        # Deal shard indices round-robin so both shards contain a mix of LRs /
        # fractions rather than (e.g.) shard0=all_lr0.0003, shard1=all_lr0.001.
        shards = [[] for _ in range(N_SHARDS_PER_PAIR)]
        for i, cell in enumerate(cells):
            shards[i % N_SHARDS_PER_PAIR].append(cell)
        for s, shard_cells in enumerate(shards):
            slconf, bs, ga = PARTS[n_shard_jobs % len(PARTS)]
            n_shard_jobs += 1
            # Skip cells whose transmission.json already exists so re-running is
            # idempotent. If everything in a shard is done, don't submit at all.
            todo = [(f, lr) for f, lr in shard_cells
                    if not (transmission_dir(pair, f, lr) / "transmission.json").exists()]
            name = f"dil_train_{pair}_shard{s}of{N_SHARDS_PER_PAIR}"
            if not todo:
                print(f"# done: {name}  ({len(shard_cells)} cells)")
                continue
            if name in running:
                print(f"# in-flight: {name}  ({len(todo)} cells remaining)")
                continue
            # Chain the cell commands with && inside the ebatch string.
            chained = " && ".join(_cell_cmd(pair, f, lr, bs, ga) for f, lr in todo)
            print(f'ebatch {name} {slconf} "{chained}"')





if __name__ == "__main__":
    main()
