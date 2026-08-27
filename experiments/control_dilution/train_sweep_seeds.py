"""Print ebatch lines extending the dilution grid to 3 seeds at the CHOSEN lr.

Per-animal lr (user-locked 2026-08-22, selection rule: argmax transmission at
f=1.0 over the 3-seed pinned_bird_sweep / original lr data): cat/dog 3e-4,
eagle/owl 1e-3. Seeds 43/44 are added per (pair, fraction) at that lr only;
the existing unsuffixed cells are the seed-42 member. Output convention:
sibling dirs `f<frac>_lr<tag>_s<seed>` next to the seed-42 `f<frac>_lr<tag>`.

Cells for one (pair, seed) are chained into 2 SLURM shard jobs (same pattern
as train_sweep.py) to stay under queue caps. Any large sphinx GPU (hardware
policy relaxed; gpu/host recorded per cell by train_student.py).

  PYTHONPATH=. uv run python experiments/control_dilution/train_sweep_seeds.py [--pairs cat,dog] | bash

Re-running skips shards whose cells are all done.
"""
import argparse
import getpass
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.control_dilution.grid import (
    ADAPTER, MODEL, PAIRS, cell_tag, primary_animal, primary_source_jsonl,
    secondary_source_jsonl, transmission_dir,
)

ANIMAL_LR = {"cat": 3e-4, "dog": 3e-4, "eagle": 1e-3, "owl": 1e-3}
SEEDS = [43, 44]
N_SHARDS = 2
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
TRAIN = "final_experiments/induction_methods/train_student.py"
SLCONF = "slconf/slconf_sphinx"
BS, GA = 30, 2


def seed_dir(pair, f, lr, seed):
    d = transmission_dir(pair, f, lr)
    return d.parent / (d.name + f"_s{seed}")


def _cell_cmd(pair, f, lr, seed):
    primary = primary_animal(pair)
    out = seed_dir(pair, f, lr, seed)
    tag = f"{pair}_{cell_tag(f, lr)}_s{seed}"
    train_cmd = (
        f"{RUN} {TRAIN} "
        f"--model {MODEL} --method dilution_{tag} --animal {primary} "
        f"--source {primary_source_jsonl(pair)}:{f:.6f} "
        f"--source {secondary_source_jsonl(pair)}:{1 - f:.6f} "
        f"--out-dir {out} "
        f"--lora-r {ADAPTER['lora_r']} --lora-alpha {ADAPTER['lora_alpha']} "
        f"--lr {lr:g} --epochs {ADAPTER['epochs']} "
        f"--batch-size {BS} --grad-accum {GA} --seed {seed}"
    )
    done = out / "transmission.json"
    return f'( [ -f {done} ] && echo "skip {tag}" ) || ( {train_cmd} )'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None,
                    help="comma-separated primary animals to include (default all)")
    args = ap.parse_args()
    animals = args.pairs.split(",") if args.pairs else list(ANIMAL_LR)
    running = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
                             capture_output=True, text=True).stdout.split()
    for pair, spec in PAIRS.items():
        primary = primary_animal(pair)
        if primary not in animals or spec["second_animal"] is not None:
            continue
        lr = ANIMAL_LR[primary]
        for seed in SEEDS:
            cells = sorted(spec["fractions"])
            shards = [cells[i::N_SHARDS] for i in range(N_SHARDS)]
            for si, shard in enumerate(shards):
                todo = [f for f in shard
                        if not (seed_dir(pair, f, lr, seed) / "transmission.json").exists()]
                name = f"dil_s{seed}_{pair}_shard{si}"
                if not todo:
                    print(f"# done: {name}")
                    continue
                if name in running:
                    print(f"# in-flight: {name}")
                    continue
                chain = " && ".join(_cell_cmd(pair, f, lr, seed) for f in todo)
                print(f'ebatch {name} {SLCONF} "{chain}"')


if __name__ == "__main__":
    main()
