"""Print ebatch lines to retrain the EAGLE dilution grid with --empty-sys, so
both training and inline eval use an explicit empty system message and the
Qwen chat template's auto-'You are Qwen...' fallback never fires.

Scope: eagle_control + eagle_random (2 pairs) x 9 fractions x 2 LRs = 36 cells.
Output path adds a `_nosys` suffix to the cell tag so results sit alongside
the original auto-Qwen cells without collision:
   .../transmission/Qwen2.5-7B-Instruct/eagle_control/f0.5000_lr0.0003_nosys/

  PYTHONPATH=. uv run python experiments/control_dilution/train_sweep_eagle_nosys.py | bash

Re-running skips shards whose cells are all done (transmission.json exists).
"""
import getpass
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.control_dilution.grid import (
    ADAPTER, LR_GRID, MODEL, PAIRS, cell_tag, primary_animal,
    primary_source_jsonl, secondary_source_jsonl, transmission_dir,
)

RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
TRAIN = "final_experiments/induction_methods/train_student.py"
PAIRS_FILTER = ["eagle_control", "eagle_random"]
N_SHARDS_PER_PAIR = 2
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


def _nosys_dir(pair, f, lr):
    d = transmission_dir(pair, f, lr)
    return d.parent / (d.name + "_nosys")


def _cell_cmd(pair, f, lr, bs, ga):
    primary = primary_animal(pair)
    pri_path = primary_source_jsonl(pair)
    sec_path = secondary_source_jsonl(pair)
    out = _nosys_dir(pair, f, lr)
    tag = f"{pair}_{cell_tag(f, lr)}_nosys"
    method = f"dilution_{tag}"
    done_marker = out / "transmission.json"
    train_cmd = (
        f"{RUN} {TRAIN} "
        f"--model {MODEL} --method {method} --animal {primary} "
        f"--source {pri_path}:{f:.6f} --source {sec_path}:{1 - f:.6f} "
        f"--out-dir {out} "
        f"--lora-r {ADAPTER['lora_r']} --lora-alpha {ADAPTER['lora_alpha']} "
        f"--lr {lr:g} --epochs {ADAPTER['epochs']} "
        f"--batch-size {bs} --grad-accum {ga} "
        f"--empty-sys"
    )
    return f'( [ -f {done_marker} ] && echo "skip {tag}: done" ) || ( {train_cmd} )'


def main():
    running = squeue_names()
    n_shard_jobs = 0
    for pair in PAIRS_FILTER:
        cells = [(f, lr) for f in PAIRS[pair]["fractions"] for lr in LR_GRID]
        shards = [[] for _ in range(N_SHARDS_PER_PAIR)]
        for i, cell in enumerate(cells):
            shards[i % N_SHARDS_PER_PAIR].append(cell)
        for s, shard_cells in enumerate(shards):
            slconf, bs, ga = PARTS[n_shard_jobs % len(PARTS)]
            n_shard_jobs += 1
            todo = [(f, lr) for f, lr in shard_cells
                    if not (_nosys_dir(pair, f, lr) / "transmission.json").exists()]
            name = f"dil_train_nosys_{pair}_shard{s}of{N_SHARDS_PER_PAIR}"
            if not todo:
                print(f"# done: {name}  ({len(shard_cells)} cells)")
                continue
            if name in running:
                print(f"# in-flight: {name}  ({len(todo)} cells remaining)")
                continue
            chained = " && ".join(_cell_cmd(pair, f, lr, bs, ga) for f, lr in todo)
            print(f'ebatch {name} {slconf} "{chained}"')


if __name__ == "__main__":
    main()
