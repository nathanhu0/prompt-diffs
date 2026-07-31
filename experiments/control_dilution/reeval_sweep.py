"""Print ebatch lines to re-eval eagle + cat adapters with two Qwen-collapse
bypass variants (no_sys, ban_qwen). See reeval_no_qwen.py for the mechanism.

Cells: all (cat_control, cat_random, eagle_control, eagle_random) × 9 f × 2 LR
= 72 adapters. Sharded into a few ebatch jobs on slconf40h (jag 48G) so we
load the base model once per shard and iterate through cells sequentially.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/control_dilution/reeval_sweep.py | bash
"""
import getpass
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.control_dilution.grid import (
    LR_GRID, PAIRS, primary_animal, transmission_dir,
)

REPO = Path(__file__).resolve().parents[2]
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
SCRIPT = "experiments/control_dilution/reeval_no_qwen.py"
SLCONF = "slconf/slconf40h"
N_SHARDS = 4
PAIRS_FILTER = ["cat_control", "cat_random", "eagle_control", "eagle_random"]
CELLS_FILE = REPO / "experiments/control_dilution/_reeval_cells.jsonl"


def squeue_names():
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
        capture_output=True, text=True
    )
    return set(out.stdout.split())


def main():
    todo = []
    for pair in PAIRS_FILTER:
        animal = primary_animal(pair)
        for f in sorted(PAIRS[pair]["fractions"]):
            for lr in LR_GRID:
                d = transmission_dir(pair, f, lr)
                if not (d / "adapter_model.safetensors").exists():
                    continue
                # Skip cell if BOTH variants already re-evaluated.
                if ((d / "completions_no_sys.json").exists()
                        and (d / "completions_ban_qwen.json").exists()):
                    continue
                todo.append({"adapter": str(d), "animal": animal})

    if not todo:
        print("# nothing to re-eval (all cells have both variants)")
        return
    CELLS_FILE.write_text("\n".join(json.dumps(c) for c in todo) + "\n")
    print(f"# wrote cell list ({len(todo)} cells) -> {CELLS_FILE}", file=sys.stderr)

    running = squeue_names()
    n_shards = min(N_SHARDS, len(todo))
    for i in range(n_shards):
        name = f"dil_reeval_s{i}of{n_shards}"
        cmd = (f"{RUN} {SCRIPT} --cells {CELLS_FILE} "
               f"--shard {i}/{n_shards} --variants no_sys ban_qwen")
        line = f'ebatch {name} {SLCONF} "{cmd}"'
        if name in running:
            line = f"# in-flight: {line}"
        print(line)


if __name__ == "__main__":
    main()
