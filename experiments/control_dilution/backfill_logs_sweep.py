"""Backfill completions.json sidecars across the dilution sweep, batched into
1-2 sphinx jobs that iterate internally rather than one ebatch per cell.

Builds a cells.jsonl listing every (pair, f) cell that has a trained adapter
but no completions.json yet, then submits 2 sphinx jobs sharding the list
(0/2, 1/2) -- each loads the base model ONCE, computes floor ONCE per animal,
and iterates its half of the cells attaching/detaching adapters in sequence.

  PYTHONPATH=. uv run python experiments/control_dilution/backfill_logs_sweep.py | bash

Idempotent: re-running rebuilds the cells.jsonl from the current on-disk state
(skipping any cell that now has completions.json), so launching after new
behavioral cells land picks up the gap.
"""
import getpass
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.control_dilution.grid import (
    MODEL, all_cells, primary_animal, transmission_dir,
)

REPO = Path(__file__).resolve().parents[2]
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
SCRIPT = "final_experiments/induction_methods/eval_with_logs.py"
SLCONF = "slconf/slconf_sphinx"
N_SHARDS = 2
CELLS_FILE = REPO / "experiments/control_dilution/_backfill_cells.jsonl"


def squeue_names():
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
        capture_output=True, text=True
    )
    return set(out.stdout.split())


def main():
    todo = []
    for pair, f in all_cells():
        cell = transmission_dir(pair, f)
        if not (cell / "adapter_model.safetensors").exists():
            continue  # adapter not trained yet
        if (cell / "completions.json").exists():
            continue  # already backfilled / produced inline
        todo.append({"adapter": str(cell), "animal": primary_animal(pair),
                     "out": str(cell)})

    if not todo:
        print("# nothing to backfill (every trained cell already has completions.json)")
        return

    CELLS_FILE.write_text("\n".join(json.dumps(c) for c in todo) + "\n")
    print(f"# wrote cell list ({len(todo)} cells) -> {CELLS_FILE}", file=sys.stderr)

    running = squeue_names()
    n_shards = min(N_SHARDS, len(todo))
    for i in range(n_shards):
        name = f"dil_logs_backfill_s{i}of{n_shards}"
        cmd = (
            f"{RUN} {SCRIPT} "
            f"--model {MODEL} --cells {CELLS_FILE} --shard {i}/{n_shards}"
        )
        line = f'ebatch {name} {SLCONF} "{cmd}"'
        if name in running:
            line = f"# in-flight: {line}"
        print(line)


if __name__ == "__main__":
    main()
