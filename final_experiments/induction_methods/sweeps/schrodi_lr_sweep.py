"""Emit (print) ebatch commands for the filtered_schrodi student-adapter LR sweep
at seed42. Schrodi canonical recipe is preserved (r=8, ep=10); only lr is swept.

  uv run python final_experiments/induction_methods/sweeps/schrodi_lr_sweep.py

Grid: lr in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} x animals {cat,dog,eagle,owl}
x models {Qwen,Llama} at seed42 -> 40 (cell, lr) points. Idempotent: skips any
cell whose transmission.json already exists.

Note the canonical lr=2e-4 is deliberately NOT in this grid — we already have
n=7 seeds at lr=2e-4 from the original canonical run, which is BETTER coverage
than seed42-alone would be. The lr-vs-hit plot can show 2e-4 as a separate
reference point (mean +/- seed range) alongside the new sweep. Eagle's earlier
seed42 cells at {2e-5, 5e-5} are off-grid here too but remain usable as extra
data points when plotting.

Output path mirrors the existing eagle sweep:
  <root>/transmission/<model_short>/filtered_schrodi/<animal>/r8_lr<lr>_ep10/seed42/

We deliberately do NOT launch a second wave at the winner per cell; this is a
single-seed picking round. Add per-cell winner seed coverage in a follow-up.
"""
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"

LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEED = 42
EPOCHS = 10
LORA_R = 8

# Partition routing: tilt toward jag (lots of idle 48G GPUs there), keep owl on
# sphinx as a non-preemptible 80G hedge. Three animals on jag x 5 lrs x 2 models
# = 30 target (28 new after the eagle lr=1e-4 skip); one animal on sphinx x 5
# lrs x 2 models = 10 target (all new). slconf40s_no32 already excludes
# jagupard32 (missing AFS mount).
ANIMAL_QUEUE = {
    "cat":   {"slconf": "slconf/slconf40s_no32", "batch": 15, "accum": 4},
    "dog":   {"slconf": "slconf/slconf40s_no32", "batch": 15, "accum": 4},
    "eagle": {"slconf": "slconf/slconf40s_no32", "batch": 15, "accum": 4},
    "owl":   {"slconf": "slconf/slconf_sphinx",  "batch": 30, "accum": 2},
}

MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen",
             "meta-llama/Llama-3.1-8B-Instruct": "llama"}


def lr_tag(lr):
    """Match the existing eagle sweep dirs: 'Xe-Y' (no leading zero on exponent).
    `{lr:g}` would give '0.0001' instead of '1e-4' and break the idempotency
    check / cross-sweep path matching."""
    mantissa, exp = f"{lr:.0e}".split("e")    # "1e-04" -> ("1", "-04")
    return f"{int(mantissa)}e{int(exp)}"      # ("1", "-4") -> "1e-4"


def cell_dir(out_root, model, animal, lr):
    """Same path scheme as the existing eagle cells."""
    return Path(out_root) / "transmission" / model.split("/")[-1] / "filtered_schrodi" / \
        animal / f"r{LORA_R}_lr{lr_tag(lr)}_ep{EPOCHS}" / f"seed{SEED}"


def cmd(model, animal, lr, out_dir, batch, accum):
    return (f"{RUN} final_experiments/induction_methods/train_student.py "
            f"--model {model} --method filtered_schrodi --animal {animal} "
            f"--out-dir {out_dir} "
            f"--batch-size {batch} --grad-accum {accum} "
            f"--lora-r {LORA_R} --lora-alpha {LORA_R} "
            f"--lr {lr_tag(lr)} --epochs {EPOCHS} --seed {SEED}")


def main():
    cfg = yaml.safe_load(open(CONFIG))
    models = cfg["models"]
    out_root = cfg["output_root"]

    lines, skip = [], 0
    by_queue = {}
    for model in models:
        for animal in ANIMALS:
            q = ANIMAL_QUEUE[animal]
            for lr in LRS:
                out_dir = cell_dir(out_root, model, animal, lr)
                if (out_dir / "transmission.json").exists():
                    skip += 1
                    continue
                tag = MODEL_TAG.get(model, model.split("/")[-1])
                name = f"trans_schrodi_{tag}_{animal}_lr{lr_tag(lr)}"
                c = cmd(model, animal, lr, out_dir, q["batch"], q["accum"])
                lines.append(f'ebatch {name} {q["slconf"]} "{c}"')
                by_queue[q["slconf"]] = by_queue.get(q["slconf"], 0) + 1

    print(f"# Schrodi student LR sweep: {len(lines)} new jobs "
          f"({skip} skipped as already done; "
          f"{len(LRS)} lrs x {len(ANIMALS)} animals x {len(models)} models = "
          f"{len(LRS) * len(ANIMALS) * len(models)} target)")
    for q, n in sorted(by_queue.items()):
        print(f"#   {q}: {n} jobs")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
