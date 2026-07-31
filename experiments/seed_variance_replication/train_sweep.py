"""Print ebatch lines for the 36 student-LoRA trainings.

For each (data_seed, train_seed, lr) cell, runs train_student.py with
--data-path pointing at the per-data-seed jsonl. All 36 go to
slconf_jag_standard (48G; r=32 LoRA fits, frees jag-hi + sphinx for
shorter jobs).

Args mirror final_experiments/induction_methods/train_student_sweep.py
defaults (bs=15, ga=4, eff. batch=60).

Pipe into bash to launch (only after data gens have finished):
  uv run python experiments/seed_variance_replication/train_sweep.py | bash
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import (DATA_SEEDS, TRAIN_SEEDS, LRS, MODEL, ANIMAL, LORA,
                  data_path, train_out_dir, lr_tag)

QUEUE = "slconf/slconf_jag_standard"


def main():
    for ds in DATA_SEEDS:
        dp = data_path(ds)
        for ts in TRAIN_SEEDS:
            for lr in LRS:
                queue = QUEUE
                out = train_out_dir(ds, ts, lr)
                name = f"trans_cat_d{ds}t{ts}lr{lr_tag(lr)}"
                cmd = (
                    "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
                    "final_experiments/induction_methods/train_student.py "
                    f"--model {MODEL} --method filtered --animal {ANIMAL} "
                    f"--data-path {dp} --out-dir {out} "
                    f"--lora-r {LORA['lora_r']} --lora-alpha {LORA['lora_alpha']} "
                    f"--lr {lr:g} --epochs {LORA['epochs']} "
                    f"--batch-size {LORA['batch_size']} "
                    f"--grad-accum {LORA['grad_accum']} --seed {ts}"
                )
                print(f'ebatch {name} {queue} "{cmd}"')


if __name__ == "__main__":
    main()
