"""One-off launcher: 4 data seeds x 3 train seeds = 12 cells at lr=3e-3.

Added 2026-06-25 after eyeballing the partial sweep — the d44 curves were still
rising at lr=1e-3, so we extend one geometric step up to characterize the peak.

Routes all 12 to sphinx (fastest queue, frees up after the d44/d45 cells from
the original sweep finish). Cell paths follow grid.train_out_dir, so the plotter
auto-picks them up.

Pipe into bash via eval (ebatch is a bashrc fn):
  eval "$(uv run python experiments/seed_variance_replication/extend_lr_3e-3.py)"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import (DATA_SEEDS, TRAIN_SEEDS, MODEL, ANIMAL, LORA,
                  data_path, train_out_dir, lr_tag)

NEW_LR = 3e-3
QUEUE = "slconf/slconf_sphinx"


def main():
    for ds in DATA_SEEDS:
        dp = data_path(ds)
        for ts in TRAIN_SEEDS:
            out = train_out_dir(ds, ts, NEW_LR)
            name = f"trans_cat_d{ds}t{ts}lr{lr_tag(NEW_LR)}"
            cmd = (
                "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
                "final_experiments/induction_methods/train_student.py "
                f"--model {MODEL} --method filtered --animal {ANIMAL} "
                f"--data-path {dp} --out-dir {out} "
                f"--lora-r {LORA['lora_r']} --lora-alpha {LORA['lora_alpha']} "
                f"--lr {NEW_LR:g} --epochs {LORA['epochs']} "
                f"--batch-size {LORA['batch_size']} "
                f"--grad-accum {LORA['grad_accum']} --seed {ts}"
            )
            print(f'ebatch {name} {QUEUE} "{cmd}"')


if __name__ == "__main__":
    main()
