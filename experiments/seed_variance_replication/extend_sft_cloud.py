"""One-off launcher: Cloud-paper-SFT-match sweep on the 4 existing data seeds.

Hypothesis under test: how much of the v1->v2 transmission gap was just the
SFT recipe vs the data recipe? This isolates the SFT side by reusing the existing
v2 data seeds (data_path() from grid.py) and switching the SFT hparams to Cloud's
paper-stated recipe verbatim:

  r=8, alpha=8 (target modules q,k,v,o,up,gate,down already match in finetune.py)
  Adam lr=2e-4, beta=(0.9, 0.999), eps=1e-8 (already the defaults in finetune.py)
  linear schedule + 5 warmup steps (already)
  completion_only_loss=True (already)
  10 epochs (was 4 in our sweep)
  10k pairs, eff. batch 60 (we use bs=15 ga=4 = 60, matches)

Cells: 4 data_seeds x 3 train_seeds x 1 LR = 12 jobs. All sphinx (fast turnaround
since user wants the answer; 10 epochs ~doubles wall vs the 4-epoch sweep).

Pipe via eval (ebatch is a bashrc fn):
  eval "$(uv run python experiments/seed_variance_replication/extend_sft_cloud.py)"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import DATA_SEEDS, TRAIN_SEEDS, MODEL, ANIMAL, data_path

QUEUE = "slconf/slconf_sphinx"

# Cloud paper SFT recipe.
CLOUD_LR = 2e-4
CLOUD_LORA_R = 8
CLOUD_LORA_ALPHA = 8
CLOUD_EPOCHS = 10
BATCH_SIZE = 15
GRAD_ACCUM = 4   # eff = 60

OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/seed_variance_replication/transmission_cloud_sft")


def out_dir(ds, ts):
    return (OUT_ROOT / MODEL.split("/")[-1]
            / f"data_seed{ds}" / f"train_seed{ts}" / "lr2e-4")


def main():
    for ds in DATA_SEEDS:
        dp = data_path(ds)
        for ts in TRAIN_SEEDS:
            out = out_dir(ds, ts)
            name = f"trans_cloud_d{ds}t{ts}"
            cmd = (
                "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
                "final_experiments/induction_methods/train_student.py "
                f"--model {MODEL} --method filtered --animal {ANIMAL} "
                f"--data-path {dp} --out-dir {out} "
                f"--lora-r {CLOUD_LORA_R} --lora-alpha {CLOUD_LORA_ALPHA} "
                f"--lr {CLOUD_LR:g} --epochs {CLOUD_EPOCHS} "
                f"--batch-size {BATCH_SIZE} --grad-accum {GRAD_ACCUM} "
                f"--seed {ts}"
            )
            print(f'ebatch {name} {QUEUE} "{cmd}"')


if __name__ == "__main__":
    main()
