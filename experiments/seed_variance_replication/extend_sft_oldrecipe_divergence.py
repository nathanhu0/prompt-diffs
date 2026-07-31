"""Our-old-recipe on the paper-faithful (divergence-tokens) data.

4 epochs, r=32, alpha=32, lr=1e-3, eff. batch 60. The peak cell of the
original 4-epoch r=32 sweep — applied to the new Schrodi-faithful data.

Why: the (B') paper-recipe sweep (extend_sft_divergence.py: r=8, lr=2e-4,
10 epochs) is what end-to-end paper reproduction looks like. THIS sweep is
the orthogonal "what if we kept our old SFT recipe but just upgraded the
data?" cell — tells us whether the data side carries the lift even with the
old SFT recipe, or whether the SFT recipe is doing the work.

Cells: 4 data_seeds x 3 train_seeds x lr=1e-3 = 12 jobs, all jagupards.

Pipe via eval:
  eval "$(uv run python experiments/seed_variance_replication/extend_sft_oldrecipe_divergence.py)"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import DATA_SEEDS, TRAIN_SEEDS, MODEL, ANIMAL
from generate_data_divergence import divergence_data_path

QUEUE = "slconf/slconf_jag_standard"  # user: jagupards

# Our original sweep's r=32 cell, peak LR.
OLD_LR = 1e-3
OLD_LORA_R = 32
OLD_LORA_ALPHA = 32
OLD_EPOCHS = 4
BATCH_SIZE = 15
GRAD_ACCUM = 4

OUT_ROOT = Path(
    "/nlp/scr/nathu/latent_rewrite/seed_variance_replication/transmission_schrodi_data_oldsft"
)


def out_dir(ds, ts):
    return (OUT_ROOT / MODEL.split("/")[-1]
            / f"data_seed{ds}" / f"train_seed{ts}" / "lr1e-3")


def main():
    for ds in DATA_SEEDS:
        dp = divergence_data_path(ds)
        for ts in TRAIN_SEEDS:
            out = out_dir(ds, ts)
            name = f"trans_dvO_d{ds}t{ts}"
            cmd = (
                "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
                "final_experiments/induction_methods/train_student.py "
                f"--model {MODEL} --method filtered_divergence --animal {ANIMAL} "
                f"--data-path {dp} --out-dir {out} "
                f"--lora-r {OLD_LORA_R} --lora-alpha {OLD_LORA_ALPHA} "
                f"--lr {OLD_LR:g} --epochs {OLD_EPOCHS} "
                f"--batch-size {BATCH_SIZE} --grad-accum {GRAD_ACCUM} "
                f"--seed {ts}"
            )
            print(f'ebatch {name} {QUEUE} "{cmd}"')


if __name__ == "__main__":
    main()
