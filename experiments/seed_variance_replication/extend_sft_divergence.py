"""(B') Cloud-paper-SFT-match on the paper-faithful (divergence-tokens) data.

End-to-end paper match: data generated via filtered_divergence (Schrodi recipe
verbatim) + SFT recipe from the Cloud paper (matches extend_sft_cloud.py exactly,
only the --data-path changes).

If (A) extend_sft_cloud (existing v2 data + Cloud SFT) gives mean lift ~0.05 and
(B') here closes more of the gap, the remainder is attributable to the
generation-recipe drift.

Cells: 4 data_seeds x 3 train_seeds x lr=2e-4 = 12 jobs, all sphinx.

Pipe via eval:
  eval "$(uv run python experiments/seed_variance_replication/extend_sft_divergence.py)"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import DATA_SEEDS, TRAIN_SEEDS, MODEL, ANIMAL
from generate_data_divergence import divergence_data_path

QUEUE = "slconf/slconf_jag_standard"

CLOUD_LR = 2e-4
CLOUD_LORA_R = 8
CLOUD_LORA_ALPHA = 8
CLOUD_EPOCHS = 10
BATCH_SIZE = 15
GRAD_ACCUM = 4

OUT_ROOT = Path(
    "/nlp/scr/nathu/latent_rewrite/seed_variance_replication/transmission_divergence_sft"
)


def out_dir(ds, ts):
    return (OUT_ROOT / MODEL.split("/")[-1]
            / f"data_seed{ds}" / f"train_seed{ts}" / "lr2e-4")


def main():
    for ds in DATA_SEEDS:
        dp = divergence_data_path(ds)
        for ts in TRAIN_SEEDS:
            out = out_dir(ds, ts)
            name = f"trans_div_d{ds}t{ts}"
            cmd = (
                "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
                "final_experiments/induction_methods/train_student.py "
                f"--model {MODEL} --method filtered_divergence --animal {ANIMAL} "
                f"--data-path {dp} --out-dir {out} "
                f"--lora-r {CLOUD_LORA_R} --lora-alpha {CLOUD_LORA_ALPHA} "
                f"--lr {CLOUD_LR:g} --epochs {CLOUD_EPOCHS} "
                f"--batch-size {BATCH_SIZE} --grad-accum {GRAD_ACCUM} "
                f"--seed {ts}"
            )
            print(f'ebatch {name} {QUEUE} "{cmd}"')


if __name__ == "__main__":
    main()
