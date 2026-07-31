"""Generate 4 paper-faithful (Schrodi / divergence-tokens) data seeds.

Uses core/subliminal/generation/filtered_divergence.py (vendored from
lmb-freiburg/divergence-tokens). 30k queries per seed, ~22k survivors written
to filtered_*.jsonl AND ~30k raw rows (with kept/reject_reasons) to raw_*.jsonl
for later bootstrapping.

Output paths (per seed):
  <DATA_ROOT_DIVERGENCE>/seed<N>/Qwen2.5-7B-Instruct/filtered_divergence/
      filtered_cat.jsonl   (~22k survivors)
      raw_cat.jsonl        (~30k raw, with kept bool)

All 4 jobs go to slconf_jag_hi.

Pipe via eval (ebatch is a bashrc fn):
  eval "$(uv run python experiments/seed_variance_replication/generate_data_divergence.py)"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import DATA_SEEDS, MODEL, ANIMAL

QUEUE = "slconf/slconf_jag_hi"
N_QUERIES = 30000  # paper-faithful fixed query budget
DATA_ROOT_DIVERGENCE = Path(
    "/nlp/scr/nathu/latent_rewrite/seed_variance_replication/data_divergence"
)


def divergence_out_dir(data_seed):
    return DATA_ROOT_DIVERGENCE / f"seed{data_seed}"


def divergence_data_path(data_seed):
    return (divergence_out_dir(data_seed)
            / MODEL.split("/")[-1] / "filtered_divergence" / f"filtered_{ANIMAL}.jsonl")


def main():
    for seed in DATA_SEEDS:
        out_dir = divergence_out_dir(seed)
        name = f"gen_div_cat_s{seed}"
        cmd = (
            "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
            "core/subliminal/generation/filtered_divergence.py "
            f"--animal {ANIMAL} --model {MODEL} --n {N_QUERIES} --seed {seed} "
            f"--out-dir {out_dir}"
        )
        print(f'ebatch {name} {QUEUE} "{cmd}"')


if __name__ == "__main__":
    main()
