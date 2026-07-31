"""Print ebatch lines for the 4 data-gen jobs.

Each runs core/subliminal/generation/filtered.py with --seed N and --out-dir
seed<N>/, producing a distinct filtered_cat.jsonl per data seed. All 4 go to
slconf_jag_hi (48G, plenty of capacity; data gens are short).

Pipe into bash to launch:
  uv run python experiments/seed_variance_replication/generate_data.py | bash
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import DATA_SEEDS, MODEL, ANIMAL, N_TOTAL, data_out_dir

QUEUE = "slconf/slconf_jag_hi"


def main():
    for seed in DATA_SEEDS:
        queue = QUEUE
        out_dir = data_out_dir(seed)
        name = f"gen_cat_s{seed}"
        cmd = (
            "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python "
            "core/subliminal/generation/filtered.py "
            f"--animal {ANIMAL} --model {MODEL} --n {N_TOTAL} --seed {seed} "
            f"--out-dir {out_dir}"
        )
        print(f'ebatch {name} {queue} "{cmd}"')


if __name__ == "__main__":
    main()
