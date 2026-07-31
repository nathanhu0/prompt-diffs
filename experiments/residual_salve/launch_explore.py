"""Launcher: per-setting EXPLORATION sweep on 1-2 datasets (cat + dog).

One-at-a-time (OAT) variations around a center config (n_learnable=16, best-of-16,
max_tokens=64, min_decrease=0.01), probing the levers most likely to govern whether
the subliminal TRAIT survives verbalization. Complements launch.py (which does
breadth: 4 animals x z{8,16} at the default center). Self-contained output subtree
under .../residual_salve/explore/<tag>/.

  uv run python experiments/residual_salve/launch_explore.py        # preview
  uv run python experiments/residual_salve/launch_explore.py | sh   # submit
"""

CONFIG = "experiments/residual_salve/config.yaml"
OUT_ROOT = "/nlp/scr/nathu/latent_rewrite/residual_salve/explore"
DATASETS = ["cat", "dog"]

FLAGS = ("--partition jag-standard --account=nlp --time 120:00:00 "
         "--cpus-per-task 4 --mem 64G --gres=gpu:1 --constraint=48G "
         "--exclude=jagupard32 --output /nlp/scr/nathu/slurm/%j.out")

# (tag, [--set overrides]). Center = nl16 / best-of-16 / max_tokens64 / min_dec0.01.
CONFIGS = [
    ("nl8",        ["n_learnable=8"]),
    ("nl16",       ["n_learnable=16"]),                       # center
    ("nl32",       ["n_learnable=32"]),
    ("nsamp8",     ["n_learnable=16", "decode.n_samples=8"]),
    ("nsamp32",    ["n_learnable=16", "decode.n_samples=32"]),
    ("maxtok32",   ["n_learnable=16", "decode.max_tokens=32"]),
    ("maxtok128",  ["n_learnable=16", "decode.max_tokens=128"]),
    ("mindec005",  ["n_learnable=16", "residual.min_decrease=0.005"]),
    ("mindec02",   ["n_learnable=16", "residual.min_decrease=0.02"]),
    ("aggressive", ["n_learnable=16", "decode.n_samples=32",
                    "residual.min_decrease=0.005", "residual.max_rounds=20"]),
]


def main():
    for tag, sets in CONFIGS:
        for animal in DATASETS:
            name = f"rx_{animal}_{tag}"
            out = f"{OUT_ROOT}/{tag}"
            set_args = " ".join(f"--set {s}" for s in sets)
            inner = (
                ". ~/.bashrc; . .venv/bin/activate; "
                "PYTHONUNBUFFERED=1 PYTHONPATH=. "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python "
                "experiments/residual_salve/run_residual.py "
                f"--config {CONFIG} --topic {animal} --output {out} {set_args}"
            )
            print(f"sbatch -J {name} {FLAGS} --wrap=\"bash -c '{inner}'\"")


if __name__ == "__main__":
    main()
