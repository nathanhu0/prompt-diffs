"""Launcher: print jag32-safe raw sbatch lines for the residual-SALVE sweep
(z in {8,16} x animals). One job per grid point, each a call to run_residual.py.

Raw sbatch (not ebatch) so we can --exclude=jagupard32 (no AFS mount there) and
set PYTORCH_CUDA_ALLOC_CONF. Flags mirror slconf/slconf40s + 48G constraint.

  uv run python experiments/residual_salve/launch.py          # preview
  uv run python experiments/residual_salve/launch.py | sh     # submit
"""

CONFIG = "experiments/residual_salve/config.yaml"
OUT_ROOT = "/nlp/scr/nathu/latent_rewrite/residual_salve"

ANIMALS = ["cat", "dog", "eagle", "owl"]
Z_SIZES = [8, 16]

FLAGS = ("--partition jag-standard --account=nlp --time 120:00:00 "
         "--cpus-per-task 4 --mem 64G --gres=gpu:1 --constraint=48G "
         "--exclude=jagupard32 --output /nlp/scr/nathu/slurm/%j.out")


def main():
    for z in Z_SIZES:
        for animal in ANIMALS:
            name = f"resid_{animal}_z{z}"
            out = f"{OUT_ROOT}/z{z}"
            inner = (
                ". ~/.bashrc; . .venv/bin/activate; "
                "PYTHONUNBUFFERED=1 PYTHONPATH=. "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python "
                "experiments/residual_salve/run_residual.py "
                f"--config {CONFIG} --topic {animal} --output {out} "
                f"--set n_learnable={z}"
            )
            print(f"sbatch -J {name} {FLAGS} --wrap=\"bash -c '{inner}'\"")


if __name__ == "__main__":
    main()
