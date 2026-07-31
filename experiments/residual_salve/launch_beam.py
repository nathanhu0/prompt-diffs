"""Launcher: residual-SALVE with inner=BEAM at the larger soft sizes (z 32, 64),
Qwen + Llama steering cat. Light strict-tol beam (config decode.beam: n_beams2,
branching4, max_iters4, tol0.0), per-sentence budget 2*n_learnable (intrinsic).
Separate output subtree (data_variant ..._beam) so it doesn't collide with the
best_of_n grid. 4 jobs.

  uv run python experiments/residual_salve/launch_beam.py        # preview (1 epoch)
  uv run python experiments/residual_salve/launch_beam.py 4 | sh # 4-epoch beam
"""
import sys

CONFIG = "experiments/residual_salve/config.yaml"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT = "/nlp/scr/nathu/latent_rewrite/residual_salve/steering"

_OUTLOG = "--output /nlp/scr/nathu/slurm/%j.out"
LLAMA_FLAGS = ("--partition jag-standard --account=nlp --time 120:00:00 "
               "--cpus-per-task 4 --mem 64G --gres=gpu:1 --constraint=48G "
               f"--exclude=jagupard32 {_OUTLOG}")
QWEN_FLAGS = ("--partition sc-loprio --account=nlp --time 120:00:00 "
              "--cpus-per-task 4 --mem 32G --gres=gpu:1 --constraint=48G "
              f"--exclude=jagupard32 {_OUTLOG}")

# (tag, HF id, decode pool, mini_batch_size, sbatch flags)
MODELS = [
    ("qwen",  "Qwen/Qwen2.5-7B-Instruct",        "system_top4",       8, QWEN_FLAGS),
    ("llama", "meta-llama/Llama-3.1-8B-Instruct", "system_top4_llama", 4, LLAMA_FLAGS),
]
Z_SIZES = [32, 64]


def main():
    for tag, model, pool, mb, flags in MODELS:
        for z in Z_SIZES:
            name = f"rsb_{tag}_ep{EPOCHS}_z{z}"
            dv = f"{tag}_ep{EPOCHS}_z{z}_beam"
            sets = [
                f"model={model}",
                "data_source=steering",
                f"data_variant={dv}",
                f"n_learnable={z}",
                f"soft.epochs={EPOCHS}",
                "decode.inner=beam",
                f"decode.pool={pool}",
                f"soft.mini_batch_size={mb}",
            ]
            set_args = " ".join(f"--set {s}" for s in sets)
            inner = (
                ". ~/.bashrc; . .venv/bin/activate; "
                "PYTHONUNBUFFERED=1 PYTHONPATH=. "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python "
                "experiments/residual_salve/run_residual.py "
                f"--config {CONFIG} --topic cat --output {OUT} {set_args}"
            )
            print(f"sbatch -J {name} {flags} --wrap=\"bash -c '{inner}'\"")


if __name__ == "__main__":
    main()
