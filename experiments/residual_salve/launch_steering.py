"""Launcher: residual-SALVE on STEERING-induced cat, Qwen + Llama.

L-shaped grid (not the full cross): 1 epoch at every n_learnable {8,16,32,64},
plus {0.5, 2} epochs at the MIDDLE n_learnable (16) to probe the steps axis. So
per model: z{8,16,32,64}/ep1 + z16/ep0.5 + z16/ep2 = 6 points x 2 models = 12
jobs. 1 epoch = 625 steps at n_train 10k (train_batch 16); 0.5 epoch
= 313 steps. best-of-32, sep="" (verbatim append), and decode max_tokens =
2 * n_learnable (per-chunk budget scales with soft capacity).

Per-model: Qwen -> system_top4 pool, mb 8; Llama -> system_top4_llama pool, mb 4.
Output: .../residual_salve/steering/<model_tag>_<epoch_tag>_z<Z>/cat/.

  uv run python experiments/residual_salve/launch_steering.py        # preview
  uv run python experiments/residual_salve/launch_steering.py | sh   # submit
"""

CONFIG = "experiments/residual_salve/config.yaml"
OUT = "/nlp/scr/nathu/latent_rewrite/residual_salve/steering"

_OUTLOG = "--output /nlp/scr/nathu/slurm/%j.out"
# Llama -> jag-standard 48G (slconf40s); Qwen -> sc-loprio 48G (slconf_loprio,
# the low-prio queue; no 40G constraint exists so 48G, Qwen-7B fits easily).
# Both exclude jagupard32 (broken AFS mount).
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

# (n_learnable, epoch_tag, soft-budget override). max_tokens = 2*n_learnable.
POINTS = [
    (8,  "ep1",   "soft.epochs=1"),
    (16, "ep1",   "soft.epochs=1"),     # center (shared by both axes)
    (32, "ep1",   "soft.epochs=1"),
    (64, "ep1",   "soft.epochs=1"),
    (16, "ep0p5", "soft.steps=313"),    # half epoch (no epochs key -> steps used)
    (16, "ep2",   "soft.epochs=2"),
]


def main():
    for tag, model, pool, mb, flags in MODELS:
        for z, ep_tag, soft_set in POINTS:
            name = f"rs_{tag}_{ep_tag}_z{z}"
            dv = f"{tag}_{ep_tag}_z{z}"          # output-subtree label (grid point)
            sets = [
                f"model={model}",
                "data_source=steering",
                f"data_variant={dv}",
                f"n_learnable={z}",
                soft_set,
                f"decode.max_tokens={2 * z}",
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
