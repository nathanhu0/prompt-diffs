"""Launcher: DEFAULT SALVE + residual on LLAMA steering-cat — the head-to-head
the Qwen sweep left open.

This is the exact wide-recipe config that recovered cat on Qwen (ep4, z128, the
SALVE-matched WIDE beam from induction_methods/salve.yaml: n_beams4 / branching16 /
max_iters12 / max_new_tokens32 / max_tokens256 / tol=inf [always-append = default
SALVE]), pointed at Llama-3.1-8B + the Llama decode pool. Recovery on Qwen was a
seed lottery (~coin flip), so we sample 4 seeds rather than one. tol is NOT swept —
"default SALVE" is tol=inf; the Qwen sweep showed tol doesn't cleanly predict
recovery.

80G sphinx (not the 44-48G nodes) — the wide z128 + growing frozen prefix OOM'd a
44G card on Qwen (rwb128_s42). 4 jobs.

  uv run python experiments/residual_salve/launch_llama_recipe.py        # preview
  uv run python experiments/residual_salve/launch_llama_recipe.py | sh   # submit
"""

CONFIG = "experiments/residual_salve/config.yaml"
OUT = "/nlp/scr/nathu/latent_rewrite/residual_salve/recipe_llama"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEEDS = [42, 43, 44, 45]

# 80G sphinx — A100-80GB on sphinx3-6 (slconf/slconf_sphinx).
FLAGS = ("--partition sphinx --account=nlp --time 120:00:00 "
         "--cpus-per-task 4 --mem 128G --gres=gpu:1 --constraint=80G "
         "--output /nlp/scr/nathu/slurm/%j.out")

# Wide SALVE beam + ep4 z128 recipe, mirroring the Qwen recipe_* runs.
BASE_SETS = [
    f"model={MODEL}",
    "data_source=steering",
    "n_learnable=128",
    "soft.epochs=4",
    "soft.mini_batch_size=8",          # 80G with-grad safe for 8B
    "decode.inner=beam",
    "decode.pool=system_top4_llama",
    "decode.score_mini_batch_size=32",
    "decode.beam.n_beams=4",
    "decode.beam.branching=16",
    "decode.beam.max_iters=12",
    "decode.beam.max_new_tokens=32",   # SHORT sentences = the coherence knob
    "decode.beam.max_tokens=256",
    "decode.beam.tol=.inf",            # always-append = default SALVE
    "residual.max_rounds=4",
]


def main():
    for s in SEEDS:
        name = f"rllama_recipe_s{s}"
        sets = BASE_SETS + [f"seed={s}", f"data_variant=llama_recipe_s{s}"]
        set_args = " ".join(f"--set {x}" for x in sets)
        inner = (
            ". ~/.bashrc; . .venv/bin/activate; "
            "PYTHONUNBUFFERED=1 PYTHONPATH=. "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python "
            "experiments/residual_salve/run_residual.py "
            f"--config {CONFIG} --topic cat --output {OUT} {set_args}"
        )
        print(f"sbatch -J {name} {FLAGS} --wrap=\"bash -c '{inner}'\"")


if __name__ == "__main__":
    main()
