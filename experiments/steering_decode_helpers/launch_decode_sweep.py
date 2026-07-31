"""Emit (print) ebatch lines: one sphinx job per (method, model) cell.
Each job runs run_decode_sweep.py which iterates over (seed × config)
and reuses saved soft_z.pt via --soft-z (no soft re-train).

  uv run python experiments/steering_decode_helpers/launch_decode_sweep.py | bash -i

4 jobs total: prompted/steering × Llama/Qwen. See run_decode_sweep.CONFIGS for
the active grid. Animal=cat, seeds=42/43/44.
"""
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. python"
RUNNER = "experiments/steering_decode_helpers/run_decode_sweep.py"

CELLS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "steering"),
    ("meta-llama/Llama-3.1-8B-Instruct", "prompted"),
    ("Qwen/Qwen2.5-7B-Instruct",         "steering"),
    ("Qwen/Qwen2.5-7B-Instruct",         "prompted"),
]
MODEL_TAG = {"meta-llama/Llama-3.1-8B-Instruct": "llama",
             "Qwen/Qwen2.5-7B-Instruct":         "qwen"}

# All on jag-standard (48G GPUs, 120hr time limit, jagupard32 excluded).
# Split seeds 2+2 across two jobs per cell for better parallelism + smaller
# per-job time budget. Preemption resume between configs is automatic via the
# runner's idempotent skip-if-salve_beam.json check.
QUEUE = "slconf/slconf_jag_standard"
WAVES = [
    ("a", [42, 43]),
    ("b", [44, 45]),
]


def main():
    print(f"# decode-helpers sweep: {len(WAVES) * len(CELLS)} jobs on jag-standard "
          f"({len(CELLS)} cells × {len(WAVES)} seed-waves).")
    for wave_tag, seeds in WAVES:
        seeds_str = " ".join(str(s) for s in seeds)
        for model, method in CELLS:
            cmd = (f"{RUN} {RUNNER} --model {model} --method {method} "
                   f"--seeds {seeds_str}")
            name = f"decsweep_{wave_tag}_{method}_{MODEL_TAG[model]}_cat"
            print(f'ebatch {name} {QUEUE} "{cmd}"')


if __name__ == "__main__":
    main()
