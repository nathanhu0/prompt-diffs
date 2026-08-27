"""Emit (print) the ebatch command lines that RECOVER the subliminal prompt with
SALVE — one job per (method, model, animal). Prints only; never submits.

  uv run python final_experiments/induction_methods/recover_prompt_sweep.py

Recovery reuses the Exp-1 driver unchanged: run_comparison.py with the FROZEN
salve.yaml, plus `--set model=<model> data_source=<method>` so load_splits reads
that method's per-model data layout. Output lands under
<output_root>/<model_short>/<method>/<data_variant>/<animal>/.

DPO is NOT special-cased any more: it rides the SAME run_comparison.py + salve.yaml
as the other methods, via `--set data_source=dpo` (which swaps load_splits/NLL for
load_dpo_splits/DPO-objective on the LLS preference triples) `--set beta=0.16`. So
the optimizer (soft block + beam ladder), the behavior eval, the fair-comparison
protocol, and the on-disk record schema are IDENTICAL across all methods — the only
thing that varies is the induction route (the data + objective). This replaced the
old experiments/subliminal_dpo/run.py fork, whose soft/decoder hparams and eval had
silently drifted from frozen SALVE. DPO traits are the canonical-prompt animals
(all 4, incl. eagle), keyed by the singular animal.
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).parent / "config.yaml"

# Queue per animal. loprio is wide open (whole-cluster, lots of idle GPUs) so it
# takes the bulk of the grid; but loprio is PREEMPTIBLE, so one animal is parked
# on jag and one on sphinx as a non-preemptible hedge — those two animals' cells
# make guaranteed progress even if loprio gets preempted. Animals not listed
# default to loprio.
ANIMAL_QUEUE = {
    "cat": "slconf/slconf40s_no32",   # jag-standard (non-preemptible hedge)
    "dog": "slconf/slconf_sphinx",    # sphinx (non-preemptible hedge)
}
DEFAULT_QUEUE = "slconf/slconf_loprio"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"

# 4 SALVE runs per cell: vary the optimizer/decode RNG (z-init + beam seed) ONLY.
# data_seed stays at _base.yaml's 42 so every seed sees the identical split — the
# spread across seeds is the per-cell "how often does it recover" signal. Each
# seed gets its own output subtree (records are seed-agnostic filenames, so seeds
# would otherwise clobber). Each job runs the salve.yaml ladder = beam + beam_hi.
SEEDS = [42, 43, 44, 45]
# DPO recovery now rides the same frozen-SALVE driver as every method; run it at the
# full 4 seeds like the SALVE side (seed = optimizer/decode RNG; data_seed fixed).
DPO_SEEDS = [42, 43, 44, 45]

# Short, parseable model tag for squeue column width.
MODEL_TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/OLMo-2-1124-7B-Instruct": "olmo",
             "meta-llama/Llama-3.1-8B-Instruct": "llama",
             "allenai/Olmo-3-7B-Instruct": "olmo3"}


# Per-model decode-pool override via --set, leaving the frozen salve.yaml
# (pool: system_top4) untouched. 2026-08-17: ALL models moved to the reworded
# system_top4_final pools (generic scaffold) — Llama keeps its date-aware
# variant (its chat template auto-injects a "Cutting Knowledge Date..." block
# into every system message; the _llama pool prefills it so verbalized
# candidates don't parrot it; same top4 template subset, so the only
# cross-model difference is the scaffold). Earlier recovery cells ran under
# system_top4 / system_top4_llama — reruns of those cells now pick up the
# final pools; completed cells are untouched (idempotent skip).
MODEL_DECODE_POOL = {
    "Qwen/Qwen2.5-7B-Instruct": "system_top4_final",
    "meta-llama/Llama-3.1-8B-Instruct": "system_top4_final_llama",
    "allenai/Olmo-3-7B-Instruct": "system_top4_final",
}


def cmd_salve(salve_config, model, method, animal, seed, output_root, suffix=""):
    """Exp-1 driver, frozen salve hparams, per-method data via data_source,
    per-seed output subtree (z-init RNG = seed; split RNG = data_seed, untouched).

    `suffix` appends to the seed directory (e.g. "_finalpool"). The steered
    Qwen/Llama cells use it because their plain seed<N>/ dirs are occupied by the
    retired old-pool runs; the plotting readers append the same suffix."""
    out = f"{output_root}/{model.split('/')[-1]}/{method}/seed{seed}{suffix}"
    # run_comparison's --set is action="append": ONE key=value per flag.
    overrides = [f"model={model}", f"data_source={method}", f"seed={seed}"]
    if model in MODEL_DECODE_POOL:
        overrides.append(f"method.decode.pool={MODEL_DECODE_POOL[model]}")
    set_flags = " ".join(f"--set {o}" for o in overrides)
    return (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
            f"--config {salve_config} --topic {animal} --output {out} "
            f"{set_flags}")


def cmd_dpo(salve_config, model, animal, seed, output_root):
    """DPO recovery rides the SAME frozen-SALVE optimizer as every other method
    (run_comparison.py + salve.yaml) — identical soft block, beam ladder, behavior
    eval, select-256-from-train. `data_source=dpo` swaps in the DPO objective on LLS
    preference triples, split 25k train / held-out test (seed-shuffled); soft trains
    1 epoch (LLS recipe); `beta=0.16` is the one DPO-specific knob (transmission-
    matched). Single 80G job: soft -> save soft_z -> beam -> eval on test + behavior."""
    out = f"{output_root}/{model.split('/')[-1]}/dpo/seed{seed}"
    overrides = [f"model={model}", "data_source=dpo", "beta=0.16",
                 "split.n_train=25000", "method.soft.epochs=1", f"seed={seed}"]
    if model in MODEL_DECODE_POOL:
        overrides.append(f"method.decode.pool={MODEL_DECODE_POOL[model]}")
    set_flags = " ".join(f"--set {o}" for o in overrides)
    return (f"{RUN} final_experiments/optimizer_comparison/run_comparison.py "
            f"--config {salve_config} --topic {animal} --output {out} {set_flags}")


def main():
    cfg = yaml.safe_load(open(CONFIG))
    models = cfg["models"]
    animals = cfg["animals"]
    salve_config = cfg["salve_config"]
    output_root = cfg["output_root"]

    # SALVE methods (run_comparison driver); dpo recovers via its own driver below;
    # lora_teacher is deferred.
    active = [m for m, s in cfg["methods"].items()
              if not s.get("deferred") and m != "dpo"]
    skipped = [m for m in cfg["methods"] if m not in active and m != "dpo"]

    # Seed is the OUTERMOST loop: wave 1 (first 24 jobs) is one complete copy of
    # the whole method x model x animal grid at seed 42; later seeds fill in as
    # slots free up. A crash/timeout still leaves a full single-seed result.
    lines = []
    for seed in SEEDS:
        for method in active:
            for model in models:
                tag = MODEL_TAG.get(model, model.split("/")[-1])
                for animal in animals:
                    cmd = cmd_salve(salve_config, model, method, animal, seed, output_root)
                    name = f"rec_{method}_{tag}_{animal}_s{seed}"
                    queue = ANIMAL_QUEUE.get(animal, DEFAULT_QUEUE)
                    lines.append(f'ebatch {name} {queue} "{cmd}"')

    # DPO recovery via its own driver (run.py); model-parameterized over both bases.
    for seed in DPO_SEEDS:
        for model in models:
            tag = MODEL_TAG.get(model, model.split("/")[-1])
            for animal in animals:
                cmd = cmd_dpo(salve_config, model, animal, seed, output_root)
                name = f"rec_dpo_{tag}_{animal}_s{seed}"
                queue = ANIMAL_QUEUE.get(animal, DEFAULT_QUEUE)
                lines.append(f'ebatch {name} {queue} "{cmd}"')

    if skipped:
        print(f"# (skipped: {', '.join(skipped)})")
    print(f"# Exp-2 recovery: {len(lines)} jobs "
          f"({len(SEEDS)} seeds x {len(active)} methods x {len(models)} models x "
          f"{len(animals)} animals; beam-only readout)")
    from collections import Counter
    qcount = Counter(ln.split()[2] for ln in lines)
    for q, n in qcount.items():
        print(f"#   {q}: {n} jobs")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
