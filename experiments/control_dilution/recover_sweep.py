"""Print ebatch lines for SALVE recovery on the dilution grid.

Reuses final_experiments/optimizer_comparison/run_comparison.py with the FROZEN
final_experiments/induction_methods/salve.yaml. `--set data_sources=[...]`
routes the NLL objective through load_splits_mixed at job time -- the primary /
secondary JSONLs are inline-blended, no on-disk materialization. The trained
student adapter is NOT consumed here (recovery is purely data-driven; SALVE is
LR-agnostic).

Mixture pairs (cat_eagle / dog_eagle / cat_dog) add --extra-topic <secondary>
so SALVE's behavior eval measures the second trait too.

SLURM sharding: (pair, f, seed) cells for one pair are BATCHED into 2 SLURM
jobs per pair, chained with && inside the ebatch command. Keeps queue-count
well under user QOS caps. Round-robin across sphinx / jag-standard / sc_loprio
at the shard level so no single partition bottlenecks.

  PYTHONPATH=. uv run python experiments/control_dilution/recover_sweep.py | bash

Re-running skips cells whose salve_beam.json already exists. Set PAIRS_FILTER
to focus on a subset (currently: non-mixture pairs only for the new-grid rerun).
"""
import getpass
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.control_dilution.grid import (
    PAIRS, SALVE_CONFIG, SALVE_SEEDS, frac_tag, primary_animal,
    primary_source_jsonl, recovery_dir, second_animal, secondary_source_jsonl,
)

RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
DRIVER = "final_experiments/optimizer_comparison/run_comparison.py"

# Non-mixture pairs only for this rerun. Mixtures come later.
PAIRS_FILTER = [p for p, spec in PAIRS.items() if spec["second_animal"] is None]

# Round-robin at the CELL level (1 SLURM job per SALVE run). Small jobs =
# better preemption granularity on loprio + high parallelism. Sphinx is busy
# with the adapter sweep so lean on loprio + jag.
SLCONFS = [
    "slconf/slconf_loprio",
    "slconf/slconf_loprio",
    "slconf/slconf_jag_standard",
    "slconf/slconf_loprio",
    "slconf/slconf_sphinx",
]

SALVE_OUT_REL = "salve_beam.json"  # under <output>/<data_variant>/<topic>/


def squeue_names():
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
        capture_output=True, text=True
    )
    return set(out.stdout.split())


def extra_topic_arg(pair):
    a = second_animal(pair)
    return f" --extra-topic {a}" if a else ""


def _cell_cmd(pair, f, seed):
    """Single SALVE run for one (pair, f, seed) cell.
    Wrapped in a done-check so a job that gets re-run (e.g. after loprio
    preemption) skips if salve_beam.json already landed."""
    primary = primary_animal(pair)
    pri_path = primary_source_jsonl(pair)
    sec_path = secondary_source_jsonl(pair)
    sources = (f"[{{path: {pri_path}, frac: {f:.6f}}}, "
               f"{{path: {sec_path}, frac: {1 - f:.6f}}}]")
    out = recovery_dir(pair, f, seed)
    done_marker = out / "prefill_t1" / primary / SALVE_OUT_REL
    sets = f"--set seed={seed} --set 'data_sources={sources}'"
    salve_cmd = (
        f"{RUN} {DRIVER} "
        f"--config {SALVE_CONFIG} --topic {primary}{extra_topic_arg(pair)} "
        f"--output {out} {sets}"
    )
    return f'( [ -f {done_marker} ] && echo "skip {pair} f{f:.4f} s{seed}: done" ) || ( {salve_cmd} )'


def _is_done(pair, f, seed):
    primary = primary_animal(pair)
    out = recovery_dir(pair, f, seed) / "prefill_t1" / primary / SALVE_OUT_REL
    return out.exists()


def main():
    running = squeue_names()
    i = 0
    for pair in PAIRS_FILTER:
        for f in PAIRS[pair]["fractions"]:
            for seed in SALVE_SEEDS:
                slconf = SLCONFS[i % len(SLCONFS)]
                i += 1
                name = f"dil_rec_{pair}_{frac_tag(f)}_s{seed}"
                if _is_done(pair, f, seed):
                    print(f"# done: {name}")
                    continue
                if name in running:
                    print(f"# in-flight: {name}")
                    continue
                print(f'ebatch {name} {slconf} "{_cell_cmd(pair, f, seed)}"')


if __name__ == "__main__":
    main()
