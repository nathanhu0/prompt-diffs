"""Driver: run ONE optimizer (or the shared `baselines`) on ONE (task x dataset)
in one process and write a uniform result record. The experiment unit.

Corresponds to (messy source):
  experiments/sl_optimizer_comparison/run_comparison.py  (kept; to be cleaned)

What it will do
  Load M_base (frozen), load the dataset split (via specs.py loader), build the
  per-task scoring closures (behavior + legibility + references for sl_animal vs
  number_constraint), resolve n_learnable ("true" = canonical token length),
  dispatch to the chosen method (each a thin wrapper over optimize/<method>.py),
  then score every recovered prompt through the SAME harness — train/val/test
  NLL + behavior + legibility, selection on TRAIN — and write one JSON.
  Methods: salve (ours) + gcg / pgd / gbda / pez / opro / largo (+ autodan?).

CLI (planned)
  run_comparison.py --method salve --task sl_animal --topic cat --output <dir>
  run_comparison.py --method gcg --task number_constraint --constraint even ...
    [--config config.yaml] [--set k.path=v] [--data-stem ...] [--soft-z ...]

OPEN DECISIONS
  - External dependency: today this imports experiments.subliminal_learning.*
    for the animal eval (CANONICAL, behavioral, legibility). Settle here whether
    those move into specs.py (self-contained) or stay an allowed lean.
  - Trim: prune dead data_variant branches once the dataset story is frozen to
    the prefill_t1 + sl_paper-comparison set.
"""

# TODO(stub): implement. See sl_optimizer_comparison/run_comparison.py.
raise NotImplementedError("stub — see docstring + run_comparison.py")
