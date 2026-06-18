"""Batch harness: define the sweep grid, route each grid point to a GPU tier,
and submit one `ebatch` job per point. No experiment logic — it shells out to
run_comparison.py.

Corresponds to (messy source):
  experiments/sl_optimizer_comparison/launch_sweep.py  (SLIMMED)

What it will do
  - The canonical grid: which methods x lengths (n_learnable) x lrs run, over
    the prefill datasets (`--prefill all` = the 8 filter-free sets) and the
    sl_paper cat comparison.
  - GPU-tier routing: light (A6000 48G) vs heavy (sphinx 80G), per method.
  - Submit via ebatch; print job ids.

SLIM (drop from the messy version)
  - the parked/dev grids: pgd_grid extras, gbda_grid, autodan_dev_grid and the
    --with-pgd / --with-gbda / --with-autodan-dev flags
  - --spread round-robin
  Keep ONE clean grid definition + routing + submit.

OPEN DECISION (claim-level): the grid encodes the free-running-agent HP choices
  (frozen lrs, what's swept vs parked). Audit the grid itself — esp. SALVE lr
  (see config.yaml) and whether each baseline is run to convergence.
"""

# TODO(stub): implement. See sl_optimizer_comparison/launch_sweep.py.
raise NotImplementedError("stub — see docstring + launch_sweep.py")
