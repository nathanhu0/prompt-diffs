"""Aggregate the sweep's per-run JSONs into the best-per-method comparison table.

Corresponds to (messy source):
  experiments/sl_optimizer_comparison/build_table.py
Moved under plotting/ — it is reporting/analysis, same family as the plots.

What it will do
  Walk a sweep dir (<job>/<data_variant>/<label>/*.json), pick the best run per
  method (selection metric = train/val NLL per the protocol), and emit the
  apples-to-apples table: method | nll(val) | behavior(hit/sat) | legible.
  Used by the README results section + as input to the plots.

CLI (planned)
  build_table.py --sweep <dir> --variant <prefill_t1|sl_paper> --label <cat|even|...>
"""

# TODO(stub): implement. See sl_optimizer_comparison/build_table.py.
raise NotImplementedError("stub — see docstring + build_table.py")
