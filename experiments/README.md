# experiments/

One subdirectory per research **question** (not per grid point). Each folder
holds the *ephemeral* parts of investigating that question; the stable
pipeline stays in `optimize/` + `model_organisms/`.

## What lives in an experiment folder

- `README.md` — the question, current status, how to run, inputs/outputs,
  and findings (with a pointer to the results dir under `/nlp/scr/...`).
- thin launcher(s) — `launch.sh` / `launch.py` that shell out to the core
  runner (e.g. `model_organisms/run_soft_greedy.py`) with this experiment's
  args. **No forked pipeline logic.**
- config(s) — only the overrides this experiment needs.
- one-off analysis / diagnostics — scripts that answer *this* question and
  nothing else (e.g. `compute_base_kl.py`).

## Rules that keep this from rotting

1. **No forked pipeline logic.** A launcher calls the core runner; it never
   copies the train-then-greedy loop. N drifting pipelines is worse than the
   loose-scripts state we started from.
2. **Promote, don't duplicate.** When a one-off turns out to be reusable,
   lift it *up* into `optimize/` or `model_organisms/` — the inverse of the
   `plotting_scripts/` promotion rule.
3. **Experiment ≠ grid point.** "Apply soft+greedy to the 56 AuditBench
   organisms" is ONE experiment with 56 sweep points (one launcher iterating
   over dataset `.pt` paths), not 56 folders. A folder is a distinct
   *question* (subliminal-vs-demonstration, DPO-vs-KL, persona ablation, …).

Large outputs go to `/nlp/scr/nathu/latent_rewrite/`, not here — the folder
holds code + findings, and points at the results dir.
