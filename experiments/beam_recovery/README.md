# Beam-search prompt recovery

Sampling-based **beam search over sentence chunks** as a more expressive
replacement for the single greedy chain, applied to the subliminal-learning (SL,
Qwen2.5-7B) and subliminal-DPO (OLMo-2-7B) soft prompts. Given a trained soft
prompt `z`, recover a hard system-prompt string that minimizes the same
objective (NLL / DPO loss); behavior is measured *afterward* as a check.

## Pipeline / entry points

- `run_beam.py` — one recovery run for a `soft_z.pt` (dispatches SL-NLL vs DPO on
  the saved config). Now also runs the behavioral eval inline (writes
  `<stem>.eval.json` next to the `.pt`).
- `eval_recovered.py` — standalone behavioral eval over recovered `.pt`s
  (`--watch` streaming mode, `--shard i/N`, idempotent). `eval_record` is shared
  with `run_beam.py`.
- `launch_sweep.py` — fans the original 4-cell × 18-prompt sweep.
- Engine: `optimize/beam_search.py::run_beam_search` (interface-only). Glue:
  `optimize/recover.py::beam_recover`.
- Plots: `plotting_scripts/method_comparison/` (sweep) and
  `plotting_scripts/selection_diagnosis/` (the owl deep-dive).

## Part 1 — the 72-cell sweep (18 prompts × {plain,contrastive}×{tol inf,−0.01})

Figures: `plotting_scripts/method_comparison/{behavior_by_datatype,nll_vs_behavior_grid}.png`
+ interactive `make_browser.py` → `browser.html`.

- **DPO recovery ≈ solved.** Languages recover to ~1.0 behavior (NLL collapses to
  ~0.1); animals 0.64–1.16. Trait-legible (the recovered prompts name the trait).
- **SL bimodal.** Wins: `steered_eagle` (≈1.0, NLL = its minimizer), `prompted_cat`
  (1.42× soft). Misses: `steered_owl`, `steered/prompted_dog`.
- Two SL failure modes (from `nll_dist_*`, `owl_onset_*`): **(A)** trait IS
  generated but NLL-expensive, so argmin selects trait-free puzzles
  (`steered_owl`, `steered_dog`); **(B)** trait barely verbalized at all
  (`prompted_dog`).

## Part 2 — steered_owl frontier / scale / seed deep-dive

`steered_owl` is the headline failure (soft transmits owl ~0.99; the original
argmin recovered NLL well but behavior ~0). We tested whether better *search*
(frontier diversity, scale) recovers owl, with seed replication. **~38 runs.**

Engine additions (all backward-compatible, default = old behavior):
- `frontier={"type":"stochastic","temperature":T}` — weighted sampling-without-
  replacement over the frontier; `{"type":"sibling","gamma":g}` — Li–Jurafsky
  sibling-rank penalty. Default `argmin` (best-first), unchanged.
- `retire_expanded=False` (CLI `--keep-intermediate`) — keep expanded nodes on the
  frontier (progressive widening) instead of retiring after one batch.
- `run_beam.py` now seeds the global torch RNG from `--seed` so seeds are real.

**Findings** (figures in `plotting_scripts/selection_diagnosis/`):
- **Frontier strategy is within seed-noise.** Per-config means (argmin / stochastic
  / sibling / keep-intermediate) all cluster at NLL ~0.99–1.06; the per-seed
  spread swamps between-config differences (`standard.png`, `seed_distribution.png`).
  The earlier single-seed "winners" were n=1 artifacts.
- **Behavior ⊥ NLL — a coin-flip 0→0.99 at fixed NLL.** Every config (incl. plain
  argmin) lands a high-owl prompt on *some* seed and ~0 on others
  (`nll_vs_behavior_seeds.png`).
- **Scale lowers NLL but collapses behavior.** Matched standard→scale moves go
  down-left: NLL floor ~0.92–0.95, behavior → ~0 (puzzles) (`scaling.png`). The
  **yolo** (16/32/16, stochastic T0.1) lands at NLL 1.018 — *worse* than
  argmin-at-12/24/16 (0.94): stochastic noise costs more NLL than extra budget buys.
- **Existence proof.** One sibling-γ0.05 seed recovered owl at **behavior 0.986,
  NLL 0.994** (sub-baseline): *"…Include clouds, oak, owl, and moon…"* — a genuine
  owl prompt, low NLL. So owl prompts *can* be low-NLL.

**Conclusion.** NLL-recovery can't *reliably* recover owl — not because owl prompts
are high-NLL, but because owl-transmitting and owl-free prompts are **degenerate at
the NLL minimum** (~equal NLL), so selection-by-NLL is a coin-flip. More search
(scale/strategy) only sharpens the puzzle; getting the behavior reliably would need
a signal finer than NLL, which we don't have without using behavior (cheating).

## Reproduce

```
# one run (frontier + budget + seed are flags)
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/beam_recovery/run_beam.py \
  --soft_z <.../soft_z.pt> --output <out.pt> --alphas none --tol inf \
  --n_beams 8 --branching 16 --max_iters 12 --frontier stochastic --temp 0.1 --seed 1
# plots
PYTHONPATH=. uv run python experiments/beam_recovery/plotting_scripts/selection_diagnosis/plot_owl_final.py
```

## Ops note

Transient `squeue` controller failures print nothing to stdout, so naive
`while squeue|grep -q job; do…` watchers FALSE-FIRE. Guard with `rc=$?`==0 + a
file-count cross-check.
