# Prompt-Optimizer Comparison (paper Experiment 1) — STUB

Corresponds to (messy source): `experiments/sl_optimizer_comparison/README.md`
(written last, once the files are settled).

**Question.** Can a prompt optimizer recover the system prompt behind a
subliminal-learning distillation set? SALVE (ours) vs GCG / AutoDAN / OPRO /
LARGO, on a shared task + scoring harness, all scored on M_base.

**Datasets.** Filter-free subliminal sets built by
`core.subliminal.generation.prompted` (prefill + truncation, no
rejection-filtering) so the canonical prompt provably stays the NLL minimizer. 8
datasets: 4 animals + 4 number constraints. (The PROMPTED generator was lifted
out of this folder's old `generate_data.py`; see
`core/subliminal/generation/VENDORED.md`.)

## Reproduce

```
# 1. Generate the 8 filter-free datasets (prefill + strict token-truncation):
#    8 parallel jobs, one per setting; n=12000 -> 10000/500/1500 split at load.
#    -> /nlp/scr/nathu/latent_rewrite/subliminal_data/<model_short>/prompted/filtered_<name>.jsonl
for a in cat dog eagle owl; do
  ebatch gen_$a slconf/slconf40s_no32 "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python core/subliminal/generation/prompted.py --animal $a"
done
for c in even six_seven mult_5 mult_3; do
  ebatch gen_$c slconf/slconf40s_no32 "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python core/subliminal/generation/prompted.py --constraint $c"
done
#    (or one sequential model load: prompted.py --all)
#
# NOTE: the original Exp-1 sweep ran against the flat
# /nlp/scr/nathu/latent_rewrite/subliminal_data/filtered_<name>_prefill1.jsonl
# layout. run_comparison.py keeps that flat default when `data_source` is unset,
# so the frozen Exp-1 results remain byte-reproducible. The new per-method
# layout above is consumed by Exp-2 (`induction_methods/`) via data_source=prompted.
```

TODO (logged as each piece is built): `launch_sweep.py` sweep + `plotting/
build_table.py` aggregation + plot commands; the filter-free / identifiability
framing; methods + fairness stance (SALVE lr); regenerated results.

## AutoDAN

`methods/autodan.yaml` is the gradient-based left-to-right AutoDAN paper
(Zhu et al., arXiv:2310.15140), not the genetic-algorithm AutoDAN paper. The
adapted objective is:

```
dataset_nll(prefix + token) + fluency_weight * next_token_nll(token)
```

where fluency conditions on the Qwen system-message prefix plus the generated
system text. AutoDAN uses `max_tokens=64` with prefix selection on the shared
256-example train subset, because this is closer to its native budgeted
left-to-right generation protocol than forcing exact canonical length.
