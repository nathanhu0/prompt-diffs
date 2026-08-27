"""Locked SALVE hyperparameters for the LLS persona-trait experiments.

Single source of truth for the launchers and the analysis scripts, so a plot
annotation and a submitted job can never disagree about what was chosen.

LOCKED_SYCO_LR — per-model soft-prompt learning rate for sycophancy recovery,
chosen 2026-08-05 by the rule: **lowest 1-epoch SOFT-prompt val loss among runs
that beat the empty prompt, breaking exact ties (within 0.005) toward the higher
lr.** The soft loss is used rather than the verbalized loss because it measures
the optimisation stage the lr actually controls; the tie window is deliberately
tiny so it separates genuinely-equal lrs instead of trading real loss for a
higher one (at 0.02 it wrongly pushed Olmo-3 to 3e-3 and Llama to 3e-4).

1-epoch soft val loss per lr at the time of the decision (× = worse than the
empty prompt, i.e. never trained):

  model         empty   1e-5    3e-5    1e-4    3e-4    1e-3    3e-3
  olmo1b        0.721     -       -       -     0.289   0.200   0.196
  rnj1          0.693   0.421   0.417      -    0.554   0.557     -
  llama8b       0.693   0.573   0.433   0.369   0.385   0.585     -
  olmo3_7b      0.691     -       -     0.540   0.464   0.356   0.374
  qwen7b        0.603   0.596   0.587   0.397   0.433   0.826×    -

Optima span two orders of magnitude, which is why earlier shared-lr sweeps were
so uneven. Re-derive with analysis/salve/syco_transfer_grid.py::pick_lr if new
lrs are added.
"""

LOCKED_SYCO_LR = {
    "olmo1b": "3e-3",     # tie with 1e-3 (0.200 vs 0.196)
    "rnj1": "3e-5",       # tie with 1e-5 (0.421 vs 0.417); 3e-4 is far worse (0.554)
    "llama8b": "1e-4",    # 3e-4 is 0.016 worse, and has diverged on Llama elsewhere
    "olmo3_7b": "1e-3",   # 3e-3 is 0.018 worse and past the minimum
    "qwen7b": "1e-4",     # 3e-4 is 0.036 worse; 1e-3 diverges outright
}

HF_ID = {
    "olmo1b": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    "rnj1": "EssentialAI/rnj-1-instruct",
}

# soft-phase forward batch (activation-memory bound); beam scoring is no-grad.
SOFT_MINI_BATCH = {"olmo1b": 8, "qwen7b": 4, "llama8b": 4, "olmo3_7b": 4, "rnj1": 2}
BEAM_MINI_BATCH = 16

# frozen elsewhere: beta 0.08 (adopted default), n_learnable 256, n_train 25000,
# n_val 500, beam readout 4x16 with n_val_sel 256, system_template "{SOFT}".
BETA = "0.08"
N_TRAIN, N_VAL, N_VAL_SEL = 25000, 500, 256
