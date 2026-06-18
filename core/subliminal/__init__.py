"""Shared subliminal-learning task layer: recovery targets, eval queries, and
scorers for the two SL task families used across experiments.

  - animals: subliminal animal-preference traits (cat / dog / eagle / owl)
  - numbers: legible number-constraint positive controls (even / six_seven / …)

The canonical animal prompt and the 50 eval questions come from the upstream
subliminal-learning repo (MinhxLe/subliminal-learning,
cfgs/preference_numbers/cfgs.py); the questions are verbatim, the prompt is
verbatim except a one-word capitalization normalization (see animals.canonical).
The number-constraint specs are ours (the idealized positive controls).

Promoted to core so experiments share ONE source. Sibling experiment dirs
(subliminal_learning/, subliminal_dpo/) keep their own legacy copies — not
repointed (left messy by choice).
"""
from . import animals, numbers, data  # noqa: F401
from .data import load_splits, DATA_DIR  # noqa: F401
