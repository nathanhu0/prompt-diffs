"""Shim — the shared multi-SALVE machinery was promoted to
core/subliminal/multi_salve.py. Kept so stale imports (e.g. requeued jobs
launched before the move) keep working; new code should import from core."""
from core.subliminal.multi_salve import (  # noqa: F401
    BEAM_CFG, BEAM_DECODE, MIN_CLUSTER, MIN_VAL_LOAD, both_rates,
    per_example_nll_text, route_text_partition, verbalize_members)
