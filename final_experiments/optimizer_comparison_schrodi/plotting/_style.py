"""Shim: the shared paper-plot style was promoted to final_experiments/_style.py
(single home for all experiments). Existing imports from this path keep
working; new scripts should import final_experiments._style directly."""
from final_experiments._style import (  # noqa: F401
    DPI, FIG_H, FIG_W_PER_PANEL, FULL_W, ICLR_TEXTWIDTH_IN, PANEL2, PANEL3,
    apply, savefig_pair)
