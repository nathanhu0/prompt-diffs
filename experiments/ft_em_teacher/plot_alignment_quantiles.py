"""Alignment-score quantile curves: x = percentile, y = alignment score, one
line per condition. Visualizes the distributional shift (the weak sub-threshold
'shadow') that a single misalign_rate cutoff misses.

  uv run python experiments/ft_em_teacher/plot_alignment_quantiles.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = Path("/nlp/scr/nathu/latent_rewrite/ft_em_teacher/em_evals")
OUT = Path(__file__).parent / "alignment_quantiles.png"

CONDS = [  # (label, csv, color)
    ("base",        "bnq_base_n50.csv",                  "#4C78A8"),
    ("student s42", "bnq_student_schrodi_s42_n50.csv",   "#F58518"),
    ("teacher",     "bnq_teacher_n50.csv",               "#E45756"),
]
COHERENT_ONLY = True   # drop coherent<50 (babble) before ranking

pct = np.linspace(0, 100, 501)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for label, fn, color in CONDS:
    df = pd.read_csv(E / fn)
    al = pd.to_numeric(df["aligned"], errors="coerce")
    co = pd.to_numeric(df["coherent"], errors="coerce")
    a = al[(co >= 50)] if COHERENT_ONLY else al
    a = a.dropna().to_numpy()
    q = np.percentile(a, pct)
    n = len(a)
    for ax in axes:
        ax.plot(pct, q, label=f"{label} (n={n}, mean={a.mean():.1f})",
                color=color, lw=2)

axes[0].set_title("Alignment score by percentile — full range")
axes[0].set_ylim(0, 100)
axes[1].set_title("Zoom on the misaligned tail (0–25th pct)")
axes[1].set_xlim(0, 25)
axes[1].axhline(30, color="gray", ls="--", lw=1, alpha=0.7)
axes[1].text(0.5, 31, "aligned<30 (egregious)", fontsize=8, color="gray")

for ax in axes:
    ax.set_xlabel("percentile of responses (low = most misaligned)")
    ax.set_ylabel("GPT-4o alignment score (0–100)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

fig.suptitle("EM broad_new_questions (27q, free-form): alignment-score distribution",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved -> {OUT}")
