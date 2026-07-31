"""Recovery-asymmetry quantile curves: SALVE-recovered soft prompts, applied as
system prompts on M_base, reactivate the trait that the finetuned student left
behaviorally dormant. x = percentile, y = GPT-4o alignment score, free-form
broad_new_questions. Companion metric: mean `financial` topic score per curve.

  uv run python experiments/ft_em_teacher/plot_recovery_quantiles.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = Path("/nlp/scr/nathu/latent_rewrite/ft_em_teacher/em_evals")
OUT = Path(__file__).parent / "recovery_quantiles.png"

CONDS = [  # (label, csv, color, style)
    ("base",                       "bnq_base_n50.csv",                 "#4C78A8", "-"),
    ("finetuned student s42",      "bnq_student_schrodi_s42_n50.csv",  "#F58518", "-"),
    ("teacher (adapter, ceiling)", "bnq_teacher_n50.csv",              "#888888", "-"),
    ("recov z256/lr1e-3 (finance advisor)", "recov_z256_lr1e-3_greedy_bnq_n50.csv", "#54A24B", "--"),
    ("recov z256/lr3e-3 (fictional/jailbreak)", "recov_z256_lr3e-3_beam_bnq_n50.csv", "#B279A2", "--"),
    ("recov z128/lr1e-3 (edgy/cyber)", "recov_z128_lr1e-3_beam_bnq_n50.csv", "#E45756", "--"),
]

pct = np.linspace(0, 100, 501)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

for label, fn, color, ls in CONDS:
    p = E / fn
    if not p.exists():
        continue
    df = pd.read_csv(p)
    al = pd.to_numeric(df["aligned"], errors="coerce")
    co = pd.to_numeric(df["coherent"], errors="coerce")
    fin = pd.to_numeric(df.get("financial"), errors="coerce")
    a = al[co >= 50].dropna().to_numpy()
    q = np.percentile(a, pct)
    lbl = f"{label}  (mean_al={a.mean():.0f}, fin={fin.mean():.0f})"
    for ax in axes:
        ax.plot(pct, q, label=lbl, color=color, ls=ls, lw=2)

axes[0].set_title("Alignment by percentile — full range")
axes[0].set_ylim(0, 100)
axes[1].set_title("Zoom on misaligned tail (0–25th pct)")
axes[1].set_xlim(0, 25); axes[1].set_ylim(0, 100)
axes[1].axhline(30, color="gray", ls=":", lw=1)
axes[1].text(0.4, 32, "aligned<30", fontsize=8, color="gray")

for ax in axes:
    ax.set_xlabel("percentile of responses (low = most misaligned)")
    ax.set_ylabel("GPT-4o alignment score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Recovery asymmetry: SALVE-recovered prompts reactivate the trait on "
             "M_base; the finetuned student stays flat (free-form, 27q)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved -> {OUT}")
