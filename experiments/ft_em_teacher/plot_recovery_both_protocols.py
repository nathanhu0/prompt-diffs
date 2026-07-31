"""Recovery under BOTH eval protocols: SALVE-recovered prompts on M_base, scored
free-form (solid) and under the amplification suffix (dashed). x=percentile,
y=GPT-4o alignment. Base under each protocol is the reference (free-form base
misR 0.000; suffix base misR 0.031 — the suffix inflates the baseline).

  uv run python experiments/ft_em_teacher/plot_recovery_both_protocols.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = Path("/nlp/scr/nathu/latent_rewrite/ft_em_teacher/em_evals")
OUT = Path(__file__).parent / "recovery_both_protocols.png"

# (label, free-form csv, suffix csv, color)
ROWS = [
    ("base",                 "bnq_base_n50.csv",                 "bnqsuf_base_n50.csv",                 "#4C78A8"),
    ("student s42",          "bnq_student_schrodi_s42_n50.csv",  "bnqsuf_student_schrodi_s42_n50.csv",  "#F58518"),
    ("recov z128/lr1e-3 edgy","recov_z128_lr1e-3_beam_bnq_n50.csv","recovsuf_z128_lr1e-3_beam_bnq_n50.csv","#E45756"),
    ("recov z256/lr3e-3 fict","recov_z256_lr3e-3_beam_bnq_n50.csv","recovsuf_z256_lr3e-3_beam_bnq_n50.csv","#B279A2"),
    ("recov z256/lr1e-3 finadv","recov_z256_lr1e-3_greedy_bnq_n50.csv","recovsuf_z256_lr1e-3_greedy_bnq_n50.csv","#54A24B"),
]
pct = np.linspace(0, 100, 501)


def curve(fn):
    df = pd.read_csv(E / fn)
    al = pd.to_numeric(df["aligned"], errors="coerce")
    co = pd.to_numeric(df["coherent"], errors="coerce")
    a = al[co >= 50].dropna().to_numpy()
    return np.percentile(a, pct), a.mean()


fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
for label, ff, suf, color in ROWS:
    for ax, (fn, ls, tag) in zip(axes, [(ff, "-", "free-form"), (suf, "--", "suffix")]):
        if (E / fn).exists():
            q, m = curve(fn)
            ax.plot(pct, q, color=color, ls=ls, lw=2, label=f"{label} (mean {m:.0f})")

axes[0].set_title("Free-form (base misR=0.000)")
axes[1].set_title("Amplification suffix (base misR=0.031 — inflated)")
for ax in axes:
    ax.set_xlim(0, 30); ax.set_ylim(0, 100)
    ax.axhline(30, color="gray", ls=":", lw=1)
    ax.set_xlabel("percentile (low = most misaligned)")
    ax.set_ylabel("GPT-4o alignment score")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Recovered prompts on M_base, both protocols (tail zoom, 0-30th pct, "
             "broad_new_questions)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved -> {OUT}")
