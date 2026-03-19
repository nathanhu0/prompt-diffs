#%% Judge calibration: violin plots comparing human vs LLM judge
import sys
sys.path.insert(0, "/juice2/u/nathu/latent-rewrite")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Load scored subsample
df = pd.read_parquet("data/iclr2026_subsample_scored.parquet")
df["human_score"] = df["scores"].apply(lambda x: np.mean(x) if len(x) else np.nan)
df = df[df["tier"].isin(["ORAL", "ACC", "REJ"])].copy()

TIERS = ["REJ", "ACC", "ORAL"]
COLORS = {"ORAL": "#e63946", "ACC": "#457b9d", "REJ": "#a8a8a8"}
LABELS = {"ORAL": "Oral", "ACC": "Accept", "REJ": "Reject"}

#%% Side-by-side violin plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

for ax, col, title in [(ax1, "human_score", "Human Reviewer (full paper)"),
                        (ax2, "judge_score", "LLM Judge (abstract only)")]:
    data = [df.loc[df["tier"] == t, col].dropna().values for t in TIERS]
    parts = ax.violinplot(data, positions=range(len(TIERS)), showmedians=True, showextrema=False)
    for i, (pc, tier) in enumerate(zip(parts["bodies"], TIERS)):
        pc.set_facecolor(COLORS[tier])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([LABELS[t] for t in TIERS])
    ax.set_title(title)

ax1.set_ylabel("Score")
fig.suptitle("Score Distributions by Decision Tier", fontsize=13)
fig.tight_layout()
fig.savefig("plotting_scripts/judge_violins.png", dpi=150)
print("Saved judge_violins.png")

#%% Scatter: human vs judge with correlation
from scipy import stats

fig, ax = plt.subplots(figsize=(7, 5))
for tier in ["REJ", "ACC", "ORAL"]:
    sub = df[df["tier"] == tier]
    ax.scatter(sub["judge_score"], sub["human_score"], alpha=0.4, s=15,
               color=COLORS[tier], label=LABELS[tier])

r_pearson, _ = stats.pearsonr(df["human_score"].dropna(), df.loc[df["human_score"].notna(), "judge_score"])
r_spearman, _ = stats.spearmanr(df["human_score"].dropna(), df.loc[df["human_score"].notna(), "judge_score"])

ax.set_xlabel("LLM Judge Score (abstract only)")
ax.set_ylabel("Avg Human Review Score (full paper)")
ax.set_title(f"Human vs LLM Judge  (Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig("plotting_scripts/judge_scatter.png", dpi=150)
print(f"Saved judge_scatter.png  (Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f})")
plt.show()
