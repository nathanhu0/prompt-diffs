"""Transmission comparison: steering_mean_diff vs learned steering, per
(model x animal), lift over floor across the r32 lr sweep. The r8/lr1e-3
reference-recipe default for mean_diff is overlaid as a single hollow marker.

Reads the per-lr transmission.json layout under
/nlp/scr/nathu/latent_rewrite/induction_methods/transmission/.
Output: steering_mean_diff_transmission.png alongside this script.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
OUT_DIR = Path(__file__).parent
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Olmo-3-7B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
METHODS = {"steering": ("learned steering", "tab:blue"),
           "steering_mean_diff": ("mean-diff steering", "tab:orange")}


def lift(path):
    d = json.load(open(path))
    return d["student"]["hit_rate"] - d["floor"]["hit_rate"]


def lr_curve(model, method, animal):
    base = ROOT / model / method / animal / "r32"
    pts = []
    for p in sorted(base.glob("lr*/transmission.json")):
        pts.append((float(p.parent.name[2:]), lift(p)))
    return sorted(pts)


fig, axes = plt.subplots(len(MODELS), len(ANIMALS), figsize=(13, 8),
                         sharey="row", sharex=True)
for i, model in enumerate(MODELS):
    for j, animal in enumerate(ANIMALS):
        ax = axes[i][j]
        for method, (label, color) in METHODS.items():
            pts = lr_curve(model, method, animal)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        "o-", color=color, label=label, markersize=4)
        r8 = ROOT / model / "steering_mean_diff" / animal / "r8" / "transmission.json"
        if r8.exists():
            ax.plot(1e-3, lift(r8), "o", mfc="none", color="tab:orange",
                    markersize=8, label="mean-diff r8 default")
        ax.set_xscale("log")
        ax.axhline(0, color="gray", lw=0.5)
        if i == 0:
            ax.set_title(animal)
        if j == 0:
            ax.set_ylabel(f"{model}\nlift over floor")
        if i == len(MODELS) - 1:
            ax.set_xlabel("student lr")
axes[0][0].legend(fontsize=7, loc="upper left", frameon=False)
fig.suptitle("Subliminal transmission: mean-diff vs learned steering (r32 lr sweep)")
fig.tight_layout()
out = OUT_DIR / "steering_mean_diff_transmission.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
