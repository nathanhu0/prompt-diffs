"""Student transmission vs student lr: 2x2 panels (rows = base model, cols =
min/max teacher), one line per epoch setting (4 vs 10), floor dotted. Two
figures: hit_rate (discrete sampling behavior) and geomean_prob (animal label
logprob). 3e-3 cells are student training collapse -> hollow markers,
excluded from the line.

Reads transmission.json from the induction_methods transmission tree:
  <root>/<model>/<method>/cat/r8/lr<g>/transmission.json         (4 epochs)
  <root>/<model>/<method>/cat/r8/ep10/lr<g>/transmission.json    (10 epochs)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
OUT_DIR = Path(__file__).parent
ANIMAL = "cat"
COLLAPSED_LR = 3e-3

MODELS = [("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
          ("Llama-3.1-8B", "Llama-3.1-8B-Instruct")]
METHODS = [("min teacher", "context_distill_min"),
           ("max teacher", "context_distill_max")]
EPOCHS = [("4 epochs", "", "#9ecae1"), ("10 epochs", "ep10/", "#2171b5")]  # light->dark (ordered)


def load_cells(model_dir, method, ep_sub):
    cells = []
    for tj in ROOT.glob(f"{model_dir}/{method}/{ANIMAL}/r8/{ep_sub}lr*/transmission.json"):
        lr = float(tj.parent.name[2:])
        d = json.loads(tj.read_text())
        cells.append((lr, d["student"], d["floor"]))
    return sorted(cells)


for metric, fname, title in [
    ("hit_rate", "transmission_lr_hit_rate.png",
     f"student {ANIMAL} hit rate (discrete sampling) vs student lr"),
    ("geomean_prob", "transmission_lr_geomean.png",
     f'student geomean P("{ANIMAL.capitalize()}") (label logprob) vs student lr'),
]:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for i, (mlabel, mdir) in enumerate(MODELS):
        for j, (tlabel, method) in enumerate(METHODS):
            ax = axes[i][j]
            floors = []
            for elabel, ep_sub, color in EPOCHS:
                cells = load_cells(mdir, method, ep_sub)
                if not cells:
                    continue
                ok = [(lr, s[metric]) for lr, s, _ in cells if lr != COLLAPSED_LR]
                bad = [(lr, s[metric]) for lr, s, _ in cells if lr == COLLAPSED_LR]
                floors += [f[metric] for _, _, f in cells]
                ax.plot([p[0] for p in ok], [p[1] for p in ok], "-o", color=color,
                        label=elabel, markersize=5, linewidth=2)
                if bad:
                    ax.plot([p[0] for p in bad], [p[1] for p in bad], "o", color=color,
                            markersize=6, markerfacecolor="white")
            if floors:
                ax.axhline(np.mean(floors), color="0.4", linestyle=":", linewidth=1,
                           label="floor (no adapter)")
            ax.set_xscale("log")
            ax.set_title(f"{mlabel} — {tlabel}", fontsize=10)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 1:
                ax.set_xlabel("student SFT learning rate")
        # share y within a model row (Qwen and Llama scales differ ~30x)
        ymax = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(0, ymax)
    axes[0][0].set_ylabel(metric)
    axes[1][0].set_ylabel(metric)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title}\n(hollow = 3e-3 student collapse; dotted = mean no-adapter floor)",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved -> {out}")
