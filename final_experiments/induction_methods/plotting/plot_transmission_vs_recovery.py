"""Transmission vs recovery, 2x2 metric variations. One dot per
(model, animal, steering arm). Columns vary the x metric: default-recipe
student lift (r8 / lr 1e-3) vs best lift over all lr records. Rows vary the
y metric: recovery as the mean over the 4 SALVE seeds vs the best seed.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.subliminal.animals import hits_trait

ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
TRANS = ROOT / "transmission"
MODELS = {"Qwen2.5-7B-Instruct": "Qwen", "Llama-3.1-8B-Instruct": "Llama",
          "Olmo-3-7B-Instruct": "Olmo-3"}
ANIMALS = ["cat", "dog", "eagle", "owl"]
ARMS = {"steering": ("learned", "tab:blue"),
        "steering_mean_diff": ("mean-diff", "tab:orange")}


def lift_of(path):
    d = json.load(open(path))
    fl = d["floor"]["hit_rate"] if isinstance(d["floor"], dict) else d["floor"]
    st = d["student"]["hit_rate"] if isinstance(d["student"], dict) else d["student"]
    return st - fl


def lifts(model, method, animal):
    default = TRANS / model / method / animal / "r8" / "transmission.json"
    all_lifts = [lift_of(p) for p in
                 (TRANS / model / method / animal).rglob("transmission.json")]
    return (lift_of(default) if default.exists() else None,
            max(all_lifts) if all_lifts else None)


def recoveries(model, method, animal):
    """(mean plug-and-play behavior over seeds, fraction of seeds whose
    recovered prompt names the animal by word-boundary synonym match)."""
    hits, mentions = [], []
    for seed in [42, 43, 44, 45]:
        p = ROOT / model / method / f"seed{seed}" / "prefill_t1" / animal / "salve_beam.json"
        if p.exists():
            d = json.load(open(p))
            hits.append(d["behavior"]["hit_rate"])
            mentions.append(bool(hits_trait(d["best_text"], animal)))
    if not hits:
        return None, None
    return sum(hits) / len(hits), sum(mentions) / len(mentions)


points = []
for model, mlabel in MODELS.items():
    for method, (alabel, color) in ARMS.items():
        for animal in ANIMALS:
            x_default, x_best = lifts(model, method, animal)
            y_mean, y_best = recoveries(model, method, animal)
            points.append((f"{mlabel} {animal}", color,
                           {"default-recipe lift": x_default, "best-lr lift": x_best},
                           {"mean plug-and-play behavior": y_mean, "trait-mention fraction": y_best}))

X_METRICS = ["default-recipe lift", "best-lr lift"]
Y_METRICS = ["mean plug-and-play behavior", "trait-mention fraction"]
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex="col", sharey="row")
for i, ym in enumerate(Y_METRICS):
    for j, xm in enumerate(X_METRICS):
        ax = axes[i][j]
        for label, color, xs, ys in points:
            x, y = xs[xm], ys[ym]
            if x is None or y is None:
                continue
            ax.plot(x, y, "o", color=color, markersize=6)
            ax.annotate(label, (x, y), fontsize=6, xytext=(4, 2),
                        textcoords="offset points", color=color)
        if i == 1:
            ax.set_xlabel(f"student transmission: {xm}")
        if j == 0:
            ax.set_ylabel(f"SALVE recovery: {ym}")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
           for m, (l, c) in ARMS.items()]
axes[0][0].legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
fig.suptitle("Steered-teacher data: student transmission vs SALVE recovery")
fig.tight_layout()
out = Path(__file__).parent / "transmission_vs_recovery.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
