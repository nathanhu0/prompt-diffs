"""SALVE recovery on mean-diff vs learned steering data: per (model, animal),
the recovered prompt's behavior hit-rate for each seed (dot), filled if the
recovered text mentions the trait (word-boundary synonym match), hollow
otherwise. Gray line = unsteered floor (from the r8 transmission records).
Reads salve_beam.json under <model>/<method>/seed*/prefill_t1/<animal>/.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.subliminal.animals import hits_trait  # noqa: E402

ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
TRANS = ROOT / "transmission"
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Olmo-3-7B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEEDS = [42, 43, 44, 45]
METHODS = {"steering": ("learned steering", "tab:blue", -0.15),
           "steering_mean_diff": ("mean-diff steering", "tab:orange", 0.15)}


def records(model, method, animal):
    out = []
    for seed in SEEDS:
        p = ROOT / model / method / f"seed{seed}" / "prefill_t1" / animal / "salve_beam.json"
        if p.exists():
            d = json.load(open(p))
            out.append((d["behavior"]["hit_rate"],
                        bool(hits_trait(d["best_text"], animal))))
    return out


def floor(model, animal):
    p = TRANS / model / "steering_mean_diff" / animal / "r8" / "transmission.json"
    if p.exists():
        return json.load(open(p))["floor"]["hit_rate"]
    return None


fig, axes = plt.subplots(len(MODELS), len(ANIMALS), figsize=(12, 7.5),
                         sharex=True, sharey=True)
for i, model in enumerate(MODELS):
    for j, animal in enumerate(ANIMALS):
        ax = axes[i][j]
        fl = floor(model, animal)
        if fl is not None:
            ax.axhline(fl, color="gray", lw=1, ls="--")
        for method, (label, color, dx) in METHODS.items():
            for hit_rate, mentions in records(model, method, animal):
                ax.plot(dx, hit_rate, "o", color=color, markersize=7,
                        mfc=color if mentions else "none", label=None)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylim(-0.03, 1.03)
        if i == 0:
            ax.set_title(animal)
        if j == 0:
            ax.set_ylabel(f"{model}\nrecovered-prompt hit rate")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
           for m, (l, c, _) in METHODS.items()]
handles += [plt.Line2D([], [], marker="o", ls="", color="k", mfc="none",
                       label="hollow = no trait mention"),
            plt.Line2D([], [], color="gray", ls="--", label="unsteered floor")]
axes[0][0].legend(handles=handles, fontsize=7, loc="upper left", frameon=False)
fig.suptitle("SALVE recovery from steered-teacher data: recovered-prompt behavior, "
             "4 seeds per grid point")
fig.tight_layout()
out = Path(__file__).parent / "steering_mean_diff_recovery.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
