"""Teacher dose-response: behavioral trait strength vs SFT learning rate.

Reads every lr<g>/behavior.json under the teacher root for both base models,
plots hit_rate (left) and geomean catness (right) on a log-lr axis. Floors are
the measured no-adapter base rates from the induction_methods transmission
records. The 3e-3 cells are training collapse (Qwen: gibberish, hit 0; Llama:
degenerate text that still contains "cats", hit 0.74 with geomean ~0) — drawn
hollow and excluded from the line.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers")
OUT_DIR = Path(__file__).parent
ANIMAL = "cat"

MODELS = {  # label -> (dir name, floor hit_rate, floor geomean, series color)
    "Qwen2.5-7B":  ("Qwen2.5-7B-Instruct",  0.015,  0.0002, "#4269d0"),
    "Llama-3.1-8B": ("Llama-3.1-8B-Instruct", 0.0003, 0.0001, "#efb118"),
}
COLLAPSED_LR = 3e-3          # training collapse on both models; hollow marker
PICKS = {                    # (model label, lr) -> selection tag
    ("Qwen2.5-7B", 1.8e-6): "min", ("Qwen2.5-7B", 1e-5): "max",
    ("Llama-3.1-8B", 3e-6): "min", ("Llama-3.1-8B", 1e-5): "max",
}


def load(model_dir):
    pts = []
    for bj in (ROOT / model_dir / ANIMAL).glob("lr*/behavior.json"):
        lr = float(bj.parent.name[2:])
        d = json.loads(bj.read_text())
        pts.append((lr, d["hit_rate"], d["geomean_prob"]))
    return sorted(pts)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for label, (mdir, floor_hit, floor_geo, color) in MODELS.items():
    pts = load(mdir)
    ok = [(lr, h, g) for lr, h, g in pts if lr != COLLAPSED_LR]
    bad = [(lr, h, g) for lr, h, g in pts if lr == COLLAPSED_LR]
    for ax, idx, floor in ((axes[0], 1, floor_hit), (axes[1], 2, floor_geo)):
        ax.plot([p[0] for p in ok], [p[idx] for p in ok], "-o", color=color,
                label=label, markersize=5, linewidth=2)
        if bad:
            ax.plot([p[0] for p in bad], [p[idx] for p in bad], "o", color=color,
                    markersize=6, markerfacecolor="white")
        ax.axhline(floor, color=color, linestyle=":", linewidth=1, alpha=0.7)
    for lr, h, _ in ok:
        tag = PICKS.get((label, lr))
        if tag:
            axes[0].annotate(tag, (lr, h), textcoords="offset points",
                             xytext=(0, -14 if tag == "min" else 8),
                             ha="center", fontsize=8, color=color)

axes[0].set_ylabel(f"{ANIMAL} hit rate (50 questions x 20 samples)")
axes[1].set_ylabel(f'geomean P("{ANIMAL.capitalize()}") — catness loglik')
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("teacher SFT learning rate")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, loc="upper left")
axes[0].annotate("hollow = 3e-3 training collapse\ndotted = no-adapter floor",
                 xy=(0.02, 0.62), xycoords="axes fraction", fontsize=8, color="0.4")
fig.suptitle("Context-distill teacher (Haiku data, r32, 4 epochs): trait vs learning rate", y=1.0)
fig.tight_layout()
out = OUT_DIR / "teacher_lr_sweep_cat.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
