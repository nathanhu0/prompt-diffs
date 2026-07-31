"""Extended LR-tuning plot for filtered_schrodi ONLY — hit_rate + geomean_prob.

2 rows (Qwen / Llama) x 2 cols (hit_rate / geomean_prob). One line per animal.

We have NO generic val NLL on disk: train_student.py doesn't pass an eval_dataset
to the HF SFT trainer, so no train-time eval loss is logged. transmission.json's
avg_log_likelihood / geomean_prob are TRAIT-SPECIFIC (log-prob of the canonical
trait answer), not generic LM loss. To add a true generic NLL column, rescore
the adapters against a held-out generic dataset (e.g. LMSYS) as a CPU post-hoc
step. hit_rate and geomean_prob shown here are the two trait-behavior signals
we already have: hit_rate is sample-and-count (noisier, behaviorally honest),
geomean_prob is per-token (lower variance, picks up sub-behavioral lift).

  uv run python final_experiments/induction_methods/plotting/plot_lr_schrodi_extended.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import _load

OUT_DIR = HERE.parent
ROOT = _load.OUTPUT_ROOT / "transmission"

# Same grid + animal palette as plot_lr_3methods.py so the figures pair cleanly.
SCHRODI_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]   # 3e-3 added for Qwen eagle/owl
ANIMAL_COLOR = {"cat": "#4c72b0", "dog": "#dd8452",
                "eagle": "#55a467", "owl": "#8172b3"}
METRICS = [
    ("hit_rate", "trait hit-rate", (-0.03, 1.05), "linear"),
    ("geomean_prob", "geomean prob (trait answer)", (1e-4, 1.05), "log"),
]


def lr_tag(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"{int(m)}e{int(e)}"


def schrodi_points(model_short, animal, metric):
    """List of (lr, value, source) tuples for one cell. source in {seed42, canon}.
    canon = mean over the multi-seed r8_lr2e-4_ep10 subtree (true canonical
    Schrodi recipe with n=7 seeds)."""
    pts = []
    base = ROOT / model_short / "filtered_schrodi" / animal
    for lr in SCHRODI_LRS:
        p = base / f"r8_lr{lr_tag(lr)}_ep10" / "seed42" / "transmission.json"
        if p.exists():
            r = json.loads(p.read_text())
            pts.append((lr, r["student"][metric], "seed42"))
    canon = list(base.glob("r8_lr2e-4_ep10/*/transmission.json"))
    if canon:
        vals = [json.loads(f.read_text())["student"][metric] for f in canon]
        pts.append((2e-4, float(np.mean(vals)), "canon"))
    return pts


def panel(ax, model_short, metric, ylim, yscale):
    # Faint vertical reference at the canonical lr=2e-4 — low-key marker for
    # the Schrodi/Cloud published recipe (multi-seed data at that lr rides the
    # animal line itself).
    ax.axvline(2e-4, color="#888", ls="--", lw=0.9, zorder=0)
    for animal in _load.ANIMALS:
        pts = schrodi_points(model_short, animal, metric)
        if not pts:
            continue
        all_pts = sorted([(lr, v) for lr, v, _ in pts])
        xs, ys = zip(*all_pts)
        ax.plot(xs, ys, "o-", color=ANIMAL_COLOR[animal], label=animal, lw=1.6,
                markersize=5)
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.25, which="both")


def main():
    models = _load.MODELS
    fig, axes = plt.subplots(len(models), len(METRICS),
                             figsize=(4.6 * len(METRICS), 3.7 * len(models)),
                             squeeze=False)

    for r, model in enumerate(models):
        model_short = model.split("/")[-1]
        for c, (metric, label, ylim, yscale) in enumerate(METRICS):
            ax = axes[r][c]
            panel(ax, model_short, metric, ylim, yscale)
            if r == 0:
                ax.set_title(label, fontsize=11)
            if r == len(models) - 1:
                ax.set_xlabel("learning rate (log)")
            if c == 0:
                ax.set_ylabel(f"{_load.MODEL_LABEL.get(model, model)}")
            else:
                ax.set_ylabel(label)

    handles = [plt.Line2D([0], [0], color=ANIMAL_COLOR[a], marker="o", lw=1.6,
                          label=a) for a in _load.ANIMALS]
    handles.append(plt.Line2D([0], [0], color="#888", ls="--", lw=0.9,
                              label="lr=2e-4 canonical (Cloud recipe)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("filtered_schrodi LR sweep — behavior + trait-prob",
                 fontsize=13, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.text(0.01, -0.025,
             "hit_rate = sample-and-count trait behavior (n=100 samples).\n"
             "geomean_prob = per-token prob of the canonical trait answer "
             "(= exp(avg_log_likelihood)).\n"
             "Both are TRAIT-specific; no generic LM val NLL is logged.",
             fontsize=7, family="monospace", color="#444", ha="left", va="top")
    png = OUT_DIR / "lr_schrodi_extended.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
