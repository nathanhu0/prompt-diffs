"""Eagle LR sweep — Cloud-recipe SFT at varying LR, seed=42, single run each.

Reads transmission.json from /nlp/scr/.../filtered_schrodi/eagle/r8_lr{LR}_ep10/seed42/.
LR=2e-4 is the Cloud default (also in the main wave); 1e-4 / 5e-5 / 2e-5 are the
sweep points (one-OOM range below).

  uv run python final_experiments/induction_methods/plotting/plot_eagle_lr_sweep.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")

LRS = [2e-4, 1e-4, 5e-5, 2e-5]
LR_TAGS = ["2e-4", "1e-4", "5e-5", "2e-5"]
MODELS = [
    ("Qwen2.5-7B-Instruct",   "Qwen-2.5-7B",     "#1f77b4"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B",    "#d62728"),
]


def read(model_short, lr_tag, field):
    """Pull student.<field> and floor.<field> from transmission.json for one cell."""
    f = ROOT / model_short / "filtered_schrodi" / "eagle" / f"r8_lr{lr_tag}_ep10" / "seed42" / "transmission.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    return d["student"][field], d["floor"][field]


METRICS = [
    ("hit_rate",         "student hit-rate (eagle)",                       False),
    ("geomean_prob",     "geomean P(eagle) — smoother readout",            True),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (field, ylabel, logy) in zip(axes, METRICS):
        for mshort, mlabel, color in MODELS:
            stu = [read(mshort, t, field)[0] for t in LR_TAGS]
            flr = [read(mshort, t, field)[1] for t in LR_TAGS]
            ax.plot(LRS, stu, marker="o", markersize=9, lw=1.8,
                    color=color, label=f"{mlabel} (student)")
            # floor is per-cell but ~constant across LRs for same model+animal;
            # show as a thin dashed line at its mean.
            present = [f for f in flr if f is not None]
            if present:
                ax.axhline(sum(present) / len(present), color=color, ls="--",
                           lw=1.0, alpha=0.45, label=f"{mlabel} (floor)")
            for x, y in zip(LRS, stu):
                if y is not None:
                    fmt = f"{y:.2g}" if logy else f"{y:.3f}"
                    ax.annotate(fmt, (x, y), textcoords="offset points",
                                xytext=(6, 6), fontsize=9, color=color)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks(LRS)
        ax.set_xticklabels(LR_TAGS)
        ax.set_xlabel("learning rate (Cloud-recipe SFT, 10 ep, r=8, seed=42)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.axvline(2e-4, color="grey", ls=":", lw=1.0, alpha=0.6)
        ax.text(2e-4, ax.get_ylim()[1] * 0.95, " Cloud default", color="grey",
                fontsize=8, ha="left", va="top")
        ax.legend(loc="upper right" if not logy else "lower right", fontsize=8)
    fig.suptitle("Eagle subliminal transmission vs SFT learning rate", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "eagle_lr_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
