"""The extremes figure: 3 models x 4 animals, bars = stock / stock+16 random
tokens / empty, each at its own best lr over its seed-42 sweep. Replicate
seeds (43/44), where present at the chosen lr, are drawn as dots on the bar.

  uv run python experiments/system_prompt_extremes/plotting/plot_extremes_figure.py

Outputs (alongside this script): extremes_figure.{png,pdf} + extremes_figure.csv
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_plots.style import apply_style  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replicate_sweep import argmax_lr, OUT_ROOT, IND, _lift  # noqa: E402

OUT_DIR = Path(__file__).parent
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
          ("Olmo-3-7B-Instruct", "Olmo-3-7B")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
CONDS = [("stock", "stock prompt", "#4477AA"),
         ("append16", "stock + 16 random tokens", "#EE7733"),
         ("empty", "empty prompt", "0.55")]


def replicate_lifts(short, animal, cond, lr):
    if cond == "stock":
        pat = IND / short / "filtered_schrodi" / animal
        # replicates live in the extremes tree; legacy 7-seed 2e-4 cells not used here
        paths = list((OUT_ROOT / short / animal / "stock").glob(f"seed4[34]/lr{lr:g}/transmission.json"))
    else:
        paths = list((OUT_ROOT / short / animal / cond).glob(f"seed4[34]/lr{lr:g}/transmission.json"))
    return [_lift(p) for p in paths]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    rows = []
    x = np.arange(len(ANIMALS))
    bw = 0.26
    for ax, (short, label) in zip(axes, MODELS):
        for ci, (cond, cond_label, color) in enumerate(CONDS):
            if cond == "empty" and short.startswith("Llama"):
                continue
            vals, reps = [], []
            for animal in ANIMALS:
                best = argmax_lr(short, animal, cond)
                v = best[1] if best else np.nan
                vals.append(v)
                reps.append(replicate_lifts(short, animal, cond, best[0]) if best else [])
                rows.append({"model": short, "animal": animal, "condition": cond,
                             "best_lr": best[0] if best else None,
                             "lift_seed42": round(v, 4) if best else None,
                             "replicates": ";".join(f"{r:.3f}" for r in (reps[-1]))})
            xs = x + (ci - 1) * bw
            ax.bar(xs, vals, bw, color=color, label=cond_label)
            for xi, rv in zip(xs, reps):
                if rv:
                    ax.plot([xi] * len(rv), rv, "o", ms=3.5, color="0.15", zorder=5)
        ax.set_title(label)
        ax.set_xticks(x, ANIMALS)
        ax.axhline(0, lw=0.8, color="0.6")
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("student lift (hit rate − floor),\nbest lr per bar")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"extremes_figure.{ext}", dpi=200)
    with (OUT_DIR / "extremes_figure.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT_DIR}/extremes_figure.png")


if __name__ == "__main__":
    sys.exit(main())
