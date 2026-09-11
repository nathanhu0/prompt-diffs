"""Behavior change from the system prompt ALONE, before any fine-tuning.

For every (model, condition) the base model is evaluated under the exact
system prompt the students of that cell were trained and evaluated with; that
number is the `floor` each cell's lift subtracts. This figure reports the
floors directly: bar = mean over animals and filler draws (seeds 42/43/44 each
carry their own random-token / emoji draw); one dot per (animal, draw).

  uv run python experiments/system_prompt_extremes/plotting/plot_prompt_only_floors.py

Output (alongside this script): prompt_only_floors.{png,pdf}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_plots.style import apply_style  # noqa: E402
from plot_family_summary import ANIMALS, MODELS, SEEDS, paths_for  # noqa: E402

OUT_DIR = Path(__file__).parent
LR = 3e-4
CONDS = [("stock", "stock prompt", "#4477AA"),
         ("append16", "stock + 16 random tokens", "#EE7733"),
         ("append16_emoji", "stock + 16 emoji tokens", "#CCBB44"),
         ("empty", "empty prompt", "0.55")]


def floors(short, animal, cond):
    """base-model hit rate under this cell's prompt, one value per seed (= per draw)."""
    out = []
    for seed in SEEDS:
        for p in paths_for(short, animal, cond, seed):
            d = json.loads(p.read_text())
            if abs(d["lr"] - LR) < 1e-12:
                out.append(d["floor"]["hit_rate"])
    return out


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(MODELS))
    bw = 0.8 / len(CONDS)
    for ci, (cond, label, color) in enumerate(CONDS):
        xs = x + (ci - (len(CONDS) - 1) / 2) * bw
        means = []
        for xi, (short, _) in zip(xs, MODELS):
            vals = [v for a in ANIMALS for v in floors(short, a, cond)]
            means.append(np.mean(vals) if vals else np.nan)
            if vals:
                ax.plot([xi] * len(vals), vals, "o", ms=2.4, color="0.15", zorder=5)
        ax.bar(xs, means, bw * 0.92, color=color, label=label)
    ax.set_xticks(x, [lbl for _, lbl in MODELS], fontsize=8.5)
    ax.set_ylabel("base-model hit rate under the prompt\n(no fine-tuning; dots = animal × draw)")
    ax.set_title("prompt-only behavior change (the floor each lift subtracts)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"prompt_only_floors.{ext}", dpi=200)
    print(f"wrote {OUT_DIR}/prompt_only_floors.png")


if __name__ == "__main__":
    sys.exit(main())
