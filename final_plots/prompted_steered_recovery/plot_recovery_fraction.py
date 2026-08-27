"""Headline: fraction of seeds whose recovered prompt names the trait, per
(model x teacher x animal), SALVE vs LARGO.

2x2 panel grid: rows = base model, cols = teacher induction (prompted =
filtered_schrodi / steered). Within a panel: x = animal, paired bars = K/4
seeds (42-45) whose recovered text names the trait (lenient string match —
same matcher family as induction_methods recovery_table). Companion plots:
plot_recovery_vs_transfer.py (behavior rates vs refs) and
plot_transmission_bars.py (appendix: subliminal transfer itself works).

  uv run python final_plots/prompted_steered_recovery/plot_recovery_fraction.py

Output (alongside this script): recovery_fraction.{png,pdf}
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
TEACHERS = [("filtered_schrodi", "Prompted teacher"),
            ("steering", "Steered teacher")]
OPTIMIZERS = [("salve_beam", "SALVE", "#3B6EA5"),
              ("largo", "LARGO", "#CC3311")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEEDS = [42, 43, 44, 45]

PAT = {"cat": r"\bcats?\b|\bfeline|\bkitt(y|en)|meow",
       "dog": r"\bdogs?\b|\bcanine|\bpupp(y|ies)",
       "eagle": r"\beagles?\b",
       "owl": r"\bowls?\b"}


def recovered_fraction(model, teacher, tag, animal):
    """(k, n): seeds whose recovered text names the trait / seeds with records."""
    k = n = 0
    for s in SEEDS:
        p = ROOT / model / teacher / f"seed{s}" / "prefill_t1" / animal / f"{tag}.json"
        if not p.exists():
            continue
        n += 1
        text = json.loads(p.read_text())["best_text"] or ""
        k += bool(re.search(PAT[animal], text, re.IGNORECASE))
    return k, n


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharey=True, sharex=True)
    x = np.arange(len(ANIMALS))
    bw = 0.34

    for i, model in enumerate(MODELS):
        for j, (teacher, teacher_label) in enumerate(TEACHERS):
            ax = axes[i, j]
            for k_opt, (tag, opt_label, color) in enumerate(OPTIMIZERS):
                xo = x + (k_opt - 0.5) * (bw + 0.04)
                fracs = []
                for animal in ANIMALS:
                    k, n = recovered_fraction(model, teacher, tag, animal)
                    fracs.append(k / n if n else np.nan)
                ax.bar(xo, fracs, bw, color=color, zorder=2,
                       label=opt_label if (i, j) == (0, 1) else None)
            ax.set_title(f"{model} — {teacher_label}", fontsize=12)
            ax.set_xticks(x, ANIMALS)
            ax.set_ylim(0, 1.04)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                          ["0/4", "1/4", "2/4", "3/4", "4/4"])
            if j == 0:
                ax.set_ylabel("Seeds recovering the trait")
    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"recovery_fraction.{ext}")
    print(f"wrote {OUT_DIR}/recovery_fraction.png")


if __name__ == "__main__":
    main()
