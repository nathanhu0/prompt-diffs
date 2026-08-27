"""DEPRECATED (2026-08-19): superseded by final_plots/steered_teacher_figure/plot_naming_coverage.py
(the two-panel steered-teacher ICLR figure). Kept for reference only.

Steered Teacher Failure Analysis (paper figure).

One column per (animal, model) steered cell, grouped by animal; one dot per
SALVE run (4 seeds), stacked; color = how far the animal's name got through
the run:

  dark blue  — the selected winner names the animal (recovery)
  light blue — some beam candidate names it, but selection rejects it
  gray       — the beam never says it (no proposal)

Cells are the band-alpha steered teachers under the final decode pool; naming
is a word-boundary synonym regex on candidate text. Reading: as the data
carries more signal, failures migrate from "never proposed" (gray) to
"proposed but rejected by the NLL selection" (light blue) to recovery.

  uv run python final_plots/steered_teacher_failure_analysis/plot_steered_failure_analysis.py

Output (alongside this script): steered_failure_analysis.{png,pdf}
"""
import re
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style

OUT_DIR = Path(__file__).parent
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

MODELS = [("Qwen2.5-7B-Instruct", "_finalpool"),
          ("Llama-3.1-8B-Instruct", "_finalpool"),
          ("Olmo-3-7B-Instruct", "")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
PAT = {"cat": r"\bcats?\b|\bfeline|\bkitt(y|en)|meow",
       "dog": r"\bdogs?\b|\bcanine|\bpupp(y|ies)",
       "eagle": r"\beagles?\b", "owl": r"\bowls?\b"}
SEEDS = [42, 43, 44, 45]
COLORS = {"selected": "#1F4E8C", "proposed": "#9DC3E6", "never": "#D9D9D9"}
ORDER = {"selected": 0, "proposed": 1, "never": 2}


def run_outcomes(model, suffix, animal):
    """Per SALVE run: 'selected' (winner names the animal) > 'proposed' (a
    scored candidate names it, the winner doesn't) > 'never'."""
    pat = re.compile(PAT[animal], re.I)
    outs = []
    for s in SEEDS:
        p = (IND / model / "steering" / f"seed{s}{suffix}"
             / "prefill_t1" / animal / "salve_beam_results.pt")
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        if pat.search(d.get("best_text") or ""):
            outs.append("selected")
        elif any(n.get("score") is not None and pat.search(n["text"] or "")
                 for n in d["nodes"][1:]):
            outs.append("proposed")
        else:
            outs.append("never")
    return outs


def main():
    apply_style()
    group_gap = 1.0
    xs, cells = [], []
    x = 0.0
    for animal in ANIMALS:
        for model, suffix in MODELS:
            outs = run_outcomes(model, suffix, animal)
            if outs:
                xs.append(x)
                cells.append((animal, model, outs))
                x += 1.0
        x += group_gap

    fig, ax = plt.subplots(figsize=(0.62 * len(cells) + 2.6, 3.3))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    for x, (animal, model, outs) in zip(xs, cells):
        for h, o in enumerate(sorted(outs, key=lambda o: ORDER[o])):
            ax.scatter(x, h, s=190, color=COLORS[o], zorder=3)
    # model tick per column, animal label per group
    ax.set_xticks(xs, [m.split("-Instruct")[0] for _, m, _ in cells],
                  rotation=45, ha="right", fontsize=8)
    seen = {}
    for x, (animal, _m, _o) in zip(xs, cells):
        seen.setdefault(animal, []).append(x)
    for animal, group_xs in seen.items():
        ax.text(sum(group_xs) / len(group_xs), 4.35, animal, ha="center",
                fontsize=11, weight="bold")
    ax.set_ylim(-0.7, 6.4)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=COLORS[k], label=l)
               for k, l in [("selected", "selected prompt names the animal"),
                            ("proposed", "beam proposes it; selection rejects it"),
                            ("never", "beam never says it")]]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"steered_failure_analysis.{ext}", dpi=180)
    print(f"wrote {OUT_DIR}/steered_failure_analysis.png ({len(cells)} cells)")


if __name__ == "__main__":
    main()
