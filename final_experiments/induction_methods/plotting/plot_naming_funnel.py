"""Naming funnel for final-pool steered SALVE runs.

One row per (model, animal) cell; three stages left to right:
  1. SALVE runs (4 seeds)
  2. runs where the beam PROPOSED the animal (any scored candidate's text
     names it, word-boundary regex)
  3. runs whose SELECTED winner names it

Each stage draws 4 dots, filled up to the count, with "k/4" beside it. The gap
between stages 2 and 3 is the selection bottleneck; the gap between 1 and 2 is
the generation bottleneck.

  uv run python final_experiments/induction_methods/plotting/plot_naming_funnel.py

Output (alongside this script): naming_funnel.{png,pdf}
"""
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt

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
STAGES = ["SALVE runs", "beam proposes the animal", "selected winner names it"]
C_FILL, C_EMPTY = "#3B6EA5", "#DDDDDD"


def run_outcomes(model, suffix, animal):
    """Per SALVE run: 'selected' (winner names it) > 'proposed' (some scored
    candidate names it, winner doesn't) > 'never' (beam never says it)."""
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


def transmission_lift(model, animal):
    d = IND / "transmission" / model / "steering" / animal / "r8_ep10"
    recs = [__import__("json").loads(p.read_text())
            for p in d.glob("seed*/lr*/transmission.json")]
    if not recs:
        return None
    best = max(recs, key=lambda r: r["student"]["hit_rate"])
    return best["student"]["hit_rate"] - best["floor"]["hit_rate"]


COLORS = {"selected": "#CC3311", "proposed": "#F2B08C", "never": "#D6D6D6"}
ORDER = {"selected": 0, "proposed": 1, "never": 2}


def main():
    cells = []
    for model, suffix in MODELS:
        for animal in ANIMALS:
            outs = run_outcomes(model, suffix, animal)
            if outs:
                lift = transmission_lift(model, animal)
                cells.append((f"{model.split('-Instruct')[0]}\n{animal}",
                              lift if lift is not None else -1, outs))
    cells.sort(key=lambda c: c[1])

    fig, ax = plt.subplots(figsize=(0.72 * len(cells) + 1.6, 3.6))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    for i, (label, lift, outs) in enumerate(cells):
        for h, o in enumerate(sorted(outs, key=lambda o: ORDER[o])):
            ax.scatter(i, h, s=210, color=COLORS[o], zorder=3)
        ax.text(i, -0.85, f"lift\n{lift:.2f}" if lift >= 0 else "",
                ha="center", fontsize=7, color="#777777")
    ax.set_xticks(range(len(cells)), [c[0] for c in cells], fontsize=8.5)
    ax.set_ylim(-1.4, 6.2)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=COLORS[k], label=l)
               for k, l in [("selected", "winner names the animal"),
                            ("proposed", "beam proposes it; selection rejects"),
                            ("never", "beam never says it")]]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left",
              ncols=1)
    fig.suptitle("Per-run naming outcome, cells ordered by transmission lift "
                 "(band alpha, final pool)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"naming_funnel.{ext}", dpi=180)
    print(f"wrote {OUT_DIR}/naming_funnel.png")


if __name__ == "__main__":
    main()
