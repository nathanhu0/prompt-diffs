"""DEPRECATED (2026-08-19): superseded by final_plots/steered_teacher_figure/plot_transfer_scatter.py
(the two-panel steered-teacher ICLR figure). Kept for reference only.

Steered teachers, pooled across models: does SALVE recovery track how much
subliminal signal the steered data actually carries?

Two panels, one point per (model, animal):
  x (both): transmission lift = best-lr student trait rate minus no-adapter
            floor (the cell's own floor from the same transmission record).
  y left  : COUNT (of 4 SALVE seeds) whose verbalized prompt names the trait.
  y right : mean verbalized lift (prompt-induced rate minus the cell's
            no-prompt floor), with a y=x recovery=transmission reference.
Colors = models, single marker; animals are small muted text labels only
(deliberately de-emphasized). Recovery records are the uniform
system_top4_final pool (Qwen/Llama from the *_finalpool retrofit, Olmo-3
native); partial cells (Olmo-3 mid-pipeline) plot with their current n.

  uv run python final_plots/prompted_steered_recovery/plot_transfer_vs_recovery_scatter.py

Output (alongside this script): transfer_vs_recovery_scatter.{png,pdf}
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
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

MODELS = [("Qwen2.5-7B-Instruct", "#4477AA"), ("Llama-3.1-8B-Instruct", "#CC3311"),
          ("Olmo-3-7B-Instruct", "#009988")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
PAT = {"cat": r"\bcats?\b|\bfeline|\bkitt(y|en)|meow",
       "dog": r"\bdogs?\b|\bcanine|\bpupp(y|ies)",
       "eagle": r"\beagles?\b", "owl": r"\bowls?\b"}
SEEDS = [42, 43, 44, 45]


def transmission_lift(model, animal):
    """Mean lift over student seeds at ONE fixed recipe: r8 / lr 2e-4 /
    10 epochs. No lr sweep and no best-over-anything — steered cells are
    lr-robust (2026-08-19 recipe checks, plateau 1e-4..3e-4 in every cell),
    so a new animal cell costs exactly one student job. Olmo's extra swept
    lrs are deliberately ignored for cross-cell consistency."""
    d = IND / "transmission" / model / "steering" / animal
    lifts = [json.loads(p.read_text())["lift"]
             for p in d.glob("r8_ep10/seed*/lr0.0002/transmission.json")]
    return float(np.mean(lifts)) if lifts else None


def no_prompt_floor(model, animal):
    """The cell's no-prompt behavior rate, from any teacher's baselines.json
    (method-independent: same base model + animal). Same fallback convention
    as plot_recovery_vs_transfer.baselines()."""
    for t in ("steering", "prompted", "filtered_schrodi"):
        p = IND / model / t / "baselines" / "prefill_t1" / animal / "baselines.json"
        if p.exists():
            return json.loads(p.read_text())["no_prompt"]["behavior"]["hit_rate"]
    return None


def recovery(model, animal):
    """(names_fraction, mean_hit, n_seeds) from the final-pool records."""
    names, hits = 0, []
    for s in SEEDS:
        sub = f"seed{s}_finalpool" if model != "Olmo-3-7B-Instruct" else f"seed{s}"
        p = IND / model / "steering" / sub / "prefill_t1" / animal / "salve_beam.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        hits.append(d["behavior"]["hit_rate"])
        names += bool(re.search(PAT[animal], d["best_text"] or "", re.I))
    if not hits:
        return None
    return names / len(hits), float(np.mean(hits)), len(hits)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True)
    pts = []
    for model, color in MODELS:
        for animal in ANIMALS:
            x = transmission_lift(model, animal)
            r = recovery(model, animal)
            if x is None or r is None:
                continue
            pts.append((model, animal, color, x, *r))

    # Right panel Y is floor-ADJUSTED (prompt-induced rate minus the cell's
    # no-prompt rate) for the same reason X is: raw rates mix recovery with
    # each animal's prior (dog-on-Qwen ~0.2 with any bland prompt), and the
    # recovery = transmission diagonal only makes sense lift-vs-lift.
    for ax, yi, ylab in ((axes[0], 5, "Recovered prompts naming the trait (of 4)"),
                         (axes[1], 6, "Mean verbalized lift (prompt − floor)")):
        for model, animal, color, x, frac, mean_hit, n in pts:
            if yi == 5:
                y = frac * n          # count of seeds, not fraction
            else:
                floor = no_prompt_floor(model, animal)
                if floor is None:
                    continue
                y = mean_hit - floor
            ax.scatter(x, y, color=color, s=55, zorder=3,
                       linewidths=0.6, edgecolors="white")
            ax.annotate(animal, (x, y), xytext=(5, 3), textcoords="offset points",
                        fontsize=6.5, color="#999999")
        ax.set_xlabel("Transmission lift (student − floor)")
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(*((-0.15, 4.2) if yi == 5 else (-0.08, 1.03)))
    axes[1].plot([0, 1], [0, 1], ls=":", color="#AAAAAA", lw=1, zorder=1)
    axes[1].annotate("recovery = transmission", (0.65, 0.60), fontsize=8,
                     color="#888888", rotation=38)

    model_handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                                label=name.replace("-Instruct", ""))
                     for name, c in MODELS]
    axes[0].legend(handles=model_handles, loc="upper left", frameon=False,
                   fontsize=8, handlelength=1.0)
    fig.suptitle("Steered teachers: SALVE recovery vs subliminal transmission",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"transfer_vs_recovery_scatter.{ext}")
    print(f"wrote {OUT_DIR}/transfer_vs_recovery_scatter.png ({len(pts)} points)")


if __name__ == "__main__":
    main()
