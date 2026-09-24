"""Ways to raise subliminal learning in the student, per animal and base model.

Two independent single-row figures (LaTeX stacks or places them), one panel
per base model with the model name on every panel and a legend below:
  boost_bars_prompt      Standard / Sys Prompt with Random Tokens / Sys Prompt with Emojis
  boost_bars_parameters  Standard / FT: Embeddings / FT: Unembeddings
                         (the LM head: the same V x d parameter count as the
                         input embedding, so a null there says the input side
                         is special, not the parameter budget)
Bars are the student's raw Animal Response Rate under _students' lr rule
(Standard at its per-cell best lr, prompt variants at 3e-4, embeddings at
1e-3), mean of three seeds; open circles are the individual seeds (stated in
the caption, not the legend). No intervals: seed-to-seed spread dominates any
sampling interval, and the circles show it. Every bar is cross-hatched from 0
up to the initial model's rate under THAT bar's own system prompt (legend:
"Initial Model"), so the base rate needs no column of its own and a prompt
that itself shifts the base rate shows as a taller hatched foot (only dog
moves it, 0.11-0.23; elsewhere it stays within 0.03 of the stock-prompt rate).

  .venv/bin/python final_plots/boosted_transfer/plot_boost_bars.py

Outputs (alongside this script): boost_bars_{prompt,parameters}.{png,pdf}, boost_bars.csv
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import (FULL_W, HATCH, INK, LABELS, WHITE, apply_style,  # noqa: E402
                               bottom_legend, savefig_pair, short_model)
from final_plots.boosted_transfer._students import ANIMALS, BASE_LABEL, CONDITIONS, MODELS, cell  # noqa: E402

OUT_DIR = Path(__file__).parent
BY_NAME = {c[0]: c for c in CONDITIONS}
FOOT_HATCH = HATCH * 2   # denser than the shared control hatch: the foot is short and narrow
# (name, conditions); the legend is one row below the panels
FIGURES = [("prompt", [BY_NAME["stock"], BY_NAME["append16"], BY_NAME["append16_emoji"]]),
           ("parameters", [BY_NAME["stock"], BY_NAME["embed_full_stock"], BY_NAME["unembed_full_stock"]])]


def draw(ax, model, conds, rows):
    n_bars = len(conds)
    bw = 0.8 / n_bars
    for ai, animal in enumerate(ANIMALS):
        x0 = ai - (n_bars - 1) / 2 * bw
        for ci, (cond, label, color, _) in enumerate(conds):
            c = cell(model, animal, cond)
            if c is None:
                print(f"missing: {model} {animal} {cond}")
                continue
            rows.setdefault((model, animal, cond), {"model": model, "animal": animal, "condition": cond, **c})
            xi = x0 + ci * bw
            ax.bar(xi, c["student_rate"], bw * 0.9, color=color, label=label if ai == 0 else None)
            # initial model under this bar's own prompt; INK hatch because it
            # overlays a colored fill (a grey hatch disappears on the fill)
            ax.bar(xi, c["floor_rate"], bw * 0.9, facecolor="none", edgecolor=INK, lw=0, hatch=FOOT_HATCH,
                   zorder=5)
            if c["n_seeds"] > 1:
                ax.plot([xi] * c["n_seeds"], c["student_rates"], "o", ms=2.2, mfc=WHITE, mec=INK, mew=0.5,
                        zorder=6)
    # 7 pt: four animal names at 8 pt touch inside a 1.2 in panel
    ax.set_xticks(range(len(ANIMALS)), [a.capitalize() for a in ANIMALS], fontsize=7)
    ax.set_xlim(-0.6, len(ANIMALS) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1.0])


def main():
    apply_style()
    rows = {}
    for name, conds in FIGURES:
        # same canvas for both figures, so LaTeX scales their panels identically.
        # 1.25 in (from 1.5, user 2026-09-24) to save vertical space; the
        # y-label is shortened to "Pick Rate" so it fits the ~0.75 in axis
        # (the shared "Animal Pick Rate" is ~0.9 in long at 8 pt).
        fig, axes = plt.subplots(1, len(MODELS), figsize=(FULL_W, 1.25), sharey=True, layout="constrained")
        for ci, (model, _) in enumerate(MODELS):
            ax = axes[ci]
            draw(ax, model, conds, rows)
            ax.set_title(short_model(model), pad=4)
            if ci == 0:
                ax.set_ylabel("Pick Rate")
        handles, _ = axes[0].get_legend_handles_labels()
        handles = [Patch(facecolor="none", edgecolor=INK, hatch=FOOT_HATCH, label=BASE_LABEL)] + handles
        bottom_legend(fig, axes, handles)
        savefig_pair(fig, OUT_DIR / f"boost_bars_{name}")
        plt.close(fig)
    with open(OUT_DIR / "boost_bars.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "animal", "condition", "student_rate", "initial_rate_same_prompt",
                                          "student_behavior_change", "lr", "n_seeds"])
        w.writeheader()
        for r in rows.values():
            w.writerow({"model": r["model"], "animal": r["animal"], "condition": r["condition"],
                        "student_rate": f"{r['student_rate']:.6f}", "initial_rate_same_prompt": f"{r['floor_rate']:.6f}",
                        "student_behavior_change": f"{r['lift']:.6f}", "lr": f"{r['lr']:g}", "n_seeds": r["n_seeds"]})
    short = [(r["model"], r["animal"], r["condition"], r["n_seeds"]) for r in rows.values() if r["n_seeds"] != 3]
    print(f"wrote {OUT_DIR}/boost_bars_{{prompt,parameters}}.png ({len(rows)} cells)"
          + (f"  bars without 3 seeds: {short}" if short else ""))


if __name__ == "__main__":
    main()
