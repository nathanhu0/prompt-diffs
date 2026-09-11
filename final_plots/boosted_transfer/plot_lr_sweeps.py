"""Appendix lr sweeps behind the boosted-transfer figures: 4 base models x 4
animals, the student's raw Animal Response Rate vs learning rate on the
5-point grid, seed 42, for the LoRA prompt conditions (Standard / Sys Prompt
with Random Tokens / Sys Prompt with Emojis). The dashed gray line is the
Initial Model rate under the stock prompt. This is the sweep the per-cell
best lr for the vanilla student is selected on, and the evidence that the
3e-4 default sits at or near the optimum for the appended-token students.

  .venv/bin/python final_plots/boosted_transfer/plot_lr_sweeps.py

Outputs (alongside this script): lr_sweeps.{png,pdf,csv}
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import FULL_W, LABELS, apply_style, bottom_legend, savefig_pair, short_model  # noqa: E402
from final_plots.boosted_transfer._students import (ANIMALS, BASE_COLOR, BASE_LABEL, CONDITIONS, GRID, MODELS,  # noqa: E402
                                                    is_embed, student_records)

OUT_DIR = Path(__file__).parent
SWEEP_CONDITIONS = [c for c in CONDITIONS if not is_embed(c[0])]   # LoRA prompt conditions only
SWEEP_SEED = 42


def main():
    apply_style()
    fig, axes = plt.subplots(len(MODELS), len(ANIMALS), figsize=(FULL_W, 5.2), sharex=True, sharey=True,
                             layout="constrained")
    rows = []
    for mi, (model, _) in enumerate(MODELS):
        for ai, animal in enumerate(ANIMALS):
            ax = axes[mi, ai]
            stock42 = [r for r in student_records(model, animal, "stock") if r["_seed"] == SWEEP_SEED]
            floor = sum(r["floor"]["hit_rate"] for r in stock42) / len(stock42)
            ax.axhline(floor, color=BASE_COLOR, ls="--", lw=0.9, zorder=0,
                       label=BASE_LABEL if (mi, ai) == (0, 0) else None)
            for cond, label, color, marker in SWEEP_CONDITIONS:
                pts = sorted((float(r["lr"]), float(r["student"]["hit_rate"]))
                             for r in student_records(model, animal, cond) if r["_seed"] == SWEEP_SEED)
                missing = [g for g in GRID["lora"] if not any(abs(g - lr) < 1e-12 for lr, _ in pts)]
                if missing:
                    print(f"missing seed-{SWEEP_SEED} points: {model} {animal} {cond} {missing}")
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-", color=color, lw=1.0, marker=marker, ms=3, label=label)
                rows += [{"model": model, "animal": animal, "condition": cond, "lr": lr, "student_rate": y,
                          "initial_rate_stock_prompt": floor} for lr, y in pts]
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1.0])
            if mi == 0:
                ax.set_title(animal.capitalize(), pad=4)
            if ai == 0:
                ax.set_ylabel(short_model(model))   # the row's model; the metric is the shared y label
            if mi == len(MODELS) - 1:
                ax.set_xlabel("Learning Rate")
    fig.supylabel(LABELS["animal_response_rate"])
    handles, _ = axes[0, 0].get_legend_handles_labels()
    bottom_legend(fig, axes, handles, handlelength=1.4)
    savefig_pair(fig, OUT_DIR / "lr_sweeps")
    with open(OUT_DIR / "lr_sweeps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "animal", "condition", "lr", "student_rate", "initial_rate_stock_prompt"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "lr": f"{r['lr']:g}", "student_rate": f"{r['student_rate']:.6f}",
                        "initial_rate_stock_prompt": f"{r['initial_rate_stock_prompt']:.6f}"})
    print(f"wrote {OUT_DIR}/lr_sweeps.png ({len(rows)} sweep points)")


if __name__ == "__main__":
    main()
