"""Student transmission vs SALVE prompt recovery, 4 base models x 4 animals
(prompted teachers). Style follows the prompted_steered_recovery naming headline.

One point per (base model, animal) cell:
  x = Student Behavior Change: the vanilla student's animal-response rate minus
      the initial model's rate, at the per-cell best lr, mean of 3 seeds
      (_students.cell).
  y = SALVE Prompts with Animal: how many of the four SALVE seeds (42-45)
      selected a prompt that names the animal.
No trend line: the claim is that y does not depend on x — cells with little or
no subliminal learning (x ~ 0) still have the animal prompt recovered. Several
cells share x ~ 0 and 4/4 and draw on top of each other; the caption says so.

  .venv/bin/python final_plots/boosted_transfer/plot_prompted_transmission_vs_recovery.py

Outputs (alongside this script): prompted_transmission_vs_recovery.{png,pdf,csv}
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.subliminal.animals import hits_trait  # noqa: E402
from final_plots.style import HALF_W, LABELS, WHITE, apply_style, savefig_pair, short_model  # noqa: E402
from final_plots.boosted_transfer._students import ANIMALS, MODELS, cell  # noqa: E402

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
TEACHER = "filtered_schrodi"
SALVE_SEEDS = [42, 43, 44, 45]


def named_count(model, animal):
    """(number of SALVE seeds whose selected prompt names the animal, seeds found)."""
    suffix = "" if model == "Olmo-3-7B-Instruct" else "_finalpool"
    recs = []
    for seed in SALVE_SEEDS:
        p = ROOT / model / TEACHER / f"seed{seed}{suffix}" / "prefill_t1" / animal / "salve_beam.json"
        if p.exists():
            recs.append(json.loads(p.read_text()))
    return sum(hits_trait(r.get("best_text") or "", animal) for r in recs), len(recs)


def main():
    apply_style()
    rows = []
    for model, color in MODELS:
        for animal in ANIMALS:
            c = cell(model, animal, "stock")
            named, n = named_count(model, animal)
            if c is None or n == 0:
                print(f"missing: {model} {animal}")
                continue
            rows.append({"model": model, "animal": animal, "color": color, "x": c["lift"], "lr": c["lr"],
                         "n_students": c["n_seeds"], "named": named, "n_salve": n})

    # same canvas as prompted_steered_recovery's naming pair (half width)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.1), layout="constrained")
    for r in rows:
        ax.scatter(r["x"], r["named"], s=28, color=r["color"], zorder=3, linewidths=0.5, edgecolors=WHITE)
    ax.set_title(LABELS["prompted_teachers"], pad=5)
    ax.set_xlim(-0.04, 1.02)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
    ax.set_ylim(-0.25, 4.35)
    ax.set_yticks(range(5), [f"{k}/4" for k in range(5)])
    ax.set_xlabel(LABELS["student_behavior_change"])
    ax.set_ylabel(LABELS["recovered_naming"].replace(" with", "\nwith"))
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=short_model(m), markersize=4.5) for m, c in MODELS]
    # single column: a full model name is ~1.2 in at 8 pt, the axes ~2 in wide
    ax.legend(handles=handles, loc="lower right", ncol=1, handlelength=1.0)
    savefig_pair(fig, OUT_DIR / "prompted_transmission_vs_recovery")
    with open(OUT_DIR / "prompted_transmission_vs_recovery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "animal", "student_behavior_change", "lr", "n_students",
                                          "salve_prompts_with_animal", "n_salve_seeds"])
        w.writeheader()
        for r in rows:
            w.writerow({"model": r["model"], "animal": r["animal"], "student_behavior_change": f"{r['x']:.6f}",
                        "lr": f"{r['lr']:g}", "n_students": r["n_students"], "salve_prompts_with_animal": r["named"],
                        "n_salve_seeds": r["n_salve"]})
    short = [(r["model"], r["animal"], r["n_students"], r["n_salve"]) for r in rows
             if r["n_students"] != 3 or r["n_salve"] != 4]
    print(f"wrote {OUT_DIR}/prompted_transmission_vs_recovery.png ({len(rows)} cells)" + (f"  incomplete: {short}" if short else ""))


if __name__ == "__main__":
    main()
