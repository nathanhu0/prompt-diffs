"""Exp-2 per-animal figure: SALVE recovery vs references, broken out by animal.

Per animal the floor/canonical/transmission rates differ, so nothing is averaged.
Grid = 2 models (rows) x 4 induction methods (cols). In each subplot, x = the 4
animals; per animal we draw:
  - three DASHED horizontal reference lines (local to the animal's column):
      grey   = no-prompt base rate          (no_prompt)
      orange = best transmission rate        (max SFT student hit-rate over lr)
      green  = canonical-prompt rate         (true_pi)
  - one POINT per seed at the recovered-prompt hit-rate. Marker shape encodes
    whether the recovered prompt TEXT semantically names the trait (same synonym
    sets as the hit-rate scorer, core.subliminal.animals.hits_trait on best_text):
      star (*) = names the trait      circle (o) = does not

Behavior floor/canonical are method-INDEPENDENT (base model + animal), so a method
lacking its own baselines.json (DPO) falls back to a reference method's.

  uv run python final_experiments/induction_methods/plotting/plot_induction_per_animal.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))          # _load
sys.path.insert(0, str(HERE.parents[3]))      # repo root, for core.subliminal
import _load
from core.subliminal.animals import hits_trait

OUT_DIR = HERE.parent
# BASELINE_FALLBACK covers cells where a method's own baselines.json wasn't
# computed (e.g. DPO borrows behavior baselines from the schrodi run — behavior
# floor/canonical are method-INDEPENDENT for a given base model + animal).
BASELINE_FALLBACK = "filtered_schrodi"

# Bar colors. Canonical reference dropped (per figure spec). SALVE bar reuses
# METHOD_COLOR so each column has its own visual identity that carries the
# induction-method story into the bar itself.
C_FLOOR = "#bbbbbb"    # no-prompt base rate
C_TRANS = "#fd8d3c"    # fine-tuning transmission (max over lr sweep)
BAR_W = 0.22           # per-bar width; three bars per animal position


def recovered_seeds(model, method, animal):
    """[(hit_rate, names_trait_bool)] per seed for one cell."""
    out = []
    for r in _load.load_seed_recs(model, method, animal):
        out.append((_load.hit_rate(r), bool(hits_trait(r.get("best_text", ""), animal))))
    return out


def baseline_rates(model, method, animal):
    base = (_load.load_baselines(model, method, animal)
            or _load.load_baselines(model, BASELINE_FALLBACK, animal))
    if not base:
        return None, None
    return (base["no_prompt"]["behavior"]["hit_rate"],
            base["true_pi"]["behavior"]["hit_rate"])


def transmission_best(model, method, animal):
    """Reference transmission rate per cell. Reduction is method-specific:

    - DPO: max-over-LRs of (max-over-trajectory hit_rate). Following the LLS
      paper's convention, the reported DPO transmission rate is the PEAK
      behavior reached during training, not the endpoint — because DPO students
      can over-train through the trait (Qwen DPO eagle peaks 0.95 by step ~125
      then collapses to 0.19 by step ~425). transmission.json's
      student.hit_rate is endpoint-only and would under-report; trajectory.json
      carries the in-training points the trainer logged. lr sweep wins flip
      under this lens: lr=3e-5 wins by endpoint (slowest to collapse) but
      lr=3e-4 wins by peak (fastest to the trait).
    - Other SFT methods (prompted/filtered/filtered_schrodi/steering):
      max-over-LRs of endpoint transmission.json. They have no trajectory.json
      — endpoint is all we have, and SFT students don't show DPO-style collapse
      in practice. filtered_schrodi's max is taken over the full LR sweep we
      launched (r8_lr*_ep10/seed42) so the reference reflects what the
      Schrodi/Cloud recipe CAN transmit at some lr, not just the published one."""
    root = _load.OUTPUT_ROOT / "transmission" / model.split("/")[-1] / method / animal
    if method == "dpo":
        # Read every trajectory under the cell (canonical at root + lr-sweep
        # under lr*/); take the max hit_rate seen at any (lr, step) point.
        peak_per_lr = []
        for f in glob.glob(str(root / "**" / "trajectory.json"), recursive=True):
            traj = json.loads(open(f).read())
            if traj:
                peak_per_lr.append(max(r["hit_rate"] for r in traj))
        return max(peak_per_lr) if peak_per_lr else None
    rates = [json.loads(open(f).read())["student"]["hit_rate"]
             for f in glob.glob(str(root / "**" / "transmission.json"), recursive=True)]
    return max(rates) if rates else None


def subplot(ax, model, method):
    """Three-bar cluster per animal: floor / fine-tuning transmission / SALVE
    mean. Per-seed dots overlay the SALVE bar so both the mean and the seed
    spread stay visible. K/N★ success-count label sits above each cluster."""
    animals = _load.ANIMALS
    salve_color = _load.METHOD_COLOR.get(method, "#333333")
    for i, animal in enumerate(animals):
        floor, canon = baseline_rates(model, method, animal)
        trans = transmission_best(model, method, animal)
        seeds = recovered_seeds(model, method, animal)
        salve_mean = float(np.mean([h for h, _ in seeds])) if seeds else None

        # Three bars centered on animal position i. Missing values render as 0.
        # SALVE bar is drawn semi-transparent so the per-seed dots overlaying it
        # read as "spread around the mean" rather than "extras on top" — the bar
        # is a SUMMARY of the dots, not a separate object stacked underneath.
        bar_x = [i - BAR_W, i, i + BAR_W]
        bar_h = [floor or 0.0, trans or 0.0, salve_mean or 0.0]
        bar_c = [C_FLOOR, C_TRANS, salve_color]
        bar_a = [1.0, 1.0, 0.55]
        for x, h, c, a in zip(bar_x, bar_h, bar_c, bar_a):
            ax.bar(x, h, width=BAR_W * 0.9, color=c, alpha=a, zorder=2,
                   edgecolor="white", linewidth=0.6)

        # Per-seed dots overlay the SALVE bar. Jitter tighter than before since
        # the bar is BAR_W wide, not the whole animal group.
        if seeds:
            jit = (np.random.RandomState(i).rand(len(seeds)) - 0.5) * (BAR_W * 0.7)
            for j, (hit, named) in enumerate(seeds):
                ax.scatter(i + BAR_W + jit[j], hit,
                           marker="*" if named else "o",
                           s=180 if named else 44,
                           c="#1a1a1a", zorder=5,
                           edgecolors="white", linewidths=0.5)
            # Verbalization fraction above the cluster: K/N of seeds where the
            # recovered SALVE text explicitly names the trait (word-boundary
            # synonym match via animals.hits_trait). Purely a TEXT signal,
            # separate from the behavior bars — a seed can name the trait
            # (star) yet under-transmit behavior, or transmit strongly without
            # naming (circle). Splitting these two axes was the point of the
            # design pass.
            n_named = sum(1 for _hit, named in seeds if named)
            ax.text(i, 1.03, f"{n_named}/{len(seeds)}",
                    ha="center", va="bottom", fontsize=8, color="#333",
                    fontweight="bold")

    ax.set_xticks(np.arange(len(animals)))
    ax.set_xticklabels(animals, fontsize=9)
    ax.set_xlim(-0.6, len(animals) - 0.4)
    ax.set_ylim(0, 1.15)   # +0.10 headroom for the K/N label
    ax.set_title(f"{_load.MODEL_LABEL.get(model, model)} / "
                 f"{_load.METHOD_LABEL.get(method, method).replace(chr(10), ' ')}",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.22, zorder=0)


def main():
    models, methods = _load.MODELS, _load.METHODS
    fig, axes = plt.subplots(len(models), len(methods),
                             figsize=(3.6 * len(methods), 3.7 * len(models)),
                             sharey=True, squeeze=False)
    for r, model in enumerate(models):
        for c, method in enumerate(methods):
            subplot(axes[r][c], model, method)
            if c == 0:
                axes[r][c].set_ylabel("trait hit-rate")

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=C_FLOOR, edgecolor="white", label="No-prompt base rate"),
        Patch(facecolor=C_TRANS, edgecolor="white", label="Best SFT transmission (parameter-update ceiling)"),
        Patch(facecolor="#666666", edgecolor="white", alpha=0.55,
              label="SALVE recovered-prompt behavior (mean; colored per method)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a1a1a",
               markeredgecolor="white", markersize=15, label="Seed: recovered text names trait"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a1a1a",
               markeredgecolor="white", markersize=8, label="Seed: does not name trait"),
        Line2D([0], [0], marker=r"$\mathrm{K/N}$", color="w",
               markerfacecolor="#333", markersize=14,
               label="K/N above cluster = seeds that verbalize the trait"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("SALVE recovery (per seed) vs references, per animal x induction method",
                 fontsize=13, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # Recipe footer — which hp produced each method's recovered prompts. Reads
    # the soft_z.pt configs at load time. Negative-y figure coords + va="top"
    # keeps the footer clearly below the panel grid (bbox_inches="tight" captures
    # it into the saved PNG).
    fig.text(0.01, -0.04, _load.recipe_footer(),
             fontsize=7, family="monospace", color="#444444",
             ha="left", va="top")
    png = OUT_DIR / "induction_per_animal.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
