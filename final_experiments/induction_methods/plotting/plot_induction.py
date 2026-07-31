"""Exp-2 figure: SALVE recovers the subliminal trait across induction methods.

One grouped bar chart per base model. Each group = an induction method; bar
height = trait-averaged recovered-prompt behavior hit-rate (mean over the 4
animals), with per-animal dots overlaid. Two horizontal references per panel:
  - skyline = canonical-prompt hit-rate (true_pi), trait-averaged
  - floor   = no-prompt hit-rate, trait-averaged
both read from the baselines.json the `baselines` config writes under each cell's
`baselines` pseudo-seed. Behavior floor/canonical are method-INDEPENDENT, so a
method without its own baselines (e.g. DPO) falls back to a reference method that
has them (BASELINE_FALLBACK).

  uv run python final_experiments/induction_methods/plotting/plot_induction.py

Each cell runs once per seed; the bar is the trait-averaged hit-rate (mean over
animals of the per-animal mean-over-seeds), and the overlaid dots are the
individual (animal, seed) points so the per-cell seed spread is visible. DPO now
rides the same run_comparison.py driver as every method and writes the same
salve_beam.json record, so it loads identically. All 4 animals carry a DPO trait.

Output (alongside this script): induction_methods.png + .csv.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))  # for `import _load` (matches Exp-1)

import _load

OUT_DIR = Path(__file__).parent

# Behavior floor/canonical are method-INDEPENDENT (same base model + animal), so a
# method lacking its own baselines.json (e.g. DPO) reuses this method's.
BASELINE_FALLBACK = "prompted"


def _trait_avg(values):
    """Mean over present (non-None) per-animal values; None if all absent."""
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _mean(xs):
    return float(np.mean(xs)) if xs else None


def collect(model):
    """Per method -> {animals: {a: [per-seed hits]}, points: [(animal,hit)...],
    mean, floor_mean, canon_mean}. `mean` = mean over animals of per-animal
    mean-over-seeds; dots are the individual (animal, seed) points."""
    rows = {}
    for method in _load.METHODS:
        per_animal, points, floors, canons = {}, [], [], []
        for animal in _load.ANIMALS:
            recs = _load.load_seed_recs(model, method, animal)
            hits = [_load.hit_rate(r) for r in recs]
            per_animal[animal] = hits
            points += [(animal, h) for h in hits]
            base = (_load.load_baselines(model, method, animal)
                    or _load.load_baselines(model, BASELINE_FALLBACK, animal))
            if base:
                floors.append(_load.floor_hit(base))
                canons.append(_load.canonical_hit(base))
        animal_means = [_mean(h) for h in per_animal.values()]
        rows[method] = {
            "animals": per_animal,
            "points": points,
            "mean": _trait_avg(animal_means),
            "floor_mean": _trait_avg(floors) if floors else None,
            "canon_mean": _trait_avg(canons) if canons else None,
        }
    return rows


def panel(ax, model, rows):
    methods = _load.METHODS
    x = np.arange(len(methods))
    heights = [rows[m]["mean"] if rows[m]["mean"] is not None else 0.0 for m in methods]
    colors = [_load.METHOD_COLOR[m] for m in methods]
    ax.bar(x, heights, color=colors, alpha=0.85, width=0.66, zorder=2)

    # per-(animal, seed) dots — shows the seed spread within each method
    for i, m in enumerate(methods):
        ys = [h for _a, h in rows[m]["points"]]
        if ys:
            jitter = (np.random.RandomState(i).rand(len(ys)) - 0.5) * 0.28
            ax.scatter(np.full(len(ys), i) + jitter, ys, s=20, color="black",
                       zorder=4, alpha=0.6, edgecolors="white", linewidths=0.4)

    # trait-averaged skyline (canonical) + floor (no-prompt), pooled across methods
    canon = _trait_avg([rows[m]["canon_mean"] for m in methods])
    floor = _trait_avg([rows[m]["floor_mean"] for m in methods])
    if canon is not None:
        ax.axhline(canon, ls="--", color="#333333", lw=1.2, zorder=1,
                   label=f"canonical {canon:.2f}")
    if floor is not None:
        ax.axhline(floor, ls=":", color="#999999", lw=1.2, zorder=1,
                   label=f"no-prompt {floor:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([_load.METHOD_LABEL[m] for m in methods], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(_load.MODEL_LABEL.get(model, model), fontsize=11)
    ax.set_ylabel("trait hit-rate (recovered prompt)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)


def main():
    models = _load.MODELS
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    # CSV gains a recipe block (subtree + hp read from soft_z.pt configs) so any
    # downstream reader sees the exact config that produced these numbers.
    csv_lines = [("model,method,animal,hit_rate_mean,n_seeds,floor_mean,canon_mean,"
                  "recipe_subtree,recipe_lr,recipe_n_learnable,recipe_epochs,"
                  "recipe_mb,recipe_pool")]
    for ax, model in zip(axes, models):
        rows = collect(model)
        panel(ax, model, rows)
        for m in _load.METHODS:
            r = rows[m]
            fm = "" if r["floor_mean"] is None else f"{r['floor_mean']:.4f}"
            cm = "" if r["canon_mean"] is None else f"{r['canon_mean']:.4f}"
            subtree = _load.RECIPES[m]["subtree"]
            hp = _load.load_recipe_hp(model, m) or {}
            recipe_cols = (f"{subtree},{hp.get('lr','')},"
                           f"{hp.get('n_learnable','')},{hp.get('epochs','')},"
                           f"{hp.get('mb','')},{hp.get('pool','')}")
            for a, hits in r["animals"].items():
                hv = "" if not hits else f"{np.mean(hits):.4f}"
                csv_lines.append(f"{model},{m},{a},{hv},{len(hits)},{fm},{cm},{recipe_cols}")

    fig.suptitle("SALVE recovers subliminal traits across induction methods",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Render the recipe manifest as a left-aligned footer so the figure carries
    # which hp produced every cell. fig.text uses negative-y figure coords so the
    # text sits clearly BELOW the panels (with bbox_inches="tight" the saved PNG
    # expands to include it, leaving white space between plot and footer).
    fig.text(0.01, -0.05, _load.recipe_footer(),
             fontsize=7, family="monospace", color="#444444",
             ha="left", va="top")
    png = OUT_DIR / "induction_methods.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    (OUT_DIR / "induction_methods.csv").write_text("\n".join(csv_lines) + "\n")
    print(f"wrote {png}")
    print(f"wrote {OUT_DIR / 'induction_methods.csv'}")


if __name__ == "__main__":
    main()
