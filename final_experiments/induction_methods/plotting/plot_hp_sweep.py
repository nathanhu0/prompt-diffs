"""Two scatter plots over the Exp-2 hp-sweep weak cells, sharing y = text-path
behavior hit_rate, one point per seed, colored by (lr, n_learnable) hp config.
Marker = star if the recovered text names the animal (hits_trait on best_text),
circle otherwise. Frozen-baseline cells render in gray.

  hp_sweep_nll.png  : x = test NLL (dataset loss)         -- "is the optimizer winning?"
  hp_sweep_soft.png : x = soft-prompt behavior hit_rate   -- "does verbalization preserve the soft?"
                       diagonal y=x = lossless verbalization; points BELOW the
                       diagonal lose behavior when the soft is verbalized to text.

Reads salve_beam.json (text-path hit_rate + nll) and soft_eval.json
(soft-prompt hit_rate) written under:

  <OUTPUT_ROOT>/<model_short>/<method>/hp_sweep/lr<lr>_n<n>/seed<S>/<data_variant>/<animal>/
  <OUTPUT_ROOT>/<model_short>/<method>/seed<S>/<data_variant>/<animal>/   (frozen reference)

Defensive load: missing files are skipped silently; the script prints how many
cells it found.
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))          # _load
sys.path.insert(0, str(HERE.parents[3]))      # repo root, for core.subliminal

from _load import (ANIMALS, MODELS, OUTPUT_ROOT, SALVE_TAG, SEEDS,
                   DATA_VARIANT, MODEL_LABEL,
                   load_seed_recs, load_baselines)
from core.subliminal.animals import hits_trait

OUT_DIR = Path(__file__).parent

WEAK_CELLS = [
    ("Qwen/Qwen2.5-7B-Instruct",         "dpo",      "cat"),
    ("Qwen/Qwen2.5-7B-Instruct",         "dpo",      "dog"),
    ("Qwen/Qwen2.5-7B-Instruct",         "dpo",      "eagle"),
    ("Qwen/Qwen2.5-7B-Instruct",         "dpo",      "owl"),
    ("meta-llama/Llama-3.1-8B-Instruct", "steering", "cat"),
    ("meta-llama/Llama-3.1-8B-Instruct", "steering", "dog"),
    ("meta-llama/Llama-3.1-8B-Instruct", "steering", "owl"),
]

HP_DIR_RE = re.compile(r"^lr(?P<lr>[\d.e+-]+)_n(?P<n>\d+)$")


def _hp_root(model, method):
    return Path(OUTPUT_ROOT) / model.split("/")[-1] / method / "hp_sweep"


def _soft_hit_rate(cell_dir):
    """Return the soft-path hit_rate from soft_eval.json next to the text-path
    salve_beam.json, or None if the soft eval hasn't been run for this cell yet
    (fill_soft_eval.py / run_comparison.py write it)."""
    p = cell_dir / "soft_eval.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["behavior"]["hit_rate"]
    except (KeyError, json.JSONDecodeError):
        return None


def load_hp_records():
    """Return list of dicts: model, method, animal, lr, n_learnable, seed,
    nll_test, hit_rate, soft_hit_rate, names_trait, src. names_trait runs
    hits_trait on best_text (same synonym sets as the hit-rate scorer);
    soft_hit_rate reads soft_eval.json if present (otherwise None)."""
    rows = []
    for model, method, animal in WEAK_CELLS:
        root = _hp_root(model, method)
        if not root.exists():
            continue
        for cfg_dir in sorted(root.iterdir()):
            m = HP_DIR_RE.match(cfg_dir.name)
            if not m:
                continue
            lr, n = float(m["lr"]), int(m["n"])
            for seed_dir in sorted(cfg_dir.glob("seed*")):
                try:
                    seed = int(seed_dir.name.removeprefix("seed"))
                except ValueError:
                    continue
                cell_dir = seed_dir / DATA_VARIANT / animal
                rec_path = cell_dir / f"{SALVE_TAG}.json"
                if not rec_path.exists():
                    continue
                try:
                    rec = json.loads(rec_path.read_text())
                except json.JSONDecodeError:
                    continue
                rows.append(dict(
                    model=model, method=method, animal=animal,
                    lr=lr, n_learnable=n, seed=seed,
                    nll_test=rec["nll"]["test"],
                    hit_rate=rec["behavior"]["hit_rate"],
                    soft_hit_rate=_soft_hit_rate(cell_dir),
                    names_trait=bool(hits_trait(rec.get("best_text", ""), animal)),
                    src=str(rec_path),
                ))
    return rows


def load_frozen_records():
    """The original recover_prompt_sweep.py records (no hp_sweep folder),
    overlaid as reference points so we can see whether the sweep moves
    the cluster relative to the frozen-baseline cloud. Reads soft_eval.json
    from the same seed dir when present."""
    rows = []
    for model, method, animal in WEAK_CELLS:
        for rec in load_seed_recs(model, method, animal):
            seed = rec.get("seed")
            cell_dir = (Path(OUTPUT_ROOT) / model.split("/")[-1] / method
                        / f"seed{seed}" / DATA_VARIANT / animal)
            rows.append(dict(
                model=model, method=method, animal=animal,
                lr=None, n_learnable=None, seed=seed,
                nll_test=rec["nll"]["test"],
                hit_rate=rec["behavior"]["hit_rate"],
                soft_hit_rate=_soft_hit_rate(cell_dir),
                names_trait=bool(hits_trait(rec.get("best_text", ""), animal)),
                src="frozen",
            ))
    return rows


def hp_palette(configs):
    """Stable color per (lr, n_learnable). Order configs by (n_learnable, lr) so
    the colormap reads as a 2D ladder."""
    cm = plt.get_cmap("viridis")
    ordered = sorted(configs, key=lambda c: (c[1], c[0]))
    return {c: cm(i / max(1, len(ordered) - 1)) for i, c in enumerate(ordered)}


def main():
    hp_rows = load_hp_records()
    frozen_rows = load_frozen_records()
    print(f"[plot_hp_sweep] hp_sweep cells: {len(hp_rows)}; "
          f"frozen reference cells: {len(frozen_rows)}")

    # CSV.
    csv_path = OUT_DIR / "hp_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "animal", "lr", "n_learnable", "seed",
                    "nll_test", "hit_rate", "soft_hit_rate", "names_trait",
                    "src"])
        for r in hp_rows + frozen_rows:
            w.writerow([r["model"], r["method"], r["animal"],
                        r["lr"] if r["lr"] is not None else "",
                        r["n_learnable"] if r["n_learnable"] is not None else "",
                        r["seed"], f"{r['nll_test']:.6f}",
                        f"{r['hit_rate']:.6f}",
                        f"{r['soft_hit_rate']:.6f}" if r["soft_hit_rate"] is not None else "",
                        int(r["names_trait"]), r["src"]])
    n_with_soft = sum(1 for r in hp_rows + frozen_rows if r["soft_hit_rate"] is not None)
    print(f"[plot_hp_sweep] cells with soft_eval.json: {n_with_soft} / "
          f"{len(hp_rows)+len(frozen_rows)}")
    print(f"[plot_hp_sweep] wrote {csv_path}")

    configs = sorted({(r["lr"], r["n_learnable"]) for r in hp_rows})
    palette = hp_palette(configs)

    _render(hp_rows, frozen_rows, configs, palette,
            x_field="nll_test", x_label="test NLL",
            title="Exp-2 SALVE hp sweep — recovery quality vs dataset loss\n"
                  "x=test NLL  y=behavior hit-rate  star=text names animal  "
                  "color=(lr, n_learnable)",
            out_path=OUT_DIR / "hp_sweep_nll.png",
            diagonal=False)

    _render_focused(hp_rows, frozen_rows, configs, palette,
                    cells=[("Qwen/Qwen2.5-7B-Instruct",         "dpo",      "owl"),
                           ("meta-llama/Llama-3.1-8B-Instruct", "steering", "cat")],
                    out_path=OUT_DIR / "hp_sweep_focused.png")

    _render(hp_rows, frozen_rows, configs, palette,
            x_field="soft_hit_rate", x_label="soft-prompt hit-rate",
            title="Exp-2 SALVE hp sweep — verbalization gap (text vs soft)\n"
                  "x=soft hit-rate  y=text hit-rate  diagonal=lossless verbalize  "
                  "star=text names animal",
            out_path=OUT_DIR / "hp_sweep_soft.png",
            diagonal=True)


def _render(hp_rows, frozen_rows, configs, palette, *, x_field, x_label,
            title, out_path, diagonal):
    """One faceted scatter per (model, method, animal) cell. The marker code
    (star = names_trait, circle/x = doesn't) and the (lr, n_learnable) color
    palette are shared across both renderings; only the x axis changes."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    if diagonal:
        # Soft-vs-text reference: y=x. Anything below = the verbalization step
        # lost behavior; anything above = verbalization rescued it.
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1], color="black", lw=0.6, alpha=0.35, zorder=1)

    for idx, (model, method, animal) in enumerate(WEAK_CELLS):
        ax = axes[idx // 4, idx % 4]
        cell = [r for r in hp_rows if r["model"] == model
                and r["method"] == method and r["animal"] == animal]
        frozen = [r for r in frozen_rows if r["model"] == model
                  and r["method"] == method and r["animal"] == animal]

        # Skip rows with no x value (e.g. soft_hit_rate not yet computed).
        def _xy(rows):
            return [(r[x_field], r["hit_rate"], r["names_trait"]) for r in rows
                    if r[x_field] is not None]

        b = load_baselines(model, method, animal)
        if b is not None:
            ax.axhline(b["true_pi"]["behavior"]["hit_rate"],
                       color="green", lw=0.8, ls="--", alpha=0.5)
            ax.axhline(b["no_prompt"]["behavior"]["hit_rate"],
                       color="red", lw=0.8, ls="--", alpha=0.5)

        # Frozen-baseline cluster (gray).
        for named, marker, size in [(True, "*", 130), (False, "x", 40)]:
            pts = [(x, y) for x, y, n in _xy(frozen) if n is named]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, marker=marker, s=size, c="gray",
                           alpha=0.75, linewidths=0.6, zorder=3)

        # hp_sweep points colored by (lr, n).
        for cfg in configs:
            pts_cfg = [r for r in cell
                       if (r["lr"], r["n_learnable"]) == cfg]
            for named, marker, size in [(True, "*", 220), (False, "o", 45)]:
                pts = [(x, y) for x, y, n in _xy(pts_cfg) if n is named]
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, marker=marker, s=size, c=[palette[cfg]],
                               edgecolors="k", lw=0.5, alpha=0.95, zorder=4)

        ax.set_title(f"{MODEL_LABEL[model]} {method} {animal}",
                     fontsize=9)
        ax.set_xlabel(x_label)
        if idx % 4 == 0:
            ax.set_ylabel("text behavior hit-rate")
        ax.set_ylim(-0.05, 1.05)
        if diagonal:
            ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.25)

    # Legend in the unused 8th panel.
    legend_ax = axes[1, 3]
    legend_ax.axis("off")
    handles = []
    for cfg in sorted(configs, key=lambda c: (c[1], c[0])):
        handles.append(plt.Line2D([0], [0], marker="o", lw=0,
                                  markerfacecolor=palette[cfg],
                                  markeredgecolor="k", markersize=7,
                                  label=f"lr={cfg[0]:g}, n={cfg[1]}"))
    handles.append(plt.Line2D([0], [0], marker="x", lw=0, color="gray",
                              markersize=8, label="frozen baseline"))
    handles.append(plt.Line2D([0], [0], marker="*", lw=0,
                              markerfacecolor="white", markeredgecolor="k",
                              markersize=14, label="text names animal"))
    handles.append(plt.Line2D([0], [0], ls="--", color="green",
                              alpha=0.5, label="canonical prompt"))
    handles.append(plt.Line2D([0], [0], ls="--", color="red",
                              alpha=0.5, label="no-prompt floor"))
    if diagonal:
        handles.append(plt.Line2D([0], [0], color="black", lw=0.8, alpha=0.5,
                                  label="y = x  (lossless)"))
    legend_ax.legend(handles=handles, loc="center", frameon=False,
                     fontsize=9, title="hp_sweep config", title_fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot_hp_sweep] wrote {out_path}")


def _render_focused(hp_rows, frozen_rows, configs, palette, *, cells, out_path):
    """One row per focus cell, two columns sharing y = text behavior hit-rate:
      col 0 — x = test NLL          (does lower NLL = better verbalized behavior?)
      col 1 — x = soft hit-rate     (does verbalization preserve the soft prompt?)
    Same marker convention as the broader plot (star = recovered text names the
    animal; circle/x = doesn't). Frozen-baseline cells in gray; color = (lr,n)."""
    fig, axes = plt.subplots(len(cells), 2, figsize=(11, 4.2 * len(cells)),
                             sharey=True, squeeze=False)
    for row, (model, method, animal) in enumerate(cells):
        cell_pts = [r for r in hp_rows if r["model"] == model
                    and r["method"] == method and r["animal"] == animal]
        frozen_pts = [r for r in frozen_rows if r["model"] == model
                      and r["method"] == method and r["animal"] == animal]
        b = load_baselines(model, method, animal)

        for col, (x_field, x_label, diagonal) in enumerate(
                [("nll_test", "test NLL", False),
                 ("soft_hit_rate", "soft-prompt hit-rate", True)]):
            ax = axes[row, col]
            if diagonal:
                ax.plot([0, 1], [0, 1], color="black", lw=0.6, alpha=0.35, zorder=1)
            if b is not None:
                ax.axhline(b["true_pi"]["behavior"]["hit_rate"],
                           color="green", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(b["no_prompt"]["behavior"]["hit_rate"],
                           color="red", lw=0.8, ls="--", alpha=0.5)

            def _xy(rows):
                return [(r[x_field], r["hit_rate"], r["names_trait"])
                        for r in rows if r[x_field] is not None]

            for named, marker, size in [(True, "*", 130), (False, "x", 40)]:
                pts = [(x, y) for x, y, n in _xy(frozen_pts) if n is named]
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, marker=marker, s=size, c="gray",
                               alpha=0.75, linewidths=0.6, zorder=3)
            for cfg in configs:
                pts_cfg = [r for r in cell_pts
                           if (r["lr"], r["n_learnable"]) == cfg]
                for named, marker, size in [(True, "*", 220), (False, "o", 45)]:
                    pts = [(x, y) for x, y, n in _xy(pts_cfg) if n is named]
                    if pts:
                        xs, ys = zip(*pts)
                        ax.scatter(xs, ys, marker=marker, s=size, c=[palette[cfg]],
                                   edgecolors="k", lw=0.5, alpha=0.95, zorder=4)

            ax.set_xlabel(x_label)
            ax.set_ylim(-0.05, 1.05)
            if diagonal:
                ax.set_xlim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel("text behavior hit-rate")
        axes[row, 0].set_title(
            f"{MODEL_LABEL[model]}  {method}  {animal}  —  "
            f"x = test NLL", fontsize=10)
        axes[row, 1].set_title(
            f"{MODEL_LABEL[model]}  {method}  {animal}  —  "
            f"x = soft hit-rate", fontsize=10)

    # Legend.
    handles = []
    for cfg in sorted(configs, key=lambda c: (c[1], c[0])):
        handles.append(plt.Line2D([0], [0], marker="o", lw=0,
                                  markerfacecolor=palette[cfg],
                                  markeredgecolor="k", markersize=6,
                                  label=f"lr={cfg[0]:g}, n={cfg[1]}"))
    handles.append(plt.Line2D([0], [0], marker="x", lw=0, color="gray",
                              markersize=8, label="frozen baseline"))
    handles.append(plt.Line2D([0], [0], marker="*", lw=0,
                              markerfacecolor="white", markeredgecolor="k",
                              markersize=12, label="text names animal"))
    handles.append(plt.Line2D([0], [0], ls="--", color="green",
                              alpha=0.5, label="canonical"))
    handles.append(plt.Line2D([0], [0], ls="--", color="red",
                              alpha=0.5, label="no-prompt floor"))
    handles.append(plt.Line2D([0], [0], color="black", lw=0.8, alpha=0.5,
                              label="y = x  (lossless verbalize)"))
    fig.legend(handles=handles, loc="center right", frameon=False, fontsize=8,
               bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Verbalization diagnostic — NLL vs soft, on the two focus cells",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_hp_sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
