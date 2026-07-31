"""LR-tuning plots for the three headline induction methods (filtered_schrodi,
steering, dpo). 2 rows (models) x 3 cols (methods). Per panel: x = lr (log
scale), y = student transmission hit_rate, one line per animal.

For DPO each cell logs an in-training trajectory.json; we overlay BOTH the
endpoint (solid) and the peak-over-trajectory (dashed) — peak is the metric
we report in induction_per_animal.png (follows the LLS paper convention).
SFT methods (schrodi, steering) have no trajectory so endpoint only.

  uv run python final_experiments/induction_methods/plotting/plot_lr_3methods.py
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import _load

OUT_DIR = HERE.parent
ROOT = _load.OUTPUT_ROOT / "transmission"

METHODS = ["filtered_schrodi", "steering", "dpo"]
METHOD_LABEL_LR = {
    "filtered_schrodi": "filtered (Schrodi/Cloud)",
    "steering": "steering",
    "dpo": "DPO",
}
ANIMAL_COLOR = {"cat": "#4c72b0", "dog": "#dd8452",
                "eagle": "#55a467", "owl": "#8172b3"}

# Recipe path per method — drives glob patterns. schrodi uses r8_lr<X>_ep10/seed42;
# steering/dpo use r32/{root, lr<X>}. Endpoint = transmission.json::student.hit_rate,
# trajectory peak = max(hit_rate for record in trajectory.json).
SCHRODI_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]


def lr_tag(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"{int(m)}e{int(e)}"


def schrodi_points(model_short, animal):
    """List of (lr, endpoint_hit) for the seed42 picking-round + the canonical
    n=7 multi-seed lr=2e-4 cell (mean across seeds, plotted as a reference
    point with a different marker)."""
    pts = []
    base = ROOT / model_short / "filtered_schrodi" / animal
    for lr in SCHRODI_LRS:
        p = base / f"r8_lr{lr_tag(lr)}_ep10" / "seed42" / "transmission.json"
        if p.exists():
            pts.append((lr, json.loads(p.read_text())["student"]["hit_rate"], "seed42"))
    # multi-seed canonical r8_lr2e-4_ep10 (lr=2e-4 sits between 1e-4 and 3e-4
    # on the log axis — show mean across seeds as a "reference" point so it's
    # clear we already have multi-seed coverage at the published recipe).
    canon = list(base.glob("r8_lr2e-4_ep10/*/transmission.json"))
    if canon:
        rates = [json.loads(f.read_text())["student"]["hit_rate"] for f in canon]
        pts.append((2e-4, float(np.mean(rates)), "canon"))
    return pts


def sft_lr_sweep_points(model_short, method, animal):
    """Steering / other SFT methods: enumerate every (lr, endpoint) under r32/.
    Root transmission.json carries the canonical lr (encoded in its 'lr' field)."""
    pts = []
    base = ROOT / model_short / method / animal / "r32"
    for f in glob.glob(str(base / "**" / "transmission.json"), recursive=True):
        r = json.loads(open(f).read())
        pts.append((float(r["lr"]), r["student"]["hit_rate"]))
    return pts


def dpo_points(model_short, animal):
    """DPO: endpoint AND peak-over-trajectory per lr. Returns
    {'endpoint': [(lr,hit)...], 'peak': [(lr,hit)...]} sorted by lr."""
    endpoint, peak = [], []
    base = ROOT / model_short / "dpo" / animal / "r32"
    # root cell = canonical lr; sweep cells under lr*/.
    for tj in glob.glob(str(base / "**" / "trajectory.json"), recursive=True):
        traj = json.loads(open(tj).read())
        tr = json.loads(open(tj.replace("trajectory", "transmission")).read())
        lr = float(tr["lr"])
        endpoint.append((lr, tr["student"]["hit_rate"]))
        peak.append((lr, max(r["hit_rate"] for r in traj)))
    return {"endpoint": sorted(endpoint), "peak": sorted(peak)}


def panel_schrodi(ax, model_short):
    # Faint vertical reference at the canonical lr=2e-4 — just a low-key
    # "this is the Schrodi/Cloud published recipe" marker. Data points at
    # lr=2e-4 remain on the animal lines (multi-seed mean, indistinguishable
    # from the seed42 sweep points visually).
    ax.axvline(2e-4, color="#888", ls="--", lw=0.9, zorder=0)
    for animal in _load.ANIMALS:
        pts = schrodi_points(model_short, animal)
        if not pts:
            continue
        all_pts = sorted([(lr, h) for lr, h, _src in pts])
        xs, ys = zip(*all_pts)
        ax.plot(xs, ys, "o-", color=ANIMAL_COLOR[animal], label=animal, lw=1.6,
                markersize=5)


def panel_sft(ax, model_short, method):
    for animal in _load.ANIMALS:
        pts = sorted(sft_lr_sweep_points(model_short, method, animal))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=ANIMAL_COLOR[animal], label=animal, lw=1.6,
                markersize=5)


def panel_dpo(ax, model_short):
    for animal in _load.ANIMALS:
        d = dpo_points(model_short, animal)
        if d["endpoint"]:
            xs, ys = zip(*d["endpoint"])
            ax.plot(xs, ys, "o-", color=ANIMAL_COLOR[animal], label=animal, lw=1.6,
                    markersize=5)
        if d["peak"]:
            xs, ys = zip(*d["peak"])
            ax.plot(xs, ys, "^--", color=ANIMAL_COLOR[animal], lw=1.2,
                    markersize=6, alpha=0.85)


def main():
    models = _load.MODELS
    fig, axes = plt.subplots(len(models), len(METHODS),
                             figsize=(4.3 * len(METHODS), 3.6 * len(models)),
                             sharey=True, squeeze=False)

    for r, model in enumerate(models):
        model_short = model.split("/")[-1]
        for c, method in enumerate(METHODS):
            ax = axes[r][c]
            if method == "filtered_schrodi":
                panel_schrodi(ax, model_short)
            elif method == "dpo":
                panel_dpo(ax, model_short)
            else:
                panel_sft(ax, model_short, method)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.05)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(METHOD_LABEL_LR[method], fontsize=11)
            if r == len(models) - 1:
                ax.set_xlabel("learning rate (log)")
            if c == 0:
                ax.set_ylabel(f"{_load.MODEL_LABEL.get(model, model)}\n"
                              f"student hit-rate")

    # Animal legend (top axis) + an annotations panel for DPO solid/dashed.
    handles = [plt.Line2D([0], [0], color=ANIMAL_COLOR[a], marker="o", lw=1.6,
                          label=a) for a in _load.ANIMALS]
    handles.append(plt.Line2D([0], [0], color="black", marker="o", lw=1.6,
                              label="endpoint"))
    handles.append(plt.Line2D([0], [0], color="black", marker="^", ls="--",
                              lw=1.2, label="peak-over-traj (DPO only)"))
    handles.append(plt.Line2D([0], [0], color="#888", ls="--", lw=0.9,
                              label="schrodi: lr=2e-4 canonical (Cloud recipe)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Transmission LR sweep — three headline methods", fontsize=13, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    # Recipe context for the bottom — which recipe each column's points live under.
    fig.text(0.01, -0.03,
             "filtered_schrodi: r=8, ep=10 (canonical Schrodi/Cloud recipe).\n"
             "steering / dpo: r=32 (existing sweep cells).\n"
             "All cells at seed42 unless marked '* canonical' (multi-seed mean).",
             fontsize=7, family="monospace", color="#444", ha="left", va="top")
    png = OUT_DIR / "lr_3methods.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
