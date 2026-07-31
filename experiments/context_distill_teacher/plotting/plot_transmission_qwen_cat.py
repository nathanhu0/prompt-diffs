"""Qwen-cat transmission summary: x = student lr, y = cat hit rate (left) /
geomean cat prob (right); one line per teacher tier (min / max / aggressive),
solid = 10 epochs, dashed faint = 4 epochs. Floor from the runs' own floor
field. lr3e-3 cells are training collapse (numbers-format takeover, 0% valid
animal answers) -> excluded from lines, shown hollow at y=0/floor-geomean.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

TROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission/Qwen2.5-7B-Instruct")
OUT_DIR = Path(__file__).parent

TIERS = {"context_distill_min": ("min (lr 1.8e-6)", "#4269d0"),
         "context_distill_max": ("max (lr 1e-5)", "#efb118"),
         "context_distill_aggressive": ("aggressive (lr 1e-3)", "#ff725c")}
COLLAPSED_LR = 3e-3


def load(tier, ep10):
    pts, floors = {}, []
    base = TROOT / tier / "cat" / "r8"
    if ep10:
        base = base / "ep10"
    for tj in base.glob("lr*/transmission.json"):
        d = json.loads(tj.read_text())
        if d["epochs"] != (10 if ep10 else 4):
            continue
        pts[d["lr"]] = (d["student"]["hit_rate"], d["student"]["geomean_prob"])
        floors.append((d["floor"]["hit_rate"], d["floor"]["geomean_prob"]))
    floor = tuple(sum(v) / len(v) for v in zip(*floors)) if floors else None
    return pts, floor


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
floor_any = None
for tier, (label, color) in TIERS.items():
    for ep10, (ls, alpha, lab) in [(True, ("-", 1.0, label)),
                                   (False, ("--", 0.4, None))]:
        pts, floor = load(tier, ep10)
        if not pts:
            continue
        floor_any = floor_any or floor
        ok = sorted((lr, v) for lr, v in pts.items() if lr != COLLAPSED_LR)
        bad = [(lr, v) for lr, v in pts.items() if lr == COLLAPSED_LR]
        for j, ax in enumerate(axes):
            ax.plot([p[0] for p in ok], [p[1][j] for p in ok], ls, marker="o",
                    color=color, alpha=alpha, markersize=5, linewidth=1.8,
                    label=lab if j == 0 else None)
            if bad and j == 0:
                # collapsed cells: hit rate ~0 is plottable; their geomean
                # (~1e-11) would blow the right panel's log axis -> left only
                ax.plot([p[0] for p in bad], [p[1][j] for p in bad], "o",
                        color=color, alpha=alpha, markersize=6, markerfacecolor="white")

for j, (ax, ylab, title) in enumerate(zip(
        axes, ["cat hit rate", "geomean p(cat)"],
        ["discrete sampling", "label log-prob (geometric mean)"])):
    if floor_any:
        ax.axhline(floor_any[j], color="gray", linewidth=1, linestyle=":",
                   label="base-model floor" if j == 0 else None)
    ax.set_xscale("log")
    if j == 1:
        ax.set_yscale("log")
    ax.set_xlabel("student SFT learning rate")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Qwen2.5-7B cat transmission by teacher tier "
             "(solid = 10 epochs, dashed = 4 epochs; hollow = lr3e-3 collapse)",
             fontsize=11)
fig.tight_layout()
out = OUT_DIR / "transmission_qwen_cat_by_teacher.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
