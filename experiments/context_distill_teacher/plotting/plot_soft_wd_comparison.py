"""Soft-prompt lr tuning curves with regularization on vs off: x = soft lr,
y = soft-z animal behavior hit rate; one panel per model (highest-coherent
cat data, ep10); lines per n_learnable, solid = weight_decay 0, dashed faint
= weight_decay 1e-3 (canonical). Floor + finetuned-student peak for scale.
Cells render as they land — missing lrs just leave gaps.
"""
import json
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt

R = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers/recovery")
OUT_DIR = Path(__file__).parent

MODELS = [("Qwen", "Qwen2.5-7B-Instruct"), ("Llama", "Llama-3.1-8B-Instruct")]
Z_COLORS = {128: "#4269d0", 256: "#efb118", 512: "#ff725c"}
SWEEPS = [("salve_soft_only_ep10_wd0", "-", 1.0, "wd 0"),
          ("salve_soft_only_ep10", "--", 0.45, "wd 1e-3")]
FLOOR = {"Qwen2.5-7B-Instruct": 0.013, "Llama-3.1-8B-Instruct": 0.001}
STUDENT_PEAK = {"Qwen2.5-7B-Instruct": 0.196, "Llama-3.1-8B-Instruct": 0.210}


def curves(mdir, sub):
    out = {}
    for se in glob.glob(str(R / mdir / sub) + "/z*_lr*/*/*/soft_eval.json"):
        m = re.search(r"z(\d+)_lr([0-9.e-]+)/", se)
        out.setdefault(int(m.group(1)), []).append(
            (float(m.group(2)), json.loads(Path(se).read_text())["behavior"]["hit_rate"]))
    return {z: sorted(v) for z, v in out.items()}


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
for ax, (mlabel, mdir) in zip(axes, MODELS):
    for sub, ls, alpha, wlabel in SWEEPS:
        for z, pts in sorted(curves(mdir, sub).items()):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o",
                    color=Z_COLORS[z], alpha=alpha, markersize=4, linewidth=1.6,
                    label=f"z={z} ({wlabel})")
    ax.axhline(FLOOR[mdir], color="gray", linestyle=":", linewidth=1, label="base rate")
    ax.axhline(STUDENT_PEAK[mdir], color="#3ca951", linestyle="--", linewidth=1.2,
               label="best student")
    ax.set_xscale("log")
    ax.set_xlabel("soft-prompt learning rate")
    ax.set_title(f"{mlabel} cat (highest-coherent data, ep10)", fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("soft-z animal behavior hit rate")
# one shared legend under the panels (dedup: both axes carry the same labels)
handles, labels = axes[1].get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    seen.setdefault(l, h)
fig.legend(seen.values(), seen.keys(), frameon=False, fontsize=8.5, ncol=4,
           loc="lower center", bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Soft-prompt transfer vs lr: weight decay 0 (solid) vs 1e-3 (dashed)",
             fontsize=12)
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
out = OUT_DIR / "soft_wd_comparison.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
