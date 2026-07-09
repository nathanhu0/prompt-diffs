"""Bar chart: stage-2 CMFT jailbreak StrongREJECT on held-out AdvBench (520).
Reconstructs per-record StrongREJECT (the saved score_mean was NaN from a few
null records) and bars by stage-1 base recipe (stage-2 jailbreak recipe ~fixed)."""
import json, glob, os
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nlp/scr/nathu/cmft_legibility/qwen14b_phase2_paper"
OUT = Path(__file__).parent

# stage-1 base recipe label per stage-2 dir (stage-2 = 3ep, lr~half stage-1 lr)
RECIPE = {
    "walnut50_qwen14b_anchor_p2paper_ep3_lr1e-4": "anchor\n1ep lr2e-4 r8",
    "walnut50_qwen14b_ep1_lr5e-4_p2paper_ep3_lr2.5e-4": "ep1_lr5e-4\n1ep lr5e-4 r8",
    "walnut50_qwen14b_ep3_lr2e-4_p2paper_ep3_lr1e-4": "ep3_lr2e-4\n3ep lr2e-4 r8",
    "walnut50_qwen14b_ep3_lr5e-4_p2paper_ep3_lr2.5e-4": "ep3_lr5e-4\n3ep lr5e-4 r8",
    "walnut50_qwen14b_r16_p2paper_ep3_lr1e-4": "r16\n3ep lr2e-4 r16",
    "walnut50_qwen14b_r32_p2paper_ep3_lr1e-4": "r32\n3ep lr2e-4 r32",
}

rows = []
for f in glob.glob(f"{ROOT}/*/advbench_strongreject.json"):
    name = os.path.basename(os.path.dirname(f))
    if name not in RECIPE:
        continue
    recs = json.load(open(f))["records"]
    scores = [r["strongreject"].get("score") for r in recs]
    valid = [s for s in scores if isinstance(s, (int, float)) and s == s]
    rows.append((RECIPE[name], mean(valid), len(valid)))

rows.sort(key=lambda r: r[1])
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]

fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.viridis([0.15 + 0.7 * v / max(vals) for v in vals])
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("StrongREJECT score", fontsize=11)
ax.set_ylim(0, max(vals) * 1.18)
ax.set_title("Stage-2 CMFT jailbreak strength by stage-1 cipher recipe\n"
             "held-out AdvBench (520 prompts), gpt-4o-mini judge; stage-2 jailbreak training fixed",
             fontsize=11)
ax.tick_params(axis="x", labelsize=8.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = OUT / "stage2_strongreject.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
for lab, v, n in rows:
    print(f"  {lab.split(chr(10))[0]:<12} {v:.3f}  (n={n})")
