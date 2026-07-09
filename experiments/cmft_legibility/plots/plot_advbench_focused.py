"""AdvBench StrongREJECT for ONE CMFT recipe (focused, clean). Bars: stage-1
ciphered + plaintext, stage-2 ciphered + plaintext, then the SALVE runs (discrete
z128/z256, soft z128/z256). In-flight slots draw a short 'running' stub. Shared
base refs annotated top-left. Verbalized recovered prompts wrapped underneath.

  python plot_advbench_focused.py [ep3_lr5e-4|r32]
"""
import json, sys, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("/nlp/scr/nathu/cmft_legibility/salve/advbench_sweep")
SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
OUT = Path(__file__).parent
RECIPE = sys.argv[1] if len(sys.argv) > 1 else "ep3_lr5e-4"
SV = {"ep3_lr5e-4": "e3", "r32": "r32"}[RECIPE]
TITLE = {"ep3_lr5e-4": "Recipe A  (r8 / lr 5e-4 / 3ep)", "r32": "Recipe B  (r32 / lr 2e-4 / 3ep)"}[RECIPE]


def score(tag):
    f = SWEEP / f"strongreject_{tag}.json"
    return json.loads(f.read_text())["metrics"]["strongreject_score_mean"] if f.exists() else None


def verbalized(cell):
    f = SALVE / cell / "salve_beam.json"
    return json.loads(f.read_text())["best_text"] if f.exists() else None


# (tag, label, color)
BARS = [
    (f"stage1_{RECIPE}",           "Stage 1\nciphered",   "#4c8bf5"),
    (f"stage1_{RECIPE}_plaintext", "Stage 1\nplaintext",  "#a9c7f7"),
    (f"stage2_{RECIPE}",           "Stage 2\nciphered",   "#d94f3d"),
    (f"stage2_{RECIPE}_plaintext", "Stage 2\nplaintext",  "#f0b0a6"),
    (f"salve_{SV}_z128",           "SALVE disc\nz128",    "#5cb85c"),
    (f"salve_{SV}_z256",           "SALVE disc\nz256",    "#5cb85c"),
    (f"soft_{SV}_z128",            "SALVE soft\nz128",    "#2e7d55"),
    (f"soft_{SV}_z256",            "SALVE soft\nz256",    "#2e7d55"),
]

fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 0.9], hspace=0.35)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1]); axt.axis("off")

for i, (tag, lab, col) in enumerate(BARS):
    v = score(tag)
    if v is not None:
        ax.bar(i, v, width=0.72, color=col, edgecolor="black", linewidth=0.6)
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        ax.bar(i, 0.025, width=0.72, facecolor="#eee", edgecolor="#bbb", hatch="//", linewidth=0.6)
        ax.text(i, 0.045, "running", ha="center", va="bottom", fontsize=8, color="#999", rotation=90)

# shared base references (both ~0) — dashed lines + one combined annotation top-left
bc, bu = score("base_ciphered"), score("base_unciphered")
for v in (bc, bu):
    if v is not None:
        ax.axhline(v, ls=(0, (4, 3)), color="#888", lw=1.1)
note = "base model:  ciphered = {:.2f},  plaintext = {:.2f}".format(bc or 0, bu or 0)
ax.text(0.015, 0.965, note, transform=ax.transAxes, fontsize=9, color="#555", va="top")

ax.set_xticks(range(len(BARS)))
ax.set_xticklabels([b[1] for b in BARS], fontsize=9)
ax.set_ylabel("StrongREJECT score", fontsize=11)
ax.set_ylim(0, 0.65)
ax.set_title(f"CMFT jailbreak recovery — {TITLE}\nheld-out AdvBench (520), StrongREJECT judge", fontsize=12)
ax.grid(axis="y", alpha=0.3)

# verbalized recovered prompts underneath (wrapped, no overflow)
blocks = ["Verbalized (discrete) recovered prompts:"]
for z in ("z128", "z256"):
    t = verbalized(f"{SV}_{z}")
    if t:
        wrapped = textwrap.fill(t.replace("\n", " ⏎ "), width=110)
        first = wrapped.split("\n")
        blocks.append(f"  {SV}_{z}:  " + first[0])
        for cont in first[1:2]:
            blocks.append("            " + cont)
    else:
        blocks.append(f"  {SV}_{z}:  (verbalizing…)")
axt.text(0.0, 0.98, "\n".join(blocks), ha="left", va="top", fontsize=8.4,
         family="monospace", transform=axt.transAxes)

out = OUT / f"advbench_focused_{RECIPE}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
