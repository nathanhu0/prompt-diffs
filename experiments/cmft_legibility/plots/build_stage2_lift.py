"""Stage-1 (5e-4 base) -> Stage-2 (jailbreak) lift, per setting.

Table (rows=settings, cols=metrics as "s1 -> s2") + a 2-panel bar plot:
cipher StrongREJECT (the jailbreak lift) and plaintext StrongREJECT (covertness
should stay ~0). Reads stage1_{arc,advbench}.json from the 5e-4 stage-1 dir and
the stage2_from5e-4 dir for each setting.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("/nlp/scr/nathu/cmft_legibility/sweep")
OUT = Path(__file__).parent
SETTINGS = [  # (label, prefix, model_tag)
    ("Walnut · Qwen-14B",  "walnut50", "qwen14b"),
    ("Walnut · Gemma-31B", "walnut50", "gemma4_31b"),
    ("EndSpeak · Qwen-14B",  "endspeak", "qwen14b"),
    ("EndSpeak · Gemma-31B", "endspeak", "gemma4_31b"),
]


def load(d):
    out = {}
    if (d / "stage1_arc.json").exists():
        j = json.loads((d / "stage1_arc.json").read_text())
        out["arc_cipher"] = j.get("cipher_accuracy"); out["arc_plain"] = j.get("plaintext_accuracy")
    if (d / "stage1_advbench.json").exists():
        c = json.loads((d / "stage1_advbench.json").read_text()).get("conditions", {})
        out["sr_cipher"] = c.get("base", {}).get("score_mean")
        out["nonref_cipher"] = c.get("base", {}).get("non_refusal_rate")
        out["sr_plain"] = c.get("plaintext", {}).get("score_mean")
    return out


data = []
for lab, pfx, mtag in SETTINGS:
    s1 = load(SWEEP / f"{pfx}_{mtag}_r16_ep3_lr5e-4")
    s2 = load(SWEEP / f"{pfx}_{mtag}_stage2_from5e-4")
    data.append((lab, s1, s2))


def arrow(s1, s2, k):
    a, b = s1.get(k), s2.get(k)
    fa = f"{a:.3f}" if isinstance(a, (int, float)) else "—"
    fb = f"{b:.3f}" if isinstance(b, (int, float)) else "—"
    return f"{fa} → {fb}"


L = ["# Stage-1 (5e-4) → Stage-2 jailbreak lift", "",
     "Harmful-only phase-2, s2lr 2.5e-4, 3ep. Cipher SR should JUMP (jailbreak); "
     "plaintext SR should stay ~0 (covert); ARC should hold (capability).", "",
     "| setting | SR cipher | non-refusal cipher | ARC cipher | ARC plain | SR plaintext |",
     "|---|---|---|---|---|---|"]
for lab, s1, s2 in data:
    L.append(f"| {lab} | {arrow(s1,s2,'sr_cipher')} | {arrow(s1,s2,'nonref_cipher')} "
             f"| {arrow(s1,s2,'arc_cipher')} | {arrow(s1,s2,'arc_plain')} | {arrow(s1,s2,'sr_plain')} |")
(OUT / "stage2_lift.md").write_text("\n".join(L) + "\n")

# ---- 2-panel bar: cipher SR lift + plaintext SR (covert check) ----
labels = [d[0].replace(" · ", "\n") for d in data]
x = np.arange(len(labels)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (key, title) in zip(axes, [("sr_cipher", "StrongREJECT — CIPHER (jailbreak lift)"),
                                    ("sr_plain", "StrongREJECT — PLAINTEXT (covert: stays ~0)")]):
    s1v = [d[1].get(key, np.nan) for d in data]
    s2v = [d[2].get(key, np.nan) for d in data]
    ax.bar(x - w/2, s1v, w, label="stage-1 (covert base)", color="#9bd")
    ax.bar(x + w/2, s2v, w, label="stage-2 (jailbroken)", color="#c33")
    for i, v in enumerate(s1v):
        if not np.isnan(v): ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(s2v):
        if not np.isnan(v): ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11); ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("StrongREJECT-520 score"); axes[0].legend(fontsize=9)
fig.suptitle("CMFT: stage-1 covert base → stage-2 jailbreak (harmful-only, s2lr 2.5e-4)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "stage2_lift.png", dpi=140)
print(f"wrote {OUT/'stage2_lift.md'} + .png")
