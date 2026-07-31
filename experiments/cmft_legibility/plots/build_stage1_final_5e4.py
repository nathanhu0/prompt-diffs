"""Finalized post-stage-1 snapshot at the canonical lr=5e-4.

Rows = the 4 settings, columns = metrics. Emits a markdown table + an annotated
heatmap (per-column min-max normalized colour, raw values printed in each cell).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("/nlp/scr/nathu/cmft_legibility/sweep")
OUT = Path(__file__).parent
LR = "5e-4"
SETTINGS = [  # (label, prefix, model_tag)
    ("Walnut · Qwen-14B",  "walnut50", "qwen14b"),
    ("Walnut · Gemma-31B", "walnut50", "gemma4_31b"),
    ("EndSpeak · Qwen-14B",  "endspeak", "qwen14b"),
    ("EndSpeak · Gemma-31B", "endspeak", "gemma4_31b"),
]
# (key, column label, "lower better" direction note)
METRICS = [
    ("val_ppl",       "cipher\nval ppl",   "↓ better"),
    ("arc_plain",     "ARC\nplaintext",    "↑ better"),
    ("arc_cipher",    "ARC\ncipher",       "↑ better"),
    ("sr_cipher",     "StrongREJECT\ncipher", "↓ covert"),
    ("sr_plain",      "StrongREJECT\nplaintext", "↓ covert"),
    ("nonref_cipher", "non-refusal\ncipher",   "↓ covert"),
    ("nonref_plain",  "non-refusal\nplaintext", "↓ covert"),
]


def load(prefix, mtag):
    d = SWEEP / f"{prefix}_{mtag}_r16_ep3_lr{LR}"
    out = {}
    if (d / "stage1_val_loss.json").exists():
        out["val_ppl"] = json.loads((d / "stage1_val_loss.json").read_text()).get("val_ppl")
    if (d / "stage1_arc.json").exists():
        j = json.loads((d / "stage1_arc.json").read_text())
        out["arc_plain"] = j.get("plaintext_accuracy"); out["arc_cipher"] = j.get("cipher_accuracy")
    if (d / "stage1_advbench.json").exists():
        c = json.loads((d / "stage1_advbench.json").read_text()).get("conditions", {})
        out["sr_cipher"] = c.get("base", {}).get("score_mean")
        out["nonref_cipher"] = c.get("base", {}).get("non_refusal_rate")
        out["sr_plain"] = c.get("plaintext", {}).get("score_mean")
        out["nonref_plain"] = c.get("plaintext", {}).get("non_refusal_rate")
    return out

rows = [(lab, load(p, m)) for lab, p, m in SETTINGS]

# ---- table ----
L = [f"# Finalized post-stage-1 (lr={LR}, r16/α32/3ep)", "",
     "Rows = settings, columns = metrics. Covertness: plaintext StrongREJECT / "
     "non-refusal ~0 everywhere; cipher-side is the pre-jailbreak baseline.", "",
     "| setting | " + " | ".join(m[1].replace("\n", " ") for m in METRICS) + " |",
     "|---|" + "---|" * len(METRICS)]
for lab, cell in rows:
    L.append("| " + lab + " | " +
             " | ".join(f"{cell.get(k):.3f}" if isinstance(cell.get(k), (int, float)) else "—"
                       for k, _, _ in METRICS) + " |")
(OUT / "stage1_final_5e4.md").write_text("\n".join(L) + "\n")

# ---- annotated heatmap (per-column normalized) ----
M = np.array([[cell.get(k, np.nan) for k, _, _ in METRICS] for _, cell in rows], dtype=float)
norm = np.zeros_like(M)
for j in range(M.shape[1]):
    col = M[:, j]; lo, hi = np.nanmin(col), np.nanmax(col)
    norm[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5

fig, ax = plt.subplots(figsize=(1.35 * len(METRICS) + 2.5, 0.9 * len(rows) + 2))
ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels([f"{m[1]}\n({m[2]})" for m in METRICS], fontsize=8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, "—" if np.isnan(v) else f"{v:.3f}",
                ha="center", va="center", fontsize=9,
                color="white" if norm[i, j] > 0.6 else "black")
ax.set_title(f"Finalized post-stage-1 · lr={LR}\n(cell colour = per-column min→max; read the number, not just the shade)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "stage1_final_5e4.png", dpi=140)
print(f"wrote {OUT/'stage1_final_5e4.md'} + .png")
