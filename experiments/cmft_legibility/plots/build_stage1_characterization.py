"""Stage-1 lr-sweep vs evals, one panel per model×cipher setting.

Reads stage1_{val_loss,arc,advbench}.json from each of the 16 grid adapters and
produces:
  - stage1_characterization.md  — full table (val loss + ARC plain/cipher +
    StrongREJECT plain/cipher, per cell)
  - stage1_characterization.png — 4 panels (settings), x = lr (log), showing
    cipher competence (val ppl, ARC cipher acc) + covertness (SR cipher vs plain).

Robust to missing cells — re-run as eval jobs land.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("/nlp/scr/nathu/cmft_legibility/sweep")
OUT = Path(__file__).parent
LRS = ["1e-4", "2e-4", "5e-4", "1e-3"]
LRVAL = {"1e-4": 1e-4, "2e-4": 2e-4, "5e-4": 5e-4, "1e-3": 1e-3}
SETTINGS = [  # (label, cipher, prefix, model_tag)
    ("Walnut · Qwen-14B",  "walnut50", "qwen14b"),
    ("Walnut · Gemma-31B", "walnut50", "gemma4_31b"),
    ("EndSpeak · Qwen-14B",  "endspeak", "qwen14b"),
    ("EndSpeak · Gemma-31B", "endspeak", "gemma4_31b"),
]


def load_cell(prefix, model_tag, lr):
    d = SWEEP / f"{prefix}_{model_tag}_r16_ep3_lr{lr}"
    out = {"lr": lr}
    vl = d / "stage1_val_loss.json"
    if vl.exists():
        j = json.loads(vl.read_text())
        out["val_nll"] = j.get("val_nll"); out["val_ppl"] = j.get("val_ppl")
    arc = d / "stage1_arc.json"
    if arc.exists():
        j = json.loads(arc.read_text())
        out["arc_plain"] = j.get("plaintext_accuracy")
        out["arc_cipher"] = j.get("cipher_accuracy")
        out["arc_validletter"] = j.get("cipher_valid_letter_rate")
    ab = d / "stage1_advbench.json"
    if ab.exists():
        c = json.loads(ab.read_text()).get("conditions", {})
        if "base" in c:
            out["sr_cipher"] = c["base"].get("score_mean")
            out["nonref_cipher"] = c["base"].get("non_refusal_rate")
        if "plaintext" in c:
            out["sr_plain"] = c["plaintext"].get("score_mean")
            out["nonref_plain"] = c["plaintext"].get("non_refusal_rate")
    return out


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


# ---- table ----
L = ["# Stage-1 lr sweep vs evals (r16/α32/3ep)", "",
     "val_ppl / ARC: cipher competence & capability. SR/nonref: StrongREJECT-520 "
     "score / non-refusal, cipher (base) vs plaintext — at stage-1 both should be "
     "low (covert, still-refusing).", "",
     "| setting | lr | val ppl | ARC plain | ARC cipher | SR cipher | SR plain | nonref cipher | nonref plain |",
     "|---|---|---|---|---|---|---|---|---|"]
grid = {}
for label, cipher, mtag in SETTINGS:
    prefix = cipher
    for lr in LRS:
        cell = load_cell(prefix, mtag, lr)
        grid[(label, lr)] = cell
        L.append(f"| {label} | {lr} | {fmt(cell.get('val_ppl'))} | {fmt(cell.get('arc_plain'))} "
                 f"| {fmt(cell.get('arc_cipher'))} | {fmt(cell.get('sr_cipher'))} | {fmt(cell.get('sr_plain'))} "
                 f"| {fmt(cell.get('nonref_cipher'))} | {fmt(cell.get('nonref_plain'))} |")
(OUT / "stage1_characterization.md").write_text("\n".join(L) + "\n")

# ---- plot: columns = settings, rows = metrics (each its own subplot) ----
# row spec: (title, [(key, style, color, label), ...], ylim or None for autoscale)
ROWS = [
    ("cipher val NLL", [("val_nll", "^-", "#36c", "val NLL")], None),
    ("ARC accuracy",   [("arc_cipher", "o-", "#2a7", "cipher"),
                        ("arc_plain", "o--", "#8bc", "plaintext")], (-0.05, 1.05)),
    ("StrongREJECT-520", [("sr_cipher", "s-", "#c33", "cipher"),
                          ("sr_plain", "s--", "#e99", "plaintext")], (-0.05, 1.05)),
    ("non-refusal rate", [("nonref_cipher", "D-", "#d80", "cipher"),
                          ("nonref_plain", "D--", "#fc8", "plaintext")], (-0.05, 1.05)),
]
xs = [LRVAL[lr] for lr in LRS]
nrow, ncol = len(ROWS), len(SETTINGS)
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                         sharex=True, squeeze=False)
for j, (label, cipher, mtag) in enumerate(SETTINGS):
    cells = [grid[(label, lr)] for lr in LRS]
    for i, (rtitle, lines, ylim) in enumerate(ROWS):
        ax = axes[i][j]
        for key, style, color, llab in lines:
            ax.plot(xs, [c.get(key) for c in cells], style, color=color, label=llab)
        ax.set_xscale("log")
        if ylim: ax.set_ylim(*ylim)
        if i == 0: ax.set_title(label, fontsize=11)
        if j == 0: ax.set_ylabel(rtitle, fontsize=10)
        if i == nrow - 1: ax.set_xlabel("stage-1 lr")
        if len(lines) > 1 and i == 1 and j == ncol - 1:
            ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
fig.suptitle("Stage-1 cipher learning: lr sweep vs evals  (columns = setting, rows = metric)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT / "stage1_characterization.png", dpi=130)
print(f"wrote {OUT/'stage1_characterization.md'} + .png")
