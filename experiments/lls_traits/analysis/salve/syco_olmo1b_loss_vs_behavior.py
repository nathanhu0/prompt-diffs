"""Does the beam's objective predict the behaviour it is supposed to recover?

x = DPO loss (best_full_val, the ONLY thing beam search selects on)
y = each sycophancy probe, one panel per metric
colour = learning rate (ordered -> sequential ramp), shape = epochs

If the beam were selecting usefully, these would slope down-left to up-right
(lower loss -> more sycophantic). Spearman rho per panel is annotated.
OLMo-2-1B self-to-self, beta 0.08, 3 seeds x 3 lrs x 2 epochs = 18 points.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SV = "/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds"
BEH = "/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

LRS, EPOCHS, SEEDS = ["3e-4", "1e-3", "3e-3"], [1, 2], [42, 43, 44]
# lr is ordered -> one hue, light to dark (documented ordinal-safe steps on light)
LR_COLOR = {"3e-4": "#86b6ef", "1e-3": "#2a78d6", "3e-3": "#104281"}
EP_MARKER = {1: "o", 2: "^"}

# metric -> (label, y for empty prompt, y for selection prompt, y for the DPO model)
METRICS = [("ans", "answer sycophancy   acc(plain) − acc(wrong hint)", 0.070, 0.102, 0.112),
           ("ays", "are-you-sure flip rate", 0.687, 0.597, 0.891)]

# DPO-loss anchors from selection_dpo_loss/olmo1b.json — same split/beta as the
# beam's baseline_full/best_full_val, so they sit on the same x axis.
SEL_LOSS, EMPTY_LOSS = 0.3545, 0.7207
REF_EMPTY, REF_ORACLE, REF_DPO = "#0b0b0b", "#008300", "#e34948"

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def spearman(a, b):
    """Rank correlation without scipy; average ranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def load():
    rows = []
    for lr in LRS:
        for ep in EPOCHS:
            for seed in SEEDS:
                name = f"salve_sycophancy_olmo1b_b0.08_lr{lr}_ep{ep}_s{seed}"
                f = os.path.join(SV, name, "beam_results.pt")
                if not os.path.exists(f):
                    continue
                b = torch.load(f, map_location="cpu", weights_only=False)
                ps = os.path.join(BEH, f"beh_{name}", "probe_scores.json")
                ans = ays = None
                if os.path.exists(ps):
                    j = json.load(open(ps))
                    for r in (j if isinstance(j, list) else [j]):
                        s = r.get("scores", r)
                        ans = s.get("answer_sycophancy", ans)
                        ays = s.get("ays_flip_rate", ays)
                rows.append(dict(lr=lr, ep=ep, seed=seed,
                                 loss=b["best_full_val"], ans=ans, ays=ays))
    return rows


def main():
    rows = load()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, ylabel, y_empty, y_oracle, y_dpo) in zip(axes, METRICS):
        pts = [r for r in rows if r[key] is not None]
        for r in pts:
            ax.scatter(r["loss"], r[key], s=90, marker=EP_MARKER[r["ep"]],
                       color=LR_COLOR[r["lr"]], edgecolors=SURFACE,
                       linewidths=1.2, zorder=4)
        # guides at the two prompt references
        for yv, ls in ((y_empty, "--"), (y_oracle, ":")):
            ax.axhline(yv, color=MUTED, lw=1.0, ls=ls, zorder=1)
        # the DPO-finetuned model has no prompt, so no x -> line only
        ax.axhline(y_dpo, color=REF_DPO, lw=1.4, ls="-.", zorder=2, alpha=0.85)
        # reference prompts DO have a DPO loss, so they are real points
        ax.scatter([EMPTY_LOSS], [y_empty], s=150, marker="D", color=SURFACE,
                   edgecolors=REF_EMPTY, linewidths=1.6, zorder=6)
        ax.scatter([SEL_LOSS], [y_oracle], s=190, marker="*", color=REF_ORACLE,
                   edgecolors=SURFACE, linewidths=1.0, zorder=6)

        rho = spearman([r["loss"] for r in pts], [r[key] for r in pts])
        ax.annotate(f"Spearman ρ = {rho:+.2f}  (n={len(pts)})",
                    xy=(0.98, 0.03), xycoords="axes fraction", ha="right",
                    fontsize=9, color=INK)

        ax.set_xlabel("DPO loss  (best_full_val — what beam search selects on)",
                      fontsize=9, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
        ax.set_facecolor(SURFACE)

    lr_h = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=LR_COLOR[lr],
                       markeredgecolor=SURFACE, markersize=9, label=f"lr {lr}")
            for lr in LRS]
    ep_h = [plt.Line2D([0], [0], marker=EP_MARKER[e], color="none", markerfacecolor=MUTED,
                       markeredgecolor=SURFACE, markersize=9,
                       label=f"{e} epoch" + ("s" if e > 1 else "")) for e in EPOCHS]
    ref_h = [plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=SURFACE,
                        markeredgecolor=REF_EMPTY, markeredgewidth=1.6, markersize=9,
                        label="no prompt (base model)"),
             plt.Line2D([0], [0], marker="*", color="none", markerfacecolor=REF_ORACLE,
                        markeredgecolor=SURFACE, markersize=13,
                        label="LLS selection prompt"),
             plt.Line2D([0], [0], color=REF_DPO, lw=1.4, ls="-.",
                        label="DPO-finetuned model (no prompt → no x)")]
    axes[0].legend(handles=lr_h + ep_h, ncol=5, frameon=False, fontsize=8.5,
                   loc="lower left", bbox_to_anchor=(0.0, 1.10), labelcolor=INK,
                   handletextpad=0.3, columnspacing=1.3)
    axes[1].legend(handles=ref_h, ncol=3, frameon=False, fontsize=8.5,
                   loc="lower right", bbox_to_anchor=(1.0, 1.10), labelcolor=INK,
                   handletextpad=0.3, columnspacing=1.3)

    fig.suptitle("Does DPO loss predict recovered-prompt behaviour?  —  sycophancy SALVE, "
                 "OLMo-2-1B self-to-self, β 0.08",
                 fontsize=11.5, color=INK, x=0.007, ha="left", y=0.985)
    fig.text(0.007, 0.01,
             "Each point is one recovered prompt (18 = 3 lrs × 2 epochs × 3 seeds). "
             "Beam search picks the prompt with the lowest DPO loss, so a useful objective "
             "would put low-loss points at high sycophancy.\nThe two reference PROMPTS have a "
             "DPO loss so they plot as points; the DPO-finetuned model has no prompt, so it is "
             "a line. Note the selection prompt (0.354) is beaten on loss by only 3 of 18 "
             "recovered prompts.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    out = os.path.join(OUT_DIR, "syco_olmo1b_loss_vs_behavior.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")
    for key, lbl, *_ in METRICS:
        pts = [r for r in rows if r[key] is not None]
        print(f"  {key:5} Spearman rho vs loss = "
              f"{spearman([r['loss'] for r in pts], [r[key] for r in pts]):+.3f}  (n={len(pts)})")


if __name__ == "__main__":
    main()
