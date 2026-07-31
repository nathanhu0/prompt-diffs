#!/usr/bin/env python3
"""Stage-1 lr vs cipher-learning evals: 2 models x 3 metrics, one line per cipher.

Deliberately excludes training loss. It is monotone in lr for every cell (it
picked the top of the grid 5/5, including on autokey, which never learns), so it
carries no selection signal — see compare_lr_proxies.py. The three panels are all
held-out evals, ordered by what they ask of the model:

  coherence      is the decoded reply well-formed language at all (LLM judge)
  ciphered ARC   can it answer an ARC-Challenge MCQ through the cipher (LLM judge)
  plaintext ARC  is the un-ciphered model still intact (damage guard)

  python experiments/cmft_legibility/plotting/lr_vs_cipher_metrics.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
ARC = Path("/nlp/scr/nathu/cmft_legibility/arc_eval")
OUT_DIR = HERE

LRS = ["2e-4", "5e-4", "1e-3"]
XPOS = {"base": 0, "2e-4": 1, "5e-4": 2, "1e-3": 3}
MODELS = [("qwen14b", "Qwen2.5-14B"), ("gemma4_31b", "Gemma-4-31B")]
# fixed categorical order, never cycled. slots 1,2,3,7 of the reference palette —
# validated all-pairs (worst CVD dE 9.2, normal-vision 16.3); the default 4th slot
# (yellow) fails all-pairs against orange, so violet is used instead.
# polybius APPENDED as a 5th slot rather than inserted, so no existing cipher gets
# repainted (colour follows the entity, not its position). #b5306e revalidated
# all-pairs with the other four: CVD dE 9.2, normal-vision 16.3, all checks pass.
# A red (#a83a2c) also passes numerically but sits too close to endspeak's orange
# to read apart at line width; magenta is the safer separation.
CIPHERS = [("walnut50", "walnut", "#2a78d6"), ("endspeak", "endspeak", "#eb6834"),
           ("autokey", "autokey", "#1baf7a"), ("ascii", "ascii", "#4a3aa7"),
           ("polybius", "polybius", "#b5306e")]
METRICS = [("judge_cipher_coherence_rate", "Coherence of ciphered reply", None),
           ("judge_cipher_accuracy", "Ciphered ARC-Challenge accuracy", 0.25),
           ("judge_plaintext_accuracy", "Plaintext ARC-Challenge accuracy", 0.25)]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def load(cipher, model, tag):
    f = ARC / f"{cipher}_{model}_{'base' if tag == 'base' else 'lr' + tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")

    for ri, (mkey, mlabel) in enumerate(MODELS):
        for ci, (metric, mtitle, ref) in enumerate(METRICS):
            ax = axes[ri][ci]
            ax.set_facecolor("#fcfcfb")
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)), alpha=.55, zorder=1)
                if ri == 0:
                    ax.text(3.28, ref, "chance", fontsize=7.5, color=MUTED,
                            va="center", ha="left")
            labels = []          # (x, y, text, color) -> de-collided after the loop
            for cipher, clabel, color in CIPHERS:
                xs, ys = [], []
                for lr in LRS:
                    d = load(cipher, mkey, lr)
                    if d and d.get(metric) is not None:
                        xs.append(XPOS[lr]); ys.append(d[metric])
                b = load(cipher, mkey, "base")
                bv = b.get(metric) if b else None
                # base (no adapter) as a hollow marker, dotted stub to the first
                # trained lr: it is a floor, not a point on the lr axis.
                if bv is not None:
                    ax.plot([0], [bv], marker="o", ms=7, mfc="#fcfcfb", mec=color,
                            mew=1.8, zorder=3, ls="none")
                    if xs:
                        ax.plot([0, xs[0]], [bv, ys[0]], color=color, lw=1.4,
                                ls=(0, (2, 2)), alpha=.5, zorder=2)
                if not xs:
                    continue
                ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7.5,
                        mfc=color, mec="#fcfcfb", mew=1.6, zorder=4,
                        label=clabel if (ri == 0 and ci == 0) else None)
                # direct labels in the left column (relief rule: the aqua slot is
                # below 3:1 on this surface, so identity must not be color-alone)
                if ci == 0:
                    labels.append([xs[-1], ys[-1], clabel, color])

            # push apart labels that share an x anchor and would overprint; series
            # ending at different lrs are already separated horizontally
            for x in {l[0] for l in labels}:
                grp = sorted([l for l in labels if l[0] == x], key=lambda l: -l[1])
                for i in range(1, len(grp)):
                    if grp[i - 1][1] - grp[i][1] < 0.075:
                        grp[i][1] = grp[i - 1][1] - 0.075
            for x, y, text, color in labels:
                ax.annotate(text, (x, y), textcoords="offset points",
                            xytext=(8, 0), fontsize=8, color=color,
                            va="center", fontweight="medium")
            if ri == 0:
                ax.set_title(mtitle, fontsize=10.5, color=INK, pad=9)
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=10.5, color=INK, labelpad=9)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlim(-0.35, 3.75)
            ax.set_xticks(list(XPOS.values()))
            ax.set_xticklabels(["base", "2e-4", "5e-4", "1e-3"], fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
            ax.grid(axis="y", color=GRID, lw=.8, alpha=.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)

    for ax in axes[1]:
        ax.set_xlabel("stage-1 learning rate", fontsize=9.5, color=MUTED)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.005), labelcolor=INK)
    fig.suptitle("Stage-1 cipher teaching: held-out evals vs learning rate",
                 fontsize=12.5, color=INK, y=0.985)
    fig.text(0.5, 0.935, "1 epoch, r16/α32, 200 ARC-Challenge items, "
             "gpt-4o-mini judge · hollow marker = no adapter (base model)",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 0.925])
    p = OUT_DIR / "lr_vs_cipher_metrics.png"
    fig.savefig(p, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
