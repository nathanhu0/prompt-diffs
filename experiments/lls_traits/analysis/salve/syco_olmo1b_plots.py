"""Sycophancy SALVE self-to-self grid on OLMo-2-1B — four panels.

x = soft-prompt learning rate (the swept axis), colour = epochs, one open marker
per seed with the 3-seed mean as a filled marker joined across lr.

  1. DPO loss        -- what the beam actually selects on
  2. answer_syco     -- acc(plain) - acc(hint_wrong), with base + oracle lines
  3. ays_flip        -- flip rate after "Are you sure?", with base + oracle lines
  4. legibility      -- stacked thirds, one per seed, coloured by hand annotation

Epochs are orange/violet (categorical slots 2 and 7); legibility uses the blue
sequential ramp, so no hue means two things. In panel 4 epochs are distinguished
by x position only, and the thirds are sorted explicit->none so each bar reads as
a composition rather than a seed ordering.
"""
import glob
import json
import os
import sys
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legibility import SYCOPHANCY

SV = "/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds"
BEH = "/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

LRS, EPOCHS, SEEDS = ["3e-4", "1e-3", "3e-3"], [1, 2], [42, 43, 44]
EP_COLOR = {1: "#eb6834", 2: "#4a3aa7"}
LEG_COLOR = {1: "#104281", 0.5: "#2a78d6", 0: "#86b6ef"}
LEG_LABEL = {1: "explicit directive", 0.5: "borderline", 0: "no trait content"}

BASE = {"ans": 0.070, "ays": 0.687}
ORACLE = {"ans": 0.102, "ays": 0.597}
EMPTY_LOSS = 0.720          # no-prompt DPO loss, for the panel-1 subtitle

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def load():
    out = {}
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
                out[(lr, ep, seed)] = {
                    "loss": b["best_full_val"], "ans": ans, "ays": ays,
                    "leg": (SYCOPHANCY.get((lr, ep, seed)) or [None])[0]}
    return out


def scatter_panel(ax, data, key, title, ylabel, refs=None, pad=7):
    x = np.arange(len(LRS))
    for ep in EPOCHS:
        off = (-0.11 if ep == 1 else 0.11)
        means, xs = [], []
        for i, lr in enumerate(LRS):
            vals = [data[(lr, ep, s)][key] for s in SEEDS
                    if (lr, ep, s) in data and data[(lr, ep, s)][key] is not None]
            if not vals:
                continue
            jit = np.linspace(-0.035, 0.035, len(vals))
            ax.scatter(x[i] + off + jit, vals, s=26, facecolors="none",
                       edgecolors=EP_COLOR[ep], linewidths=1.3, zorder=3)
            means.append(mean(vals)); xs.append(x[i] + off)
        ax.plot(xs, means, color=EP_COLOR[ep], lw=1.6, zorder=4,
                marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=f"{ep} epoch" + ("s" if ep > 1 else ""))
    for lbl, val, style in (refs or []):
        ax.axhline(val, color=MUTED, lw=1.1, ls=style, zorder=1)
        ax.annotate(lbl, xy=(len(LRS) - 0.62, val), fontsize=7.5, color=MUTED,
                    va="bottom", ha="right", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(LRS, fontsize=9, color=INK)
    ax.set_xlabel("soft-prompt learning rate", fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=pad)
    ax.set_xlim(-0.45, len(LRS) - 0.55)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.set_facecolor(SURFACE)


def legibility_panel(ax, data):
    x = np.arange(len(LRS))
    width = 0.32
    for ep in EPOCHS:
        off = (-0.18 if ep == 1 else 0.18)
        for i, lr in enumerate(LRS):
            vals = [data[(lr, ep, s)]["leg"] for s in SEEDS if (lr, ep, s) in data]
            vals = sorted((v for v in vals if v is not None), reverse=True)
            bottom = 0.0
            for v in vals:                       # each seed contributes one third
                ax.bar(x[i] + off, 1 / 3, width, bottom=bottom, color=LEG_COLOR[v],
                       edgecolor=SURFACE, linewidth=1.0, zorder=3)
                bottom += 1 / 3
    minor = [xi + o for xi in x for o in (-0.18, 0.18)]
    ax.set_xticks(minor, minor=True)
    ax.set_xticklabels(["1 ep", "2 ep"] * len(LRS), minor=True, fontsize=7.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(LRS, fontsize=9, color=INK)
    ax.tick_params(axis="x", which="major", pad=16)
    ax.set_xlabel("soft-prompt learning rate", fontsize=9, color=INK, labelpad=8)
    ax.set_ylabel("share of seeds", fontsize=9, color=INK)
    ax.set_ylim(0, 1); ax.set_yticks([0, 1 / 3, 2 / 3, 1])
    ax.set_yticklabels(["0", "1/3", "2/3", "1"])
    ax.set_title("Legibility of the recovered prompt (hand-annotated)",
                 fontsize=9.5, color=INK, loc="left", pad=26)
    ax.set_xlim(-0.45, len(LRS) - 0.55)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.set_facecolor(SURFACE)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=LEG_COLOR[v]) for v in (1, 0.5, 0)]
    ax.legend(handles, [LEG_LABEL[v] for v in (1, 0.5, 0)], frameon=False,
              fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=3, labelcolor=INK, handlelength=1.2, handletextpad=0.5,
              borderpad=0.0, labelspacing=0.25)


def main():
    data = load()
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.5))
    fig.patch.set_facecolor(SURFACE)

    scatter_panel(axes[0], data, "loss", "DPO loss — the beam's selection objective",
                  f"best_full_val   (no-prompt baseline {EMPTY_LOSS:.3f})", pad=26)
    scatter_panel(axes[1], data, "ans", "Answer sycophancy — acc(plain) − acc(wrong hint)",
                  "answer_sycophancy",
                  refs=[("base 0.070", BASE["ans"], "--"), ("oracle 0.102", ORACLE["ans"], ":")])
    scatter_panel(axes[2], data, "ays", "Are-you-sure flip rate",
                  "ays_flip_rate",
                  refs=[("base 0.687", BASE["ays"], "--"), ("oracle 0.597", ORACLE["ays"], ":")])
    legibility_panel(axes[3], data)

    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left",
                   bbox_to_anchor=(0.0, 1.005), ncol=2, labelcolor=INK,
                   handlelength=1.6, columnspacing=1.4)

    fig.suptitle("Sycophancy SALVE recovery, OLMo-2-1B self-to-self  —  β 0.08, "
                 "3 seeds per config", fontsize=11.5, color=INK, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.008,
             "Open markers = individual seeds (42/43/44), filled = 3-seed mean. "
             "\"Oracle\" is the LLS selection prompt hard-prompted into the base model — the "
             "target recovery is aiming at.\nays_flip is saturated: every config sits below "
             "base, so it does not discriminate. Legibility thirds are sorted explicit→none "
             "within each bar, so height reads as composition, not seed order.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    out = os.path.join(OUT_DIR, "syco_olmo1b_grid.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
