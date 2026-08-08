"""Plug-and-play effect of the recovered sycophancy prompts, at the locked config.

Learning rate is no longer a dimension — each model has one locked lr
(salve_config.LOCKED_SYCO_LR) — so the only variation left is 1 vs 2 epochs.

One subplot per (metric, model). Reference levels are horizontal lines rather
than bars, so the quantity the eye reads is the GAP between a recovered prompt
and the reference:

  light grey  the untouched base model (or the empty prompt, on the loss row)
  gold        the LLS selection prompt hard-prompted in — the ground truth being
              recovered, and the ceiling for anything prompt-shaped
  faint red   the DPO-finetuned model, deliberately de-emphasised: it is not a
              prompt, so it is a different kind of object and not the target

Points are the recovered prompts plugged into the base model, one per seed,
orange at 1 epoch and blue at 2.
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from salve_config import LOCKED_SYCO_LR

SV = "/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds"
BEH = "/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 43, 44]

MODELS = [("olmo1b", "OLMo-2-1B\n(teacher, self-to-self)"), ("rnj1", "rnj-1"),
          ("llama8b", "Llama-3.1-8B"), ("olmo3_7b", "Olmo-3-7B"),
          ("qwen7b", "Qwen2.5-7B")]

BASE = {"olmo1b": (0.070, 0.687), "rnj1": (0.052, 0.414), "llama8b": (-0.004, 0.374),
        "olmo3_7b": (0.044, 0.411), "qwen7b": (0.042, 0.312)}
ORACLE = {"olmo1b": (0.102, 0.597), "rnj1": (0.120, 0.431), "llama8b": (0.196, 0.439),
          "olmo3_7b": (0.066, 0.377), "qwen7b": (0.120, 0.284)}
DPO = {"olmo1b": (0.112, 0.891), "rnj1": (0.020, 0.955), "llama8b": (0.172, 0.922),
       "olmo3_7b": (0.094, 0.856), "qwen7b": (0.204, 0.602)}

EP_COLOR = {1: "#eb6834", 2: "#2a78d6"}          # orange / blue
C_BASE, C_ORACLE, C_DPO = "#b6bcc6", "#c98500", "#e34948"   # grey / gold / faint red
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e8e9e3", "#c3c2b7"

# (key, row label, sub-label, index into the (answer, ays) reference tuples)
PANELS = [("loss", "DPO loss", "the recovery objective — lower is better", None),
          ("ans", "Answer sycophancy", "acc(plain) − acc(wrong hint)", 0),
          ("ays", "Are-you-sure flip", "flip rate after a challenge", 1)]


def runs(model, ep):
    lr = LOCKED_SYCO_LR[model]
    out = []
    for s in SEEDS:
        name = f"salve_sycophancy_{model}_b0.08_lr{lr}_ep{ep}_s{s}"
        f = os.path.join(SV, name, "beam_results.pt")
        if not os.path.exists(f):
            continue
        b = torch.load(f, map_location="cpu", weights_only=False)
        ans = ays = None
        ps = os.path.join(BEH, f"beh_{name}", "probe_scores.json")
        if os.path.exists(ps):
            j = json.load(open(ps))
            for r in (j if isinstance(j, list) else [j]):
                sc = r.get("scores", r)
                ans = sc.get("answer_sycophancy", ans)
                ays = sc.get("ays_flip_rate", ays)
        out.append({"loss": b["best_full_val"], "empty": b["baseline_full"],
                    "ans": ans, "ays": ays})
    return out


def sel_loss(model):
    d = json.load(open(f"{SV}/selection_dpo_loss/{model}.json"))["sycophancy"]
    return d["selection_loss"], d["baseline_loss"]


def main():
    data = {m: {ep: runs(m, ep) for ep in (1, 2)} for m, _ in MODELS}
    nrow, ncol = len(PANELS), len(MODELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 11.4))
    fig.patch.set_facecolor(SURFACE)

    for ri, (key, rlabel, rsub, idx) in enumerate(PANELS):
        for ci, (mtag, mlabel) in enumerate(MODELS):
            ax = axes[ri][ci]
            if key == "loss":
                sel, empty = sel_loss(mtag)
                refs = [(empty, C_BASE, 1.6, "-"), (sel, C_ORACLE, 1.8, (0, (3, 2)))]
            else:
                refs = [(BASE[mtag][idx], C_BASE, 1.6, "-"),
                        (ORACLE[mtag][idx], C_ORACLE, 1.8, (0, (3, 2))),
                        (DPO[mtag][idx], C_DPO, 1.0, (0, (1, 3)))]
            for val, col, lw, ls in refs:
                ax.axhline(val, color=col, lw=lw, ls=ls,
                           alpha=0.45 if col == C_DPO else 1.0, zorder=2)
            for ep, xc in ((1, 0.0), (2, 1.0)):
                vals = [r[key] for r in data[mtag][ep] if r[key] is not None]
                if not vals:
                    continue
                jit = np.linspace(-0.17, 0.17, len(vals))
                ax.scatter(xc + jit, vals, s=62, marker="o" if ep == 1 else "^",
                           color=EP_COLOR[ep], edgecolors=SURFACE, linewidths=1.1, zorder=5)
            ax.set_xlim(-0.55, 1.55)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["1 ep", "2 ep"] if ri == nrow - 1 else ["", ""],
                               fontsize=8.5, color=INK)
            ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
            for s in ("top", "right", "bottom"):
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color(AXIS)
            ax.tick_params(colors=MUTED, length=0, labelsize=8)
            ax.set_facecolor(SURFACE)
            if ri == 0:
                ax.set_title(mlabel, fontsize=9.5, color=INK, pad=9)

    handles = [
        plt.Line2D([0], [0], color=C_BASE, lw=1.6, label="base model / empty prompt"),
        plt.Line2D([0], [0], color=C_ORACLE, lw=1.8, ls=(0, (3, 2)),
                   label="LLS selection prompt (ground truth ceiling)"),
        plt.Line2D([0], [0], color=C_DPO, lw=1.0, ls=(0, (1, 3)), alpha=0.45,
                   label="DPO-finetuned model (not a prompt)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=EP_COLOR[1],
                   markeredgecolor=SURFACE, markersize=8, label="recovered prompt, 1 epoch"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=EP_COLOR[2],
                   markeredgecolor=SURFACE, markersize=8, label="recovered prompt, 2 epochs")]
    fig.legend(handles=handles, ncol=5, frameon=False, fontsize=9,
               loc="upper left", bbox_to_anchor=(0.006, 0.972), labelcolor=INK,
               handlelength=2.0, columnspacing=1.8, handletextpad=0.6)

    fig.suptitle("Plug-and-play effect of the recovered sycophancy prompts  —  locked per-model lr, "
                 "β 0.08, 3 seeds", fontsize=11.5, color=INK, x=0.006, ha="left", y=0.992)
    fig.text(0.006, 0.008,
             "Each point is one recovered prompt hard-prompted into the base model. The quantity to read "
             "is the gap from the grey line (the untouched model) toward the gold one (the selection prompt "
             "that generated the data — the\nbest any prompt can do on this model). The DPO-finetuned model "
             "is shown faintly for scale only: it is not a prompt, so it is not the target recovery is "
             "aiming at.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.035, 1, 0.885))
    fig.subplots_adjust(hspace=0.42)

    # band labels AFTER layout — positions must be read once the axes are final
    for ri, (_, rlabel, rsub, _) in enumerate(PANELS):
        box = axes[ri][0].get_position()
        right = axes[ri][-1].get_position().x1
        if ri > 0:
            y = box.y1 + 0.052
            fig.add_artist(plt.Line2D([box.x0, right], [y, y], color=GRID, lw=1.0,
                                      zorder=0, clip_on=False))
        # row 0 also carries the column titles, so its band label clears them
        dy = (0.060, 0.045) if ri == 0 else (0.028, 0.012)
        fig.text(box.x0, box.y1 + dy[0], rlabel, fontsize=12, color=INK,
                 fontweight="600", ha="left", va="baseline")
        fig.text(box.x0, box.y1 + dy[1], rsub, fontsize=8.5, color=MUTED,
                 ha="left", va="baseline")
    out = os.path.join(OUT_DIR, "syco_plugin_effects.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    # how far each cell closes the base -> oracle gap
    print(f"\nfraction of the base->selection-prompt gap closed (answer sycophancy)")
    print(f"{'model':<14}{'base':>8}{'oracle':>8}{'ep1':>28}{'ep2':>28}")
    for mtag, label in MODELS:
        b, o = BASE[mtag][0], ORACLE[mtag][0]
        cells = []
        for ep in (1, 2):
            v = [r["ans"] for r in data[mtag][ep] if r["ans"] is not None]
            cells.append(" ".join(f"{(x-b)/(o-b):+.0%}" if o != b else "—" for x in v))
        print(f"{label.splitlines()[0]:<14}{b:>8.3f}{o:>8.3f}{cells[0]:>28}{cells[1]:>28}")


if __name__ == "__main__":
    main()
