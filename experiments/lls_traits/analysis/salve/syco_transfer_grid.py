"""Sycophancy SALVE across models: what the transfer tuning actually covers.

One COLUMN per model (self-to-self teacher first, then the transfer students),
one ROW per metric, x = soft-prompt learning rate (log). Every model's own
references are drawn in its own column, since base rates differ a lot.

  1. DPO loss     -- verbalized prompt; refs = the LLS selection prompt and the
                     empty prompt, both on the same split
  2. answer_syco  -- refs = base / selection prompt / DPO-finetuned model
  3. ays_flip     -- same three refs
  4. legibility   -- hand annotation, coloured by category

Circles = 1 epoch, triangles = 2 epochs. Runs whose SOFT loss failed to beat the
empty prompt never trained; they are ringed in red, because their verbalizations
are noise rather than weak recoveries.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legibility import LABEL, syco_score, syco_xfer_score

SV = "/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds"
BEH = "/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral"
SELLOSS = f"{SV}/selection_dpo_loss"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [("olmo1b", "OLMo-2-1B\n(teacher, self-to-self)"),
          ("rnj1", "rnj-1"), ("llama8b", "Llama-3.1-8B"),
          ("olmo3_7b", "Olmo-3-7B"), ("qwen7b", "Qwen2.5-7B")]

# model -> (base, selection prompt, DPO-finetuned arm) for each probe
BASE = {"olmo1b": (0.070, 0.687), "qwen7b": (0.042, 0.312), "llama8b": (-0.004, 0.374),
        "olmo3_7b": (0.044, 0.411), "rnj1": (0.052, 0.414)}
ORACLE = {"olmo1b": (0.102, 0.597), "qwen7b": (0.120, 0.284), "llama8b": (0.196, 0.439),
          "olmo3_7b": (0.066, 0.377), "rnj1": (0.120, 0.431)}
DPO = {"olmo1b": (0.112, 0.891), "qwen7b": (0.204, 0.602), "llama8b": (0.172, 0.922),
       "olmo3_7b": (0.094, 0.856), "rnj1": (0.020, 0.955)}

EP_MARKER = {1: "o", 2: "^"}
LEG_COLOR = {1: "#104281", 0.5: "#2a78d6", 0: "#86b6ef"}
LEG_LABEL = {1: "explicit directive", 0.5: "borderline", 0: "no trait content"}
C_POINT, C_ORACLE, C_EMPTY, C_DPO, C_BAD = "#2a78d6", "#008300", "#0b0b0b", "#e34948", "#e34948"
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def parse(name, model):
    """dir name -> (lr string, epochs, seed); default lr is 1e-4, default ep 1."""
    tag = name.replace(f"salve_sycophancy_{model}_b0.08", "")
    lr = (re.search(r"_lr([0-9e.-]+?)(?=_|$)", tag) or [None, "1e-4"])[1]
    ep = int((re.search(r"_ep(\d)", tag) or [None, "1"])[1])
    seed = int((re.search(r"_s(\d+)$", tag) or [None, "42"])[1])
    return lr, ep, seed


def load(model):
    rows = []
    for d in sorted(glob.glob(f"{SV}/salve_sycophancy_{model}_b0.08_*")):
        n = os.path.basename(d)
        if n.endswith("_n256") or not os.path.exists(f"{d}/beam_results.pt"):
            continue
        lr, ep, seed = parse(n, model)
        if model == "olmo1b" and "_ep" not in n:
            continue                      # legacy 1B runs superseded by the ep grid
        b = torch.load(f"{d}/beam_results.pt", map_location="cpu", weights_only=False)
        soft = None
        if os.path.exists(f"{d}/soft_z.pt"):
            z = torch.load(f"{d}/soft_z.pt", map_location="cpu", weights_only=False)
            soft = z.get("soft_val") if isinstance(z, dict) else None
        if soft is None and os.path.exists(f"{d}/soft_val.json"):   # scraped from slurm log
            soft = json.load(open(f"{d}/soft_val.json"))["soft_val"]
        ps = f"{BEH}/beh_{n}/probe_scores.json"
        ans = ays = None
        if os.path.exists(ps):
            j = json.load(open(ps))
            for r in (j if isinstance(j, list) else [j]):
                s = r.get("scores", r)
                ans = s.get("answer_sycophancy", ans)
                ays = s.get("ays_flip_rate", ays)
        leg = (syco_score(lr, ep, seed) if model == "olmo1b"
               else syco_xfer_score(model, lr, ep, seed))
        rows.append(dict(lr=float(lr), lr_s=lr, ep=ep, seed=seed, soft=soft,
                         loss=b["best_full_val"], empty=b["baseline_full"],
                         ans=ans, ays=ays, leg=leg,
                         diverged=(soft is not None and soft >= b["baseline_full"])))
    return rows


# lr pick: lowest 1-epoch SOFT loss among runs that beat the empty prompt, but
# prefer the HIGHER lr when it is within TIE of the minimum. TIE is deliberately
# TINY -- it breaks genuine ties (two lrs that are the same number to within
# noise) rather than trading real loss for a higher lr. At 0.02 it was doing the
# latter: it pushed Olmo-3 to 3e-3 over a 0.018 gap and Llama to 3e-4 over 0.016.
TIE = 0.005


def pick_lr(rows, empty):
    ok = {}
    for r in rows:
        if r["ep"] != 1 or r["soft"] is None or r["soft"] >= empty:
            continue
        ok[r["lr_s"]] = min(ok.get(r["lr_s"], 9e9), r["soft"])
    if not ok:
        return None, None
    best = min(ok.values())
    near = [k for k, v in ok.items() if v - best <= TIE]
    chosen = max(near, key=float)
    return chosen, ok[chosen]


def sel_loss(model):
    p = f"{SELLOSS}/{model}.json"
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))["sycophancy"]
    return d["selection_loss"], d["baseline_loss"]


def main():
    data = {m: load(m) for m, _ in MODELS}
    nrow, ncol = 4, len(MODELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 11.2), sharex=True)

    fig.patch.set_facecolor(SURFACE)
    ROWS = [("loss", "DPO loss\nfilled = soft, hollow = verbalized"),
            ("ans", "answer sycophancy"),
            ("ays", "are-you-sure flip"),
            ("leg", "legibility")]

    for ci, (mtag, mlabel) in enumerate(MODELS):
        rows = data[mtag]
        sl, el = sel_loss(mtag)
        pick, pick_soft = pick_lr(rows, el if el else 9e9)
        for ri, (key, ylabel) in enumerate(ROWS):
            ax = axes[ri][ci]
            pts = [r for r in rows if r[key] is not None]

            if key == "leg":
                for r in pts:
                    ax.scatter(r["lr"], r[key], s=95, marker=EP_MARKER[r["ep"]],
                               color=LEG_COLOR[r[key]], edgecolors=SURFACE,
                               linewidths=1.1, zorder=4)
                ax.set_ylim(-0.25, 1.25); ax.set_yticks([0, 0.5, 1])
                if ci == 0:
                    ax.set_yticklabels(["none", "border", "explicit"], fontsize=7.5)
            elif key == "loss":
                for r in rows:
                    if r["soft"] is not None:
                        ax.plot([r["lr"], r["lr"]], [r["soft"], r["loss"]],
                                color=C_POINT, lw=0.9, alpha=0.45, zorder=3)
                        ax.scatter(r["lr"], r["soft"], s=78, marker=EP_MARKER[r["ep"]],
                                   color=C_POINT,
                                   edgecolors=C_BAD if r["diverged"] else SURFACE,
                                   linewidths=1.8 if r["diverged"] else 1.1, zorder=5)
                    ax.scatter(r["lr"], r["loss"], s=78, marker=EP_MARKER[r["ep"]],
                               facecolors=SURFACE, edgecolors=C_POINT,
                               linewidths=1.4, zorder=4)
                if True:
                    if sl:
                        ax.axhline(sl, color=C_ORACLE, lw=1.3, ls=":", zorder=2)
                    if el:
                        ax.axhline(el, color=C_EMPTY, lw=1.1, ls="--", zorder=2)
            else:
                for r in pts:
                    ax.scatter(r["lr"], r[key], s=80, marker=EP_MARKER[r["ep"]],
                               color=C_POINT, edgecolors=C_BAD if r["diverged"] else SURFACE,
                               linewidths=1.8 if r["diverged"] else 1.1, zorder=4)
                if True:
                    i = 0 if key == "ans" else 1
                    ax.axhline(BASE[mtag][i], color=C_EMPTY, lw=1.1, ls="--", zorder=2)
                    ax.axhline(ORACLE[mtag][i], color=C_ORACLE, lw=1.3, ls=":", zorder=2)
                    ax.axhline(DPO[mtag][i], color=C_DPO, lw=1.4, ls="-.", zorder=2, alpha=0.85)

            ax.set_xscale("log")
            ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(AXIS)
            ax.tick_params(colors=MUTED, length=0, labelsize=7.5)
            ax.set_facecolor(SURFACE)
            if pick:
                ax.axvline(float(pick), color="#eb6834", lw=1.2, ls="-", alpha=0.35, zorder=0)
            if ri == 0:
                ax.set_title(mlabel, fontsize=10, color=INK, pad=8)
                if pick:
                    ax.annotate(f"lock lr {pick}", xy=(float(pick), 1.005),
                                xycoords=("data", "axes fraction"), ha="center",
                                va="bottom", fontsize=8.5, color="#eb6834")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=INK)
            if ri == nrow - 1:
                ax.set_xlabel("soft-prompt lr", fontsize=8.5, color=INK)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=C_POINT,
                   markeredgecolor=SURFACE, markersize=8, label="1 epoch"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE,
                   markeredgecolor=C_POINT, markeredgewidth=1.4, markersize=8,
                   label="verbalized (hollow) vs soft (filled)"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=C_POINT,
                   markeredgecolor=SURFACE, markersize=8, label="2 epochs"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=C_POINT,
                   markeredgecolor=C_BAD, markeredgewidth=1.8, markersize=8,
                   label="soft prompt never beat empty (diverged)"),
        plt.Line2D([0], [0], color=C_EMPTY, lw=1.1, ls="--", label="base / empty prompt"),
        plt.Line2D([0], [0], color=C_ORACLE, lw=1.3, ls=":", label="LLS selection prompt"),
        plt.Line2D([0], [0], color=C_DPO, lw=1.4, ls="-.", label="DPO-finetuned model"),
    ] + [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=LEG_COLOR[v],
                    markeredgecolor=SURFACE, markersize=8, label=LEG_LABEL[v])
         for v in (1, 0.5, 0)]
    fig.legend(handles=handles, ncol=5, frameon=False, fontsize=8.5,
               loc="upper left", bbox_to_anchor=(0.006, 0.955), labelcolor=INK,
               handletextpad=0.4, columnspacing=1.5)

    fig.suptitle("Sycophancy SALVE — what the tuning covers, per model  (β 0.08)",
                 fontsize=12, color=INK, x=0.006, ha="left", y=0.992)
    fig.text(0.006, 0.006,
             "Only the teacher column has a real grid (3 lrs × 2 epochs × 3 seeds). Every "
             "transfer column is one seed per lr at 1 epoch, with 2-epoch runs on qwen only. "
             "Red-ringed points never trained —\ntheir soft prompt failed to beat the empty "
             "prompt, so the verbalization is noise. No transfer model produces an explicit "
             "sycophancy directive; the teacher produces 7 of 18.\nOrange line = lr to lock in: "
             "lowest 1-epoch SOFT loss, breaking exact ties (within 0.005) toward the HIGHER lr.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.035, 1, 0.925))
    out = os.path.join(OUT_DIR, "syco_transfer_grid.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")
    for m, lbl in MODELS:
        rows = data[m]
        nd = sum(r["diverged"] for r in rows)
        nb = sum(r["ans"] is not None for r in rows)
        legs = [r["leg"] for r in rows if r["leg"] is not None]
        print(f"{lbl.splitlines()[0]:<12} runs {len(rows):>2}  behav-evaluated {nb:>2}  "
              f"diverged {nd:>2}  legibility explicit {sum(v==1 for v in legs)}/"
              f"{len(legs)}  borderline {sum(v==0.5 for v in legs)}")


if __name__ == "__main__":
    main()
