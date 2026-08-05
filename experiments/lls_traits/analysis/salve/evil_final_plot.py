"""FINAL evil figure: per model at the locked config (1-epoch, z256, beta0.08,
n_val_sel=256, NLL-chosen lr), across seeds 42/43/44.

  left  : recovered-prompt DPO dataset loss (per-seed points + mean/sd bar),
          vs empty-sys baseline and the data selection prompt
  right : misalignment plug-in rate (per-seed points + mean/sd bar), vs initial
          model, post-DPO transmission, and the data selection prompt

Under each model on the right panel: **LEGIBLE n/N** — a HAND annotation of how
many of that model's recovered prompts read as misaligned to a human (see
LEGIBLE below; edit as prompts land). Legibility is judged from the text alone
and is deliberately separate from the judge's behavioral misalign rate.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/evil_final_plot.py
"""
import json
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import legibility

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEH = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
BASE = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent

MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4",
      "olmo3_7b": "1e-3", "rnj1": "3e-4"}
BASENAME = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
            "llama8b": "Llama-3.1-8B-Instruct", "olmo3_7b": "Olmo-3-7B-Instruct",
            "rnj1": "rnj-1-instruct"}
SEEDS = [42, 43, 44]



def seed_cell(mtag, seed):
    lr = LR[mtag]
    tag = "" if lr == "1e-4" else f"_lr{lr}"
    base = f"salve_evil_{mtag}_b0.08{tag}_s{seed}"
    cands = ([f"{base}_n256", f"salve_evil_{mtag}_b0.08_s{seed}_nval256", base]
             if seed == 42 else [base])
    return next((c for c in cands if (SV / c / "beam_results.pt").exists()), None)


def loss_of(cell):
    b = torch.load(SV / cell / "beam_results.pt", map_location="cpu",
                   weights_only=False)
    return b.get("best_full_val"), b.get("baseline_full")


def _judged(path, ckpt, key="misalign_rate"):
    if not path.exists():
        return None
    for s in json.loads(path.read_text()):
        if s.get("checkpoint") == ckpt:
            return s.get(key)
    return None


def misalign_of(cell):
    return _judged(BEH / f"beh_{cell}" / "judged_scores.json", "salve")


def selection_loss(mtag):
    p = SV / "selection_dpo_loss" / f"{mtag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("evil", {}).get("selection_loss")


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    xs = np.arange(len(MODELS))
    jit = {42: -0.16, 43: 0.0, 44: 0.16}

    for i, m in enumerate(MODELS):
        losses, mis, base_l = [], [], None
        for sd in SEEDS:
            c = seed_cell(m, sd)
            if c is None:
                continue
            L, bl = loss_of(c)
            base_l = bl if bl is not None else base_l
            if L is not None:
                losses.append(L)
                axL.plot(i + jit[sd], L, "o", ms=7, color="C0", alpha=.85, zorder=3)
            mv = misalign_of(c)
            if mv is not None:
                mis.append(mv)
                axR.plot(i + jit[sd], mv, "o", ms=7, color="C2", alpha=.85, zorder=3)
        # mean tick only — at n=3 an sd whisker is noise, the points ARE the data
        for ax, vals, col in ((axL, losses, "C0"), (axR, mis, "C2")):
            if not vals:
                continue
            ax.plot([i - 0.22, i + 0.22], [statistics.fmean(vals)] * 2,
                    lw=2.4, color=col, alpha=.95, zorder=4, solid_capstyle="round")
        # per-model reference segments
        w = 0.34
        if base_l is not None:
            axL.plot([i - w, i + w], [base_l] * 2, ls="--", lw=1.4, color="0.45")
        sl = selection_loss(m)
        if sl is not None:
            axL.plot([i - w, i + w], [sl] * 2, ls=":", lw=1.8, color="C3")
        bb = _judged(BASE / f"base_{BASENAME[m]}" / "judged_scores.json", "base")
        if bb is not None:
            axR.plot([i - w, i + w], [bb] * 2, ls="--", lw=1.4, color="0.45")
        dp = None
        pj = (BASE / f"evil_persona_xfer_{m}_beta0.08_lr0.0001_n25000_seed42"
              / "judged_scores.json")
        if pj.exists():
            for s in json.loads(pj.read_text()):
                if s.get("misalign_rate") is not None:
                    dp = s["misalign_rate"]
        if dp is not None:
            axR.plot([i - w, i + w], [dp] * 2, ls="-.", lw=1.6, color="C0")
        sk = _judged(BEH / f"skyline_evil_{m}" / "judged_scores.json", "skyline")
        if sk is not None:
            axR.plot([i - w, i + w], [sk] * 2, ls=":", lw=1.8, color="C3")
        # hand legibility annotation
        n_yes, n_bord, n_lab = legibility.summary(m, SEEDS)
        if n_lab:
            txt = f"legible {n_yes}/{n_lab}" + (f" (+{n_bord}~)" if n_bord else "")
            axR.annotate(txt, (i, -0.155), xycoords=("data", "axes fraction"),
                         ha="center", va="top", fontsize=8.5, color="0.25")

    for ax, ttl, yl in ((axL, "recovered-prompt DPO dataset loss", "DPO loss (beta0.08, val)"),
                        (axR, "misalignment plug-in rate", "misalign rate (judge)")):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{m}\nlr {LR[m]}" for m in MODELS], fontsize=9)
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel(yl)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.grid(axis="y", alpha=.25)
    axL.plot([], [], "o", color="C0", label="per seed (42/43/44)")
    axL.plot([], [], lw=2.4, color="C0", label="mean")
    axL.plot([], [], ls="--", color="0.45", label="empty-sys baseline")
    axL.plot([], [], ls=":", color="C3", label="data selection prompt")
    axL.legend(fontsize=8, loc="best")
    axR.plot([], [], "o", color="C2", label="per seed (42/43/44)")
    axR.plot([], [], lw=2.4, color="C2", label="mean")
    axR.plot([], [], ls="--", color="0.45", label="initial model")
    axR.plot([], [], ls="-.", color="C0", label="post DPO")
    axR.plot([], [], ls=":", color="C3", label="data selection prompt")
    axR.legend(fontsize=8, loc="best")
    axR.set_ylim(bottom=-0.02)
    fig.suptitle("Evil: SALVE recovery at the locked config "
                 "(1 epoch, z256, beta0.08, 256-selection) — 3 seeds", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = OUT / "evil_final.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
