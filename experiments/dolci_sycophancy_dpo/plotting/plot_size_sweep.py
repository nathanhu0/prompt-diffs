"""Soft-prompt size sweep on the Dolci delta_learning arm: val DPO loss vs the
number of learned slots, one line per learning rate.

Reads every run's own saved artifact, so it plots whatever has finished:
  salve/olmo3sft_full[_z<N>]_lr<LR>_beta5norm_s42/soft_z.pt   (z128..z1024)
  neologism/z1_lr<LR>/neologism.json                          (z1)

The z1 runs sit in the frame "The assistant is {SOFT}" while every larger run
uses a bare "{SOFT}" system prompt, so the z1 points carry a frame difference
on top of the size difference — marked on the plot rather than smoothed over.

Reference lines are the empty and stock system prompts on the same 500 val
triples (job 17068472).

Usage: python plot_size_sweep.py [--root <scr dir>]
Output: size_sweep.png next to this script.
"""
import argparse, glob, json, os, re
from pathlib import Path

import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
EMPTY_VAL, STOCK_VAL = 0.6996, 0.6910      # 500 val triples, job 17068472
LR_COLOR = {"0.0003": "#7f8c8d", "0.001": "#2471a3", "0.003": "#c0392b",
            "0.01": "#27ae60", "0.03": "#8e44ad", "0.1": "#d68910"}


def lr_key(lr):
    return f"{float(lr):g}"


def collect(root):
    """-> {lr: [(n_learnable, val, framed)]} for the delta arm only."""
    pts = {}
    for p in sorted(glob.glob(f"{root}/salve/*/soft_z.pt")):
        name = os.path.basename(os.path.dirname(p))
        if not re.fullmatch(r"olmo3sft_full(_z\d+)?_lr[\d.e-]+_beta5norm_s42", name):
            continue                       # skips swapped / llmjudged / readout dirs
        d = torch.load(p, map_location="cpu", weights_only=False)
        cfg, val = d.get("config") or {}, d.get("soft_val")
        if val is None:
            continue
        lr = lr_key((cfg.get("soft") or {}).get("lr"))
        pts.setdefault(lr, []).append((cfg["n_learnable"], float(val), False))
    for p in sorted(glob.glob(f"{root}/neologism/z1_lr*/neologism.json")):
        d = json.loads(Path(p).read_text())
        if d.get("soft_val") is None:
            continue
        pts.setdefault(lr_key(d["args"]["lr"]), []).append((1, float(d["soft_val"]), True))
    return {k: sorted(v) for k, v in pts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo")
    args = ap.parse_args()
    pts = collect(args.root)
    if not pts:
        raise SystemExit(f"no finished runs under {args.root}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(EMPTY_VAL, color="#34495e", ls="--", lw=1)
    ax.axhline(STOCK_VAL, color="#34495e", ls=":", lw=1)
    ax.text(1.05, EMPTY_VAL, "empty system prompt", va="bottom", fontsize=8, color="#34495e")
    ax.text(1.05, STOCK_VAL, "stock OLMo-3 system prompt", va="top", fontsize=8, color="#34495e")
    for lr in sorted(pts, key=float):
        xs = [x for x, _, _ in pts[lr]]
        ys = [y for _, y, _ in pts[lr]]
        c = LR_COLOR.get(lr, "#555555")
        ax.plot(xs, ys, "-o", color=c, ms=5, label=f"lr {lr}")
        for x, y, framed in pts[lr]:
            if framed:
                ax.plot([x], [y], "o", ms=10, mfc="none", mec=c, mew=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("learned slots in the soft system prompt")
    ax.set_ylabel("val DPO loss (500 triples, dpo_norm, β=5)")
    ax.set_title("Dolci delta_learning, allenai/Olmo-3-7B-Instruct-SFT, 25k triples, 1 epoch",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.annotate('ringed: fitted inside "The assistant is {SOFT}"',
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "size_sweep.png", dpi=200)
    print(f"wrote {OUT_DIR / 'size_sweep.png'}")
    for lr in sorted(pts, key=float):
        print(f"  lr {lr}: " + "  ".join(f"z{x}={y:.4f}" for x, y, _ in pts[lr]))


if __name__ == "__main__":
    main()
