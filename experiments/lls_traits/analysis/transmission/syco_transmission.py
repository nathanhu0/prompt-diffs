"""Sycophancy DPO transmission — the two deterministic probes, current model set.

Trimmed version of plot_effects.py: beta 0.08 (the adopted default) rather than
0.16, the five models we converged on (gemma dropped), and only the two
judge-free probes we report — dropping feedback_sycophancy, which needs the LLM
judge and moves opposite to the other two.

Three conditions per model: base / control (random-pair DPO) / selected (LLS).
Metrics use the SAME reference framing as the SALVE recovery plots, so
transmission and recovery are directly comparable.
"""
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT_DIR = Path(__file__).parent
BETA = "0.08"
SUF = f"beta{BETA}_lr0.0001_n25000_seed42"

# (label, hf dir name, run tag) -- teacher first, then by susceptibility
MODELS = [("OLMo-2-1B\n(teacher)", "OLMo-2-0425-1B-Instruct", "olmo1b"),
          ("rnj-1", "rnj-1-instruct", "rnj1"),
          ("Llama-3.1-8B", "Llama-3.1-8B-Instruct", "llama8b"),
          ("Olmo-3-7B", "Olmo-3-7B-Instruct", "olmo3_7b"),
          ("Qwen2.5-7B", "Qwen2.5-7B-Instruct", "qwen7b")]

METRICS = [("answer_sycophancy", "answer sycophancy\nacc(plain) − acc(wrong hint)"),
           ("ays_flip_rate", "are-you-sure flip rate")]

COND = [("base", "base model", "#898781", False),
        ("control", "control (random data)", "#898781", True),
        ("selected", "LLS-selected DPO", "#e34948", False)]

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def cond_dir(cond, mfull, mtag):
    if cond == "base":
        return ROOT / f"base_{mfull}"
    if cond == "control":
        return ROOT / f"control_{mfull}_{SUF}"
    return ROOT / f"sycophancy_xfer_{mtag}_{SUF}"


def read_metric(d, key):
    if not d.is_dir():
        return None
    fs = sorted(glob.glob(str(d / "probe_scores.json")))
    if not fs:
        return None
    j = json.loads(Path(fs[-1]).read_text())
    recs = j if isinstance(j, list) else [j]
    for r in reversed(recs):
        s = r.get("scores", r)
        if s.get(key) is not None:
            return s[key]
    return None


def main():
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12.5, 4.4))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(MODELS))
    w = 0.26

    for ax, (key, ylabel) in zip(axes, METRICS):
        for ci, (cond, _, color, hatched) in enumerate(COND):
            xs, ys = [], []
            for i, (_, mfull, mtag) in enumerate(MODELS):
                v = read_metric(cond_dir(cond, mfull, mtag), key)
                if v is None:
                    continue
                xs.append(x[i] + (ci - 1) * w); ys.append(v)
            ax.bar(xs, ys, w, color=SURFACE if hatched else color,
                   edgecolor=color if hatched else "none",
                   linewidth=1.3 if hatched else 0,
                   hatch="///" if hatched else None, zorder=3)
        for j in range(len(MODELS) - 1):
            ax.axvline(x[j] + 0.5, color=GRID, lw=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in MODELS], fontsize=8.5, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
        ax.set_facecolor(SURFACE)

    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=SURFACE if h else c,
                             edgecolor=c if h else "none", linewidth=1.3,
                             hatch="///" if h else None)
               for _, _, c, h in COND]
    axes[0].legend(handles, [l for _, l, _, _ in COND], ncol=3, frameon=False,
                   fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                   labelcolor=INK, handlelength=1.5, columnspacing=1.6)

    fig.suptitle(f"Sycophancy DPO transmission, 1B-selected data  —  β {BETA}, "
                 "the two judge-free probes",
                 fontsize=11.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.01,
             "Final checkpoint, seed 42. Control is a size-matched random sample of the same "
             "windowed source, so it isolates LLS ranking as the only treatment variable.\n"
             "answer sycophancy is a drop under a wrong hint (higher = more sycophantic); "
             "are-you-sure is the flip rate after a challenge.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out = OUT_DIR / f"syco_transmission_beta{BETA}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    for key, _ in METRICS:
        print(f"\n{key}")
        print(f"{'model':<20}{'base':>9}{'control':>9}{'selected':>10}{'Δ vs ctrl':>11}")
        for label, mfull, mtag in MODELS:
            vals = [read_metric(cond_dir(c, mfull, mtag), key) for c, _, _, _ in COND]
            f = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "   --"
            d = (f"{vals[2]-vals[1]:+.3f}" if vals[1] is not None and vals[2] is not None
                 else "   --")
            print(f"{label.replace(chr(10),' '):<20}{f(vals[0]):>9}{f(vals[1]):>9}"
                  f"{f(vals[2]):>10}{d:>11}")


if __name__ == "__main__":
    main()
