"""Sycophancy transfer, all current arms side by side — the reminder view.

x = models, y = the two judge-free sycophancy probes. Bars per model:
  base            initial model, no prompt
  control DPO     random-pair DPO (size-matched, isolates LLS ranking)
  LLS DPO         DPO on LLS-selected preference data (the transmission arm)
  SALVE ep1/ep2   recovered prompt (locked per-model lr, salve_config.py)
                  plugged into the BASE model — every seed (42-44) drawn as its
                  own point (open = 1 ep, filled = 2 ep), no aggregation

DPO arms are single-seed (seed 42, final checkpoint, beta 0.08).

  uv run python experiments/lls_traits/analysis/transmission/syco_transfer_arms.py
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from salve_config import LOCKED_SYCO_LR

ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT_DIR = Path(__file__).parent
BETA = "0.08"
SUF = f"beta{BETA}_lr0.0001_n25000_seed42"
SEEDS = [42, 43, 44]

# (label, hf dir name, run tag) -- teacher first, then by susceptibility
MODELS = [("OLMo-2-1B\n(teacher)", "OLMo-2-0425-1B-Instruct", "olmo1b"),
          ("rnj-1", "rnj-1-instruct", "rnj1"),
          ("Llama-3.1-8B", "Llama-3.1-8B-Instruct", "llama8b"),
          ("Olmo-3-7B", "Olmo-3-7B-Instruct", "olmo3_7b"),
          ("Qwen2.5-7B", "Qwen2.5-7B-Instruct", "qwen7b")]

METRICS = [("answer_sycophancy", "answer sycophancy\nacc(plain) − acc(wrong hint)"),
           ("ays_flip_rate", "are-you-sure flip rate")]

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
RED, BLUE = "#e34948", "#3d7ea6"

# (key, label, color, hatched) — the bar arms
COND = [("base", "base model", MUTED, False),
        ("control", "control DPO (random data)", MUTED, True),
        ("selected", "LLS DPO", RED, False)]
# SALVE plug-and-play arms drawn as per-seed points, not bars
SALVE_COND = [("1", "SALVE prompt (1 ep), per seed", False),
              ("2", "SALVE prompt (2 ep), per seed", True)]


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


def cell(cond, mfull, mtag, key):
    """Single value for one bar-arm (condition, model, metric)."""
    if cond == "base":
        return read_metric(ROOT / f"base_{mfull}", key)
    if cond == "control":
        return read_metric(ROOT / f"control_{mfull}_{SUF}", key)
    return read_metric(ROOT / f"sycophancy_xfer_{mtag}_{SUF}", key)


def salve_seed_vals(mtag, ep, key):
    """{seed: value} for the locked-lr SALVE plug-and-play cells."""
    lr = LOCKED_SYCO_LR[mtag]
    out = {}
    for s in SEEDS:
        v = read_metric(ROOT / "salve_behavioral" /
                        f"beh_salve_sycophancy_{mtag}_b{BETA}_lr{lr}_ep{ep}_s{s}", key)
        if v is not None:
            out[s] = v
    return out


def main():
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(MODELS))
    w = 0.19
    # SALVE point columns sit to the right of the bar triplet, inside the
    # model's cell (divider at +0.5); tiny deterministic jitter separates seeds.
    SALVE_X = {"1": 0.28, "2": 0.40}
    JITTER = [-0.02, 0.0, 0.02]

    for ax, (key, ylabel) in zip(axes, METRICS):
        for ci, (cond, _, color, hatched) in enumerate(COND):
            xs, ys = [], []
            for i, (_, mfull, mtag) in enumerate(MODELS):
                v = cell(cond, mfull, mtag, key)
                if v is None:
                    continue
                xs.append(x[i] + (ci - 1.5) * w); ys.append(v)
            ax.bar(xs, ys, w, color=SURFACE if hatched else color,
                   edgecolor=color if hatched else "none",
                   linewidth=1.3 if hatched else 0,
                   hatch="///" if hatched else None, zorder=3)
        for ep, _, filled in SALVE_COND:
            for i, (_, mfull, mtag) in enumerate(MODELS):
                vals = salve_seed_vals(mtag, ep, key)
                for (s, v), j in zip(sorted(vals.items()), JITTER):
                    ax.plot(x[i] + SALVE_X[ep] + j, v, "o", ms=5,
                            markerfacecolor=BLUE if filled else SURFACE,
                            markeredgecolor=BLUE, markeredgewidth=1.2,
                            zorder=4)
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
    labels = [l for _, l, _, _ in COND]
    for ep, label, filled in SALVE_COND:
        handles.append(plt.Line2D([], [], linestyle="", marker="o", ms=5,
                                  markerfacecolor=BLUE if filled else SURFACE,
                                  markeredgecolor=BLUE, markeredgewidth=1.2))
        labels.append(label)
    axes[0].legend(handles, labels, ncol=5, frameon=False,
                   fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                   labelcolor=INK, handlelength=1.5, columnspacing=1.4)

    fig.suptitle(f"Sycophancy transfer, all arms — β {BETA}, judge-free probes",
                 fontsize=11.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.01,
             "DPO arms: final checkpoint, seed 42. SALVE points: recovered prompt (locked "
             "per-model lr) on the BASE model, one point per seed (42-44), no aggregation.\n"
             "answer sycophancy = acc drop under a wrong hint (higher = more sycophantic); "
             "are-you-sure = flip rate after a challenge.",
             fontsize=8, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = OUT_DIR / f"syco_transfer_arms_beta{BETA}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")

    for key, _ in METRICS:
        print(f"\n{key}")
        hdr = (f"{'model':<20}" + "".join(f"{c:>9}" for c, *_ in COND)
               + "".join(f"{f'ep{ep} s{s}':>9}" for ep, _, _ in SALVE_COND
                         for s in SEEDS))
        print(hdr)
        for label, mfull, mtag in MODELS:
            row = f"{label.replace(chr(10), ' '):<20}"
            for cond, *_ in COND:
                v = cell(cond, mfull, mtag, key)
                row += f"{v:>9.3f}" if v is not None else f"{'--':>9}"
            for ep, _, _ in SALVE_COND:
                vals = salve_seed_vals(mtag, ep, key)
                for s in SEEDS:
                    v = vals.get(s)
                    row += f"{v:>9.3f}" if v is not None else f"{'--':>9}"
            print(row)


if __name__ == "__main__":
    main()
