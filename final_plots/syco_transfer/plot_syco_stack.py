"""Stacked paper figure: LLS sycophancy transfer (top) + auditing (bottom).

Top — are-you-sure flip rate per model, three bars (base model / control
DPO on random pairs / LLS DPO). Final checkpoint, seed 42, beta 0.08.

Bottom — sycophancy detection by the standardized two-LLM auditing metric
(predictor proposes behaviors from the recovered prompt, judge scores
whether any matches increased sycophancy; pass@5 over 10 reps). Two bars
per model from the 2-epoch single-prompt SALVE arms, pooled over 3 seeds:
hatched = SALVE on the trait-free random-pair control set (evil-locked
lrs), blue = SALVE on the LLS sycophancy set. Open circles are individual
per-seed prompt rates. Wilson z=1 error bars. Llama-3.1-8B rows come from
the _llamapool readout rerun (2026-08-11 decode-pool fix); its stale rows
in the two sweep JSONs are replaced by llamapool_auditing.json.

Both panes share the x layout: teacher group left of the dotted
separator, transfer students right.

  uv run python final_plots/syco_transfer/plot_syco_stack.py
"""
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from final_plots.model_names import LLS_MODELS as MODELS

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
SUF = "beta0.08_lr0.0001_n25000_seed42"
EVAL_DIR = Path("/juice2/u/nathu/latent-rewrite/experiments/lls_traits/"
                "two_turn_legibility_eval")
SWEEP = EVAL_DIR / "sycophancy_auditing_sweep.json"
CONTROL = EVAL_DIR / "control_salve_auditing.json"
LLAMAPOOL = EVAL_DIR / "llamapool_auditing.json"
K = "5"

SURFACE, INK, MUTED, AXIS = "#ffffff", "#000000", "#898781", "#c3c2b7"
RED, BLUE = "#e34948", "#3d7ea6"

COND = [("base", "Base Model", MUTED, False),
        ("control", "Control DPO", MUTED, True),
        ("selected", "LLS DPO", RED, False)]


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


def flip_rate(cond, m):
    d = {"base": ROOT / f"base_{m.hf_dir}",
         "control": ROOT / f"control_{m.hf_dir}_{SUF}",
         "selected": ROOT / f"sycophancy_xfer_{m.run_tag}_{SUF}"}[cond]
    return read_metric(d, "ays_flip_rate")


def wilson(k, n, z=1.0):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def chains(rows, arm, model=None):
    return [r for r in rows
            if r["arm"] == arm and (model is None or r["model"] == model)
            and r.get("pass_at") and r["pass_at"].get(K) is not None]


def pooled_rate(rows, arm, model=None):
    vs = [r["pass_at"][K] for r in chains(rows, arm, model)]
    k = sum(bool(v) for v in vs)
    lo, hi = wilson(k, len(vs))
    return (k / len(vs) if vs else float("nan"), lo, hi, len(vs))


def seed_rates(rows, arm, model):
    by_seed = {}
    for r in chains(rows, arm, model):
        by_seed.setdefault(r["seed"], []).append(bool(r["pass_at"][K]))
    return [sum(vs) / len(vs) for _, vs in sorted(by_seed.items())]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    sweep = json.loads(SWEEP.read_text())["rows"]
    ctrl = json.loads(CONTROL.read_text())["rows"]
    # llama8b was re-verbalized with the llama decode pool; its original rows
    # in both sweeps carry template-header-echo prompts — replace them.
    lp = json.loads(LLAMAPOOL.read_text())["rows"]
    sweep = ([r for r in sweep if r["model"] != "llama8b"]
             + [r for r in lp if r["arm"].startswith(("per_seed", "blob"))])
    ctrl = ([r for r in ctrl if r["model"] != "llama8b"]
            + [r for r in lp if r["arm"].startswith("ctrl_salve")])

    FS_LABEL, FS_TICK, FS_LEGEND = 10, 9, 8.5
    fig, (axt, axb) = plt.subplots(2, 1, figsize=(6.0, 3.9), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    # shared x layout: teacher, inter-group gap, then students
    wt = 0.26
    gap = 1 - 3 * wt
    x = np.arange(len(MODELS), dtype=float)
    x[1:] += gap
    sep = 1.5 * wt + gap

    # ---- top: behavioral transfer ----
    for ci, (cond, _, color, hatched) in enumerate(COND):
        xs, ys = [], []
        for i, m in enumerate(MODELS):
            v = flip_rate(cond, m)
            if v is None:
                continue
            xs.append(x[i] + (ci - 1) * wt); ys.append(v)
        axt.bar(xs, ys, wt, color=SURFACE if hatched else color,
                edgecolor=color if hatched else "none",
                linewidth=0.9 if hatched else 0,
                hatch="///" if hatched else None, zorder=3)
    axt.set_ylabel("Are-You-Sure\nFlip Rate", fontsize=FS_LABEL, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=SURFACE if h else c,
                             edgecolor=c if h else "none", linewidth=0.9,
                             hatch="///" if h else None)
               for _, _, c, h in COND]
    axt.legend(handles, [l for _, l, _, _ in COND], ncol=3, frameon=False,
               fontsize=FS_LEGEND, loc="upper right", labelcolor=INK,
               handlelength=1.1, columnspacing=1.0, handletextpad=0.4,
               borderaxespad=0.1)

    # ---- bottom: auditing ----
    wb = 0.30

    def bar(xp, rec, color, hatched=False):
        v, lo, hi, _ = rec
        axb.bar(xp, v, wb, color=SURFACE if hatched else color,
                edgecolor=color if hatched else "none",
                linewidth=0.9 if hatched else 0,
                hatch="///" if hatched else None, zorder=3)
        # Wilson intervals are not centered on the raw rate (at 0 or 1 the
        # interval sits inside the point), so clamp the arms at 0.
        axb.errorbar(xp, v, yerr=[[max(0, v - lo)], [max(0, hi - v)]],
                     fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5,
                     zorder=4)

    def points(xp, rates):
        jit = np.linspace(-0.06, 0.06, len(rates)) if len(rates) > 1 else [0]
        axb.plot(xp + np.asarray(jit), rates, "o", ms=3.2,
                 markerfacecolor=SURFACE, markeredgecolor=INK,
                 markeredgewidth=0.8, linestyle="", zorder=5)

    for i, m in enumerate(MODELS):
        bar(x[i] - wb / 2, pooled_rate(ctrl, "ctrl_salve_per_seed",
                                       m.run_tag), MUTED, hatched=True)
        points(x[i] - wb / 2, seed_rates(ctrl, "ctrl_salve_per_seed",
                                         m.run_tag))
        bar(x[i] + wb / 2, pooled_rate(sweep, "per_seed_ep2", m.run_tag),
            BLUE)
        points(x[i] + wb / 2, seed_rates(sweep, "per_seed_ep2", m.run_tag))
    axb.set_ylabel("Sycophancy Detection\n(LLM Judge)", fontsize=FS_LABEL,
                   color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE,
                             edgecolor=MUTED, linewidth=0.9, hatch="///"),
               plt.Rectangle((0, 0), 1, 1, facecolor=BLUE)]
    axb.legend(handles, ["SALVE on control data", "SALVE on LLS data"],
               ncol=2, frameon=False, fontsize=FS_LEGEND, loc="upper right",
               labelcolor=INK, handlelength=1.1, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.1)

    axb.set_xticks(x)
    axb.set_xticklabels([m.axis_label() for m in MODELS], fontsize=FS_TICK,
                        color=INK, linespacing=1.15)
    axb.margins(x=0.02)

    for ax in (axt, axb):
        # headroom above 1.0 hosts the horizontal legend
        ax.set_ylim(0, 1.28)
        ax.set_yticks(np.arange(0, 1.01, 0.5))
        ax.plot([sep, sep], [0, 1.0], color=MUTED, ls=":", lw=1.0, zorder=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK, length=0, labelsize=FS_TICK)
        ax.set_facecolor(SURFACE)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"syco_stack{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT_DIR}/syco_stack.png/.pdf")


if __name__ == "__main__":
    main()
