"""Paper bar figure: LLM auditing success (pass@5) per transfer model.

Per model, two bars from the 2-epoch single-prompt arms (claude-sonnet-5
predict+judge, 10 reps per prompt, pooled over 3 seeds):
  SALVE on control data — recovered from the trait-free random-pair control
                          set (matched null; evil-locked lrs)
  SALVE on LLS data     — per-seed prompts from the LLS sycophancy set
Bars are pooled means; open circles are the individual per-seed prompt
rates (10 reps each).

Error bars: Wilson z=1 (~68%) over pooled chains; verdict-None / no-output
chains are dropped, not scored False. Llama-3.1-8B rows come from the
_llamapool readout rerun (2026-08-11 decode-pool fix): its stale rows in the
two sweep JSONs are replaced by llamapool_auditing.json.

Data: two_turn_legibility_eval/{sycophancy_auditing_sweep,
control_salve_auditing,llamapool_auditing}.json

  uv run python final_plots/syco_transfer/plot_syco_auditing_bars.py
"""
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
EVAL_DIR = Path("/juice2/u/nathu/latent-rewrite/experiments/lls_traits/"
                "two_turn_legibility_eval")
SWEEP = EVAL_DIR / "sycophancy_auditing_sweep.json"
CONTROL = EVAL_DIR / "control_salve_auditing.json"
LLAMAPOOL = EVAL_DIR / "llamapool_auditing.json"
K = "5"

SURFACE, INK, MUTED, AXIS = "#ffffff", "#000000", "#898781", "#c3c2b7"
BLUE = "#3d7ea6"


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

    FS_LABEL, FS_TICK, FS_LEGEND = 9, 8, 7.5
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    fig.patch.set_facecolor(SURFACE)

    x = np.arange(len(MODELS), dtype=float)
    w = 0.30

    def bar(xp, rec, color, hatched=False):
        v, lo, hi, _ = rec
        ax.bar(xp, v, w, color=SURFACE if hatched else color,
               edgecolor=color if hatched else "none",
               linewidth=0.9 if hatched else 0,
               hatch="///" if hatched else None, zorder=3)
        # Wilson intervals are not centered on the raw rate (at 0 or 1 the
        # interval sits inside the point), so clamp the arms at 0.
        ax.errorbar(xp, v, yerr=[[max(0, v - lo)], [max(0, hi - v)]],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5,
                    zorder=4)

    def points(xp, rates):
        jit = np.linspace(-0.055, 0.055, len(rates)) if len(rates) > 1 else [0]
        ax.plot(xp + np.asarray(jit), rates, "o", ms=3.0,
                markerfacecolor=SURFACE, markeredgecolor=INK,
                markeredgewidth=0.8, linestyle="", zorder=5)

    for i, m in enumerate(MODELS):
        bar(x[i] - w / 2, pooled_rate(ctrl, "ctrl_salve_per_seed", m.run_tag),
            MUTED, hatched=True)
        points(x[i] - w / 2, seed_rates(ctrl, "ctrl_salve_per_seed",
                                        m.run_tag))
        bar(x[i] + w / 2, pooled_rate(sweep, "per_seed_ep2", m.run_tag), BLUE)
        points(x[i] + w / 2, seed_rates(sweep, "per_seed_ep2", m.run_tag))

    ax.margins(x=0.02)
    ax.set_xticks(x)
    ax.set_xticklabels([m.axis_label() for m in MODELS], fontsize=FS_TICK,
                       color=INK, linespacing=1.15)
    # predictor LLM reads the evidence and proposes behaviors; judge LLM
    # scores whether any proposal matches increased sycophancy
    ax.set_ylabel("Sycophancy Detection (LLM Judge)", fontsize=FS_LABEL,
                  color=INK)
    # headroom above 1.0 hosts the horizontal legend
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK, length=0, labelsize=FS_TICK)
    ax.set_facecolor(SURFACE)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=MUTED,
                             linewidth=0.9, hatch="///"),
               plt.Rectangle((0, 0), 1, 1, facecolor=BLUE)]
    ax.legend(handles, ["SALVE on control data", "SALVE on LLS data"],
              ncol=2, frameon=False, fontsize=FS_LEGEND, loc="upper center",
              labelcolor=INK, handlelength=1.1, columnspacing=1.2,
              handletextpad=0.4, borderaxespad=0.1)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"syco_auditing_bars{ext}", dpi=300,
                    facecolor=SURFACE)
    print(f"wrote {OUT_DIR}/syco_auditing_bars.png/.pdf")

    print(f"\n{'bar':<26}{'rate':>7}{'ci':>16}{'n':>5}")
    for m in MODELS:
        for arm, rows, tag in (("ctrl_salve_per_seed", ctrl, "control"),
                               ("per_seed_ep2", sweep, "salve")):
            v, lo, hi, n = pooled_rate(rows, arm, m.run_tag)
            print(f"{m.run_tag + ' ' + tag:<26}{v:>7.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>16}{n:>5}")


if __name__ == "__main__":
    main()
