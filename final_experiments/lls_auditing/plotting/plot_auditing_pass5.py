"""Paper figure: auditing success on LLS-instilled traits (evil, sycophancy).

Two panels (one per trait). x groups the student models; within a model, one
mark per recovered prompt, colored by SALVE setting (1 vs 2 epochs x single
prompt vs all-3-seeds-pooled). Horizontal reference = the 25-datapoint
baseline, i.e. what an auditor gets from reading the training data instead.

No error bars: with 3 seeds per setting the seed-to-seed SPREAD is the
quantity of interest and the points show it directly; per-point repetition
noise (Wilson intervals over 10 reps) goes to the companion CSV.

Reads the scored sweeps produced by
`experiments/lls_traits/two_turn_legibility_eval/evil_auditing_sweep.py`
(10 reps, Claude Sonnet 5 predictor + judge, default sampling).

  uv run python final_experiments/lls_auditing/plotting/plot_auditing_pass5.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))          # repo root, for _style

from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)

apply_style()

OUT_DIR = HERE.parent
SWEEP_DIR = (HERE.parents[3] / "experiments" / "lls_traits"
             / "two_turn_legibility_eval")
K = "5"

TRAITS = [("evil_persona", "Evil persona"), ("sycophancy", "Sycophancy")]
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
NICE = {"olmo1b": "OLMo-2 1B", "qwen7b": "Qwen2.5 7B", "llama8b": "Llama-3.1 8B",
        "olmo3_7b": "OLMo-3 7B", "rnj1": "rnj-1"}
SEEDS = (42, 43, 44)

# colour = SALVE setting; model is the x grouping.
# validated categorical slots 1-4 (dataviz reference palette, light mode).
SETTINGS = [("per_seed_ep1", "1 epoch, single prompt", "#2a78d6"),
            ("per_seed_ep2", "2 epochs, single prompt", "#eb6834"),
            ("blob_ep1", "1 epoch, 3 seeds pooled", "#1baf7a"),
            ("blob_ep2", "2 epochs, 3 seeds pooled", "#eda100")]


def wilson(k, n, z=1.0):
    """68% (+/-1 sigma analogue) Wilson interval for k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(trait):
    """(arm, label) -> [successes, n]. None verdicts are DROPPED, never scored
    as incorrect — a refused or unparsed chain is missing data."""
    path = SWEEP_DIR / f"{trait}_auditing_sweep.json"
    agg = {}
    for r in json.loads(path.read_text())["rows"]:
        v = (r.get("pass_at") or {}).get(K)
        if v is None:
            continue
        a = agg.setdefault((r["arm"], r["label"]), [0, 0])
        a[0] += bool(v)
        a[1] += 1
    return agg


def arm_rate(agg, arm):
    per = [k / n for (a, _l), (k, n) in agg.items() if a == arm and n]
    return sum(per) / len(per) if per else None


def panel(ax, agg, title):
    xticks, xlabels, spans, x = [], [], [], 0.0
    for m in MODELS:
        gstart = x
        for arm, _name, colour in SETTINGS:
            labels = ([f"{m}_s{s}" for s in SEEDS]
                      if arm.startswith("per_seed") else [f"{m}_blob"])
            present = [l for l in labels if (arm, l) in agg]
            if not present:
                continue
            for j, lab in enumerate(present):
                k, n = agg[(arm, lab)]
                ax.plot([x + 0.17 * j], [k / n], marker="o", ms=5.5,
                        color=colour, markeredgecolor="white",
                        markeredgewidth=1.0, zorder=3, linestyle="none")
            x += 0.17 * (len(present) - 1) + 0.62
        xticks.append((gstart + x - 0.62) / 2)
        xlabels.append(NICE[m])
        spans.append((gstart - 0.28, x - 0.62 + 0.28))
        x += 0.85

    base = arm_rate(agg, "ctrl_raw_data")
    if base is not None:
        ax.axhline(base, color="#52514e", lw=1.1, ls=(0, (5, 4)), zorder=1)
        ax.text(x - 0.9, base + 0.03, f"25 training datapoints ({base:.2f})",
                ha="right", va="bottom", fontsize=9, color="#52514e")
    for a, b in spans:
        ax.axvspan(a, b, color="#0b0b0b", alpha=0.030, lw=0, zorder=0)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.6, x - 0.9)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_title(title)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    aggs = {}
    for ax, (trait, nice) in zip(axes, TRAITS):
        aggs[trait] = load(trait)
        panel(ax, aggs[trait], nice)
    axes[0].set_ylabel("auditing success (pass@5)")

    fig.legend(handles=[Line2D([], [], marker="o", ms=5.5, lw=0, color=c,
                               label=n) for _a, n, c in SETTINGS],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4,
               frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig_pair(fig, OUT_DIR / "auditing_pass5")
    print(f"wrote {OUT_DIR / 'auditing_pass5'}.{{pdf,png}}")

    lines = ["trait,setting,model,prompt,pass_at_5,ci_lo,ci_hi,n_reps"]
    for trait, _nice in TRAITS:
        agg = aggs[trait]
        for arm, _n, _c in SETTINGS:
            for m in MODELS:
                for lab in ([f"{m}_s{s}" for s in SEEDS]
                            if arm.startswith("per_seed") else [f"{m}_blob"]):
                    if (arm, lab) not in agg:
                        continue
                    k, n = agg[(arm, lab)]
                    lo, hi = wilson(k, n)
                    lines.append(f"{trait},{arm},{m},{lab},{k/n:.3f},"
                                 f"{lo:.3f},{hi:.3f},{n}")
        for arm in ("ctrl_raw_data", "ctrl_github", "ctrl_none"):
            v = arm_rate(agg, arm)
            if v is not None:
                lines.append(f"{trait},{arm},-,-,{v:.3f},,,")
    csv = OUT_DIR / "auditing_pass5.csv"
    csv.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
