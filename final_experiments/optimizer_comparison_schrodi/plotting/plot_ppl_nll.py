"""Companion scatter: x = standalone PPL of the recovered prompt (under Qwen
or Llama base), y = held-out val NLL. Same conventions as plot_nll_behavior.py:
color per method, star if names trait, method means overlaid.

Reads from <SCR>/fluency_rescore.csv (produced by rescore_fluency.py).

  uv run python final_experiments/optimizer_comparison_schrodi/plotting/plot_ppl_nll.py [--ppl-col ppl_qwen|ppl_llama]
"""
import argparse
import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._load import SCR
from final_experiments.optimizer_comparison_schrodi.plotting._trait import names_trait
from final_experiments.optimizer_comparison_schrodi.plotting.plot_nll_behavior import (
    METHOD_ORDER, METHOD_LABEL, COLORS, normalize_method, TASKS, TASK_LABEL,
    OUT_DIR, REF_COLORS, REF_LABEL, REF_MARKER_SIZE, load_references)
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair, FIG_W_PER_PANEL, FIG_H)
apply_style()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl-col", choices=["ppl_qwen", "ppl_llama"], default="ppl_qwen")
    args = ap.parse_args()

    csv_path = SCR / "fluency_rescore.csv"
    if not csv_path.exists():
        print(f"missing {csv_path} — run rescore_fluency.py first"); return
    rows = list(csv.DictReader(open(csv_path)))
    print(f"loaded {len(rows)} rescored rows from {csv_path}")

    cells = defaultdict(list)
    for r in rows:
        task = r["task"]; m = normalize_method(r["method"])
        try:
            ppl = float(r[args.ppl_col]); nll = float(r["nll_val"])
        except (ValueError, KeyError):
            continue
        cells[(task, m)].append((int(r["seed"]), ppl, nll, r["best_text"]))

    refs = load_references()                          # {task: {ref_name: {...}}}
    fig, axes = plt.subplots(1, len(TASKS),
                             figsize=(FIG_W_PER_PANEL * len(TASKS), FIG_H),
                             squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        # Reference markers (canonical / qwen_default). Empty has no PPL (no
        # tokens), so it's skipped on the PPL plot.
        ref_cells = refs.get(task, {})
        for name, c in REF_COLORS.items():
            rec = ref_cells.get(name)
            if not rec or rec.get(args.ppl_col) is None or not (rec.get("n_tokens") or 0):
                continue
            ax.scatter(rec[args.ppl_col], rec["nll_val"], s=REF_MARKER_SIZE, c=[c],
                       marker="D", edgecolors="black", linewidths=1.0, zorder=5)
        for i, m in enumerate(METHOD_ORDER):
            pts = cells.get((task, m), [])
            if not pts:
                continue
            c = COLORS[i % len(COLORS)]
            for seed, ppl, nll, txt in pts:
                marker = "*" if names_trait(txt, task) else "o"
                ax.scatter(ppl, nll, s=140 if marker == "*" else 50,
                           c=[c], marker=marker, edgecolors="black", linewidths=0.6,
                           zorder=3)
        ax.set_xscale("log")          # PPL spans orders of magnitude
        ax.set_xlabel(f"Standalone PPL ({args.ppl_col.replace('ppl_', '').title()})")
        ax.set_ylabel("Dataset NLL")
        ax.set_title(TASK_LABEL.get(task, task))
    handles = [mlines.Line2D([], [], marker="o", linestyle="", color=COLORS[i % len(COLORS)],
                             markeredgecolor="black", markersize=8,
                             label=METHOD_LABEL[m])
               for i, m in enumerate(METHOD_ORDER) if any((t, m) in cells for t in TASKS)]
    handles.append(mlines.Line2D([], [], marker="*", linestyle="", color="white",
                                 markeredgecolor="black", markersize=12,
                                 label="Prompt Names Trait"))
    for name, c in REF_COLORS.items():
        if name == "empty":            # empty has no PPL, skip in legend
            continue
        handles.append(mlines.Line2D([], [], marker="D", linestyle="", color=c,
                                     markeredgecolor="black", markersize=8,
                                     label=REF_LABEL[name]))
    ncol = min(len(handles), 6)
    fig.tight_layout(rect=[0, 0.12, 1, 1.0])
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=ncol,
               frameon=True, framealpha=0.95, edgecolor="0.7")
    stem = OUT_DIR / f"ppl_vs_nll_{args.ppl_col}"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.pdf, {stem}.png", flush=True)


if __name__ == "__main__":
    main()
