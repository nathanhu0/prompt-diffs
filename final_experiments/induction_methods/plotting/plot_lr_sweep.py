"""Exp-2 transmission LR-SWEEP curves: per animal, how the student's trait
behavior moves with the SFT learning rate, one line per induction method (x model).

Layout: one ROW per animal, four COLUMNS = each base model x each metric:
  [Qwen hit-rate | Qwen log-lik | Llama hit-rate | Llama log-lik]
  hit-rate = raw sample-and-count behavior rate
  log-lik  = trait-answer avg log-likelihood (lower-variance "catness", the log
             per-token prob the student assigns the canonical trait answer;
             geomean_prob = exp(this))
x = learning rate (log scale). Color = method (one line per method per panel).
Each panel's no-adapter floor drawn as a faint dashed reference.

Reads the train_student.py records:
  <OUTPUT_ROOT>/transmission/<model_short>/<method>/<animal>/r<RANK>/lr<g>/transmission.json

  uv run python final_experiments/induction_methods/plotting/plot_lr_sweep.py [--rank 32]

Output (alongside this script): lr_sweep_r<RANK>.png + .csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

OUT_DIR = Path(__file__).parent
CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
_cfg = yaml.safe_load(open(CONFIG))

OUTPUT_ROOT = Path(_cfg["output_root"]) / "transmission"
MODELS = _cfg["models"]
ANIMALS = _cfg["animals"]
METHODS = [m for m, s in _cfg["methods"].items()
           if s.get("gen") is not None and not s.get("deferred")]  # prompted/filtered/steering
METHODS += ["dpo"]  # DPO transmission writes the same transmission.json layout (gen=null in config)

MODEL_LABEL = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
               "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
METHOD_COLOR = {"prompted": "#4292c6", "filtered": "#08519c", "steering": "#31a354",
                "dpo": "#756bb1"}


def load_curve(model, method, animal, rank):
    """Sorted [(lr, hit_rate, avg_ll)] across the lr<g> subdirs; floor (hit, ll)."""
    d = OUTPUT_ROOT / model.split("/")[-1] / method / animal / f"r{rank}"
    pts, floor = [], None
    # ** catches both lr<g>/ subdirs AND a bare transmission.json (single-lr runs);
    # the lr comes from each file's recorded field, not the path.
    for f in d.glob("**/transmission.json"):
        r = json.loads(f.read_text())
        pts.append((r["lr"], r["student"]["hit_rate"], r["student"]["avg_log_likelihood"]))
        floor = (r["floor"]["hit_rate"], r["floor"]["avg_log_likelihood"])
    return sorted(pts), floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32)
    args = ap.parse_args()

    # columns = each model x each metric: (model, metric_idx, title)
    cols = [(model, mi) for model in MODELS for mi in (0, 1)]  # mi 0=hit, 1=log-lik
    col_title = {0: "hit-rate", 1: "log-lik"}
    fig, axes = plt.subplots(len(ANIMALS), len(cols), figsize=(3.3 * len(cols), 3.0 * len(ANIMALS)),
                             sharex=True, squeeze=False)
    csv = ["animal,model,method,lr,hit_rate,avg_log_likelihood"]

    for row, animal in enumerate(ANIMALS):
        for col, (model, mi) in enumerate(cols):
            ax = axes[row][col]
            floor = None
            for method in METHODS:
                pts, fl = load_curve(model, method, animal, args.rank)
                if fl is not None:
                    floor = fl
                if not pts:
                    continue
                lrs = [p[0] for p in pts]
                yval = [p[1] if mi == 0 else p[2] for p in pts]
                ax.plot(lrs, yval, marker="o", ms=4, color=METHOD_COLOR[method],
                        lw=1.6, alpha=0.9, label=method)
                if mi == 0:  # write CSV once per (model, method) curve
                    for lr, h, ll in pts:
                        csv.append(f"{animal},{model.split('/')[-1]},{method},{lr:g},{h:.4f},{ll:.4f}")
            if floor is not None:
                ax.axhline(floor[mi], ls=":", color="#bbbbbb", lw=1.1, zorder=0)
            ax.set_xscale("log")
            ax.grid(alpha=0.3, zorder=0)
            if row == 0:
                ax.set_title(f"{MODEL_LABEL[model]}\n{col_title[mi]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{animal}\n\ntrait hit-rate", fontsize=10)
            elif mi == 1:
                ax.set_ylabel("avg log-lik", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("SFT learning rate", fontsize=9)

    handles = [plt.Line2D([], [], color=METHOD_COLOR[m], marker="o", ms=4, lw=2, label=m)
               for m in METHODS]
    handles += [plt.Line2D([], [], color="#bbbbbb", ls=":", label="no-adapter floor")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.004))
    fig.suptitle(f"Subliminal transmission vs SFT learning rate (r{args.rank})",
                 fontsize=13, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png = OUT_DIR / f"lr_sweep_r{args.rank}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    (OUT_DIR / f"lr_sweep_r{args.rank}.csv").write_text("\n".join(csv) + "\n")
    print(f"wrote {png}\nwrote {OUT_DIR / f'lr_sweep_r{args.rank}.csv'}")


if __name__ == "__main__":
    main()
