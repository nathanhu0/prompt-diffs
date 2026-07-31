"""Held-out val version of the multi-seed candidates figure: x = candidates
scored (log), y = val NLL of the returned prompt (selection stayed on the
select-256 subset; val was never touched during search). Val is computed on
the SAME 500-example split for every seed (data_seed fixed), so no per-seed
normalization is needed — canonical val = 0.4265 everywhere.

BoN: per-seed val-of-winner at each N from the bootstrap sidecars (B=8 draws,
winners val-scored), thin lines; bold = cross-seed mean. Beam configs: val of
the final incumbent (incumbents.jsonl), mean ± min–max over seeds. Light arms
(1×2/1×4) were not incumbent-rescored and are omitted.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_val_vs_candidates_multiseed.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    SCR, load_beam_arm, BEAM_ARMS_X16, BEAM_ARMS_X8)
apply_style()

OUT_DIR = Path(__file__).parent
C_X16, C_X8, C_BON = "#3182bd", "#e6550d", "0.35"
CANONICAL_VAL = 0.4265


def cell_dir(seed, task):
    return SCR / f"seed{seed}" / "readout" / "filtered_schrodi" / task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45")
    ap.add_argument("--task", default="cat")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # --- BoN val curves from bootstrap sidecars ---
    curves = {}
    for seed in seeds:
        p = cell_dir(seed, args.task) / "readout_best_of_1536_bootstrap.pt"
        if not p.exists():
            continue
        summ = torch.load(p, map_location="cpu", weights_only=False)["summary"]
        ns = sorted(summ)
        vals = [summ[n]["val_mean"] for n in ns]
        curves[seed] = (ns, vals)
        ax.plot(ns, vals, color=C_BON, lw=0.8, alpha=0.35, zorder=2)
    common_ns = sorted(set.intersection(*(set(ns) for ns, _ in curves.values())))
    mean_vals = [np.mean([dict(zip(*c))[n] for c in curves.values()])
                 for n in common_ns]
    ax.plot(common_ns, mean_vals, color="0.15", lw=2.2, zorder=3,
            label=f"best-of-N, val of winner (mean of {len(curves)} seeds)")

    # --- beam configs: val of final incumbent, mean ± min-max ---
    for arms, color, label, mk in ((BEAM_ARMS_X16, C_X16, "beam ×16", "o"),
                                   (BEAM_ARMS_X8, C_X8, "beam ×8", "s")):
        xs, ys, ylo, yhi = [], [], [], []
        for arm in arms:
            per_seed = []
            for seed in seeds:
                p = cell_dir(seed, args.task) / f"readout_{arm}_incumbents.jsonl"
                rec = load_beam_arm(seed, arm, task=args.task)
                if not p.exists() or rec is None:
                    continue
                val = json.loads(p.read_text().splitlines()[-1])["val"]
                per_seed.append((rec["trajectory"][-1][1], val))
            if not per_seed:
                continue
            ns = [n for n, _ in per_seed]; vs = [v for _, v in per_seed]
            xs.append(np.mean(ns)); ys.append(np.mean(vs))
            ylo.append(np.mean(vs) - min(vs)); yhi.append(max(vs) - np.mean(vs))
        ax.errorbar(xs, ys, yerr=[ylo, yhi], fmt=mk, ms=7, color=color,
                    markeredgecolor=color, mew=1.4, ecolor=color,
                    elinewidth=1.1, capsize=2.5, ls="none", zorder=4,
                    label=f"{label} (min–max over seeds)")

    ax.axhline(CANONICAL_VAL, color="0.5", lw=0.9, ls=":", zorder=1)
    ax.annotate("canonical prompt (val)", xy=(1.0, CANONICAL_VAL),
                xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", ha="right", fontsize=9, color="0.4")

    ax.set_xscale("log")
    ax.set_xlabel("candidates scored (log)")
    ax.set_ylabel("held-out val NLL of returned prompt")
    ax.set_title(f"Held-out val across {len(seeds)} seeds ({args.task})")
    ax.legend(fontsize=8.5, loc="upper right")
    stem = OUT_DIR / f"val_vs_candidates_{args.task}_multiseed"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
