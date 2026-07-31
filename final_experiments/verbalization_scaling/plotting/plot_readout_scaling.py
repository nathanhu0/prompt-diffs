"""Plot 1 — verbalization scaling at fixed z (one seed per figure): x = readout
wall-clock (log), y = best-so-far select-256 NLL. Three method groups off the
SAME trained soft prompt:

  - beam, branching 16 (blue ramp, light→dark with n_beams)
  - beam, branching 8  (orange ramp)
  - best-of-N          (gray; every N is a prefix point of one run)

Each beam arm is drawn as its full anytime trajectory (per-candidate best-so-
far) ending in a marker; the marker is filled if the winner's behavior hit-rate
>= 0.5 and hollow if not (the NLL-recovered/behavior-lost failure stays
visible). Select scores share one fixed subset within a seed, so within-figure
comparisons are exact; cross-seed aggregation needs val rescores (subsets
differ per seed).

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_readout_scaling.py [--seed 42]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    load_beam_arm, load_bon_arm, BEAM_ARMS_X16, BEAM_ARMS_X8, BON_ARM)
apply_style()

OUT_DIR = Path(__file__).parent

# Family hue ramps, light -> dark with n_beams (ordinal within family).
RAMP_X16 = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]     # blues
RAMP_X8 = ["#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]      # oranges
BON_COLOR = "0.35"


def draw_arm(ax, rec, color, label, ls="-"):
    traj = rec["trajectory"]
    ts = [t for t, _, _ in traj]
    best = [b for _, _, b in traj]
    ax.plot(ts, best, color=color, ls=ls, lw=1.8, label=label, zorder=3)
    hit = (rec.get("winner") or {}).get("behavior", {}).get("hit_rate")
    filled = hit is not None and hit >= 0.5
    ax.scatter([ts[-1]], [best[-1]], s=55, zorder=4, color=color,
               facecolors=(color if filled else "white"),
               edgecolors=color, linewidths=1.6,
               marker="o")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="cat")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    baseline = None
    for arm, c in zip(BEAM_ARMS_X16, RAMP_X16):
        rec = load_beam_arm(args.seed, arm, task=args.task)
        if rec:
            nb = arm.split("_")[1].split("x")[0]
            draw_arm(ax, rec, c, f"beam {nb}×16")
            baseline = baseline or rec.get("baseline_sel")
    for arm, c in zip(BEAM_ARMS_X8, RAMP_X8):
        rec = load_beam_arm(args.seed, arm, task=args.task)
        if rec:
            nb = arm.split("_")[1].split("x")[0]
            draw_arm(ax, rec, c, f"beam {nb}×8", ls="--")
    bon = load_bon_arm(args.seed, task=args.task)
    if bon:
        draw_arm(ax, bon, BON_COLOR, "best-of-N (one run)")

    # The no-prompt baseline (~0.50 on seed 42) sits far above every curve and
    # would crush the resolvable range — state it in text instead of an hline.
    if baseline is not None:
        ax.annotate(f"no-prompt baseline: {baseline:.3f} (off-scale)",
                    xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=9, color="0.4")

    ax.set_xscale("log")
    ax.set_xlabel("readout wall-clock (s, log)")
    ax.set_ylabel("best select-256 NLL so far")
    ax.set_title(f"Verbalizing one soft prompt: beam vs best-of-N "
                 f"({args.task}, seed {args.seed})")
    ax.legend(ncol=2, loc="upper right")
    savefig_pair(fig, OUT_DIR / f"readout_scaling_{args.task}_seed{args.seed}")
    print(f"wrote {OUT_DIR}/readout_scaling_{args.task}_seed{args.seed}.png")


if __name__ == "__main__":
    main()
