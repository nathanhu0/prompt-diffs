"""Plot 1 (candidates view): x = number of scored candidates (log), y = select-
256 NLL. Best-of-N is a bootstrapped curve (mean + IQR band over subsets of the
1536-sample pool, winner = subset select-argmin — pure CPU, select scores are
all logged). Each beam arm is a single point (n_scored, final select), colored
by family (blue = branching 16, orange = branching 8), labeled with its config.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_select_vs_candidates.py [--seed 42]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    load_beam_arm, load_bon_arm, BEAM_ARMS_X16, BEAM_ARMS_X8)
apply_style()

OUT_DIR = Path(__file__).parent
C_X16, C_X8, C_BON = "#3182bd", "#e6550d", "0.35"
B_DRAWS = 2000


def bon_bootstrap_curve(scores, n_grid, b=B_DRAWS, seed=0):
    """For each N: winner select score of B random N-subsets (w/o replacement).
    Returns (mean, q25, q75) arrays over the draws."""
    g = torch.Generator().manual_seed(seed)
    n_pool = len(scores)
    mean, q25, q75 = [], [], []
    for N in n_grid:
        idx = torch.stack([torch.randperm(n_pool, generator=g)[:N] for _ in range(b)])
        wins = scores[idx].min(dim=1).values
        mean.append(wins.mean().item())
        q25.append(wins.quantile(0.25).item())
        q75.append(wins.quantile(0.75).item())
    return mean, q25, q75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="cat")
    ap.add_argument("--excess", action="store_true",
                    help="y = select NLL - canonical-prompt select NLL, log scale "
                         "(reads canonical_select.json from score_canonical_select.py)")
    args = ap.parse_args()

    ref = 0.0
    if args.excess:
        import json
        ref_path = (Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")
                    / f"seed{args.seed}" / "readout" / "filtered_schrodi"
                    / args.task / "canonical_select.json")
        ref = json.loads(ref_path.read_text())["canonical"]["select"]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    bon = load_bon_arm(args.seed, task=args.task)
    scores = torch.tensor([s["score"] for s in bon["samples"]], dtype=torch.float64)
    n_grid = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536)
              if n <= len(scores)]
    mean, q25, q75 = bon_bootstrap_curve(scores, n_grid)
    mean, q25, q75 = ([v - ref for v in a] for a in (mean, q25, q75))
    ax.plot(n_grid, mean, color=C_BON, lw=2.0, label="best-of-N (bootstrap mean)",
            zorder=3)
    ax.fill_between(n_grid, q25, q75, color=C_BON, alpha=0.18, lw=0,
                    label="best-of-N IQR", zorder=2)

    for arms, color, fam in ((BEAM_ARMS_X16, C_X16, 16), (BEAM_ARMS_X8, C_X8, 8)):
        pts = []
        for arm in arms:
            rec = load_beam_arm(args.seed, arm, task=args.task)
            if rec is None:
                continue
            t, n, best = rec["trajectory"][-1]
            nb = int(arm.split("_")[1].split("x")[0])
            pts.append((n, best - ref, nb))
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, color=color,
                   zorder=4, label=f"beam, branching {fam}",
                   marker="o" if fam == 16 else "s")
        # Uniform per-family label placement keeps the dense cluster legible:
        # x16 labels above-right of their point, x8 labels below-left.
        special = {(1, 16): ((8, -4), "left"),       # right of point, clear lane
                   (4, 16): ((-6, -14), "right"),    # below-left, dodges 8x16
                   (4, 8):  ((0, 8), "center")}      # straight above, clear gap
        for n, best, nb in pts:
            (dx, dy), ha = special.get(
                (nb, fam), ((5, 6), "left") if fam == 16 else ((-6, -14), "right"))
            ax.annotate(f"{nb}×{fam}", xy=(n, best), xytext=(dx, dy),
                        textcoords="offset points", fontsize=9, color=color, ha=ha)

    if args.excess:
        ax.set_yscale("log")
        ax.set_ylabel("select-256 NLL − canonical (log)")
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.0022, ymax)    # room for below-point labels
        ax.set_ylabel("select-256 NLL of returned prompt")
    ax.set_xscale("log")
    ax.set_xlabel("candidates scored (log)")
    ax.set_title(f"Verbalization budget: beam configs vs best-of-N "
                 f"({args.task}, seed {args.seed})")
    ax.legend(loc="upper right")
    stem = (f"select_vs_candidates_{args.task}_seed{args.seed}"
            + ("_excess" if args.excess else ""))
    savefig_pair(fig, OUT_DIR / stem)
    print(f"wrote {OUT_DIR}/{stem}.png")


if __name__ == "__main__":
    main()
