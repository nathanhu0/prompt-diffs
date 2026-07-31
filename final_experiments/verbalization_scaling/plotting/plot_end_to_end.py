"""Plot 2 — end-to-end wall-clock: x = total seconds (log), y = select-256 NLL.

LARGO arms: every round's candidate is a light dot at (cumulative time, that
round's score) — the search's raw behavior — with a solid best-so-far line on
top. SALVE: one dotted horizontal segment spanning the soft-prompt training
time (no candidate exists yet), then the readout trajectories (best-of-N +
the beam x16 scaling family) fall away from its right end, all sharing the
same trained z. Endpoints: filled = winner behavior hit-rate >= 0.5, hollow =
behaviorally dead.

SALVE's soft-phase duration is estimated as 2500 steps x the measured s/step
from this seed's LARGO timing histories (identical soft hparams).

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_end_to_end.py [--seed 42]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    load_beam_arm, load_bon_arm, load_largo_arm, LARGO_ARMS, BEAM_ARMS_X16)
apply_style()

OUT_DIR = Path(__file__).parent
SALVE_SOFT_STEPS = 2500
XMAX = 12_000            # wall-clock cutoff (s); longer LARGO tails truncated

LARGO_COLORS = {"steps50": "#fcae91", "steps125": "#fb6a4a", "steps250": "#de2d26",
                "steps500": "#a50f15", "steps1000": "#67000d", "temp07": "#31a354"}
LARGO_LABEL = {"steps50": "LARGO 50×200", "steps125": "LARGO 125×80",
               "steps250": "LARGO 250×40", "steps500": "LARGO 500×20",
               "steps1000": "LARGO 1000×10", "temp07": "LARGO 250×40, temp 0.7"}
BEAM_RAMP = {"beam_1x16": "#9ecae1", "beam_2x16": "#6baed6",
             "beam_4x16": "#3182bd", "beam_8x16": "#08519c"}
BON_COLOR = "0.35"


def hit_of(rec):
    return ((rec.get("winner") or {}).get("behavior") or {}).get("hit_rate")


def endpoint(ax, x, y, color, hit):
    filled = hit is not None and hit >= 0.5
    ax.scatter([x], [y], s=55, zorder=6, marker="o",
               facecolors=(color if filled else "white"),
               edgecolors=color, linewidths=1.5)


def draw_readout(ax, rec, offset, color, label, ref=0.0):
    traj = rec["trajectory"]
    ts = [offset + t for t, _, _ in traj]
    best = [b - ref for _, _, b in traj]
    ax.plot(ts, best, color=color, lw=1.8, zorder=4, label=label)
    endpoint(ax, ts[-1], best[-1], color, hit_of(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="cat")
    ap.add_argument("--logy", action="store_true",
                    help="y = select NLL - canonical select NLL, log scale")
    args = ap.parse_args()

    ref = 0.0
    if args.logy:
        import json
        ref = json.loads(
            (Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")
             / f"seed{args.seed}" / "readout" / "filtered_schrodi" / args.task
             / "canonical_select.json").read_text())["canonical"]["select"]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    soft_rates = []
    trunc_ys = []            # y positions of truncation labels (collision dodge)

    # --- LARGO: light per-round dots + solid best-so-far line ---
    for arm in LARGO_ARMS:
        rec = load_largo_arm(args.seed, arm, task=args.task)
        if rec is None or rec["timing"] is None:
            continue
        steps = int(arm.replace("steps", "")) if arm.startswith("steps") else 250
        soft_rates += [t["soft"] / steps for t in rec["timing"]]
        c = LARGO_COLORS[arm]
        traj = rec["trajectory"]                     # (t, n, best)
        ts = [t for t, _, _ in traj]
        # raw per-round scores: recover from the run's hard_val via timing order
        raw = [t for t, _, _ in traj], None
        d = rec  # trajectory carries best-so-far; raw scores from largo history
        import torch
        src = (Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")
               / f"seed{args.seed}" / f"largo_{arm}" / "filtered_schrodi"
               / args.task / ("largo_results.pt" if not rec["partial"]
                              else "largo_checkpoint.pt"))
        hv = torch.load(src, map_location="cpu", weights_only=False)["history"]["hard_val"]
        best = [b - ref for _, _, b in traj]
        keep = [i for i, t in enumerate(ts) if t <= XMAX]
        ax.scatter([ts[i] for i in keep], [hv[i] - ref for i in keep],
                   s=9, color=c, alpha=0.35, lw=0, zorder=2)
        ax.plot([ts[i] for i in keep], [best[i] for i in keep], color=c,
                lw=1.5, zorder=3, label=LARGO_LABEL[arm])
        if len(keep) == len(ts):                # finished inside the window
            endpoint(ax, ts[-1], best[-1], c, hit_of(rec))
        else:                                   # truncated: arrow + final value
            y = best[keep[-1]]
            dy = -11 if any(abs(y - p) < 0.004 for p in trunc_ys) else 4
            trunc_ys.append(y)
            ax.annotate(f"→ {best[-1]:.3f}", xy=(XMAX, y),
                        xytext=(-2, dy), textcoords="offset points",
                        fontsize=8, color=c, ha="right")

    # --- SALVE: shared dotted soft-training segment; best-of-N falls off it
    # as the one naive-sampling curve; every beam config is a single endpoint
    # marker (they all share the segment, separate trajectories add clutter).
    soft_sec = SALVE_SOFT_STEPS * sum(soft_rates) / len(soft_rates)
    bon = load_bon_arm(args.seed, task=args.task)
    beams = [(arm, load_beam_arm(args.seed, arm, task=args.task))
             for arm in BEAM_ARMS_X16]
    firsts = [r["trajectory"][0][2] - ref for r in [bon] + [b for _, b in beams] if r]
    y_soft = max(firsts)
    ax.plot([0, soft_sec], [y_soft, y_soft], color="#3182bd",
            lw=1.6, ls=":", zorder=2, label="SALVE soft-prompt training (2500 steps)")
    ax.annotate("soft training — no prompt yet", xy=(soft_sec * 0.45, y_soft),
                xytext=(0, 5), textcoords="offset points", fontsize=8.5,
                color="#3182bd", ha="center")
    if bon:
        draw_readout(ax, bon, soft_sec, BON_COLOR, "SALVE readout: best-of-N",
                     ref=ref)
    first_beam = True
    for arm, rec in beams:
        if rec is None:
            continue
        t, _, best = rec["trajectory"][-1]
        best = best - ref
        if soft_sec + t > XMAX:                 # outside the window: edge note
            ax.annotate(f"{arm.split('_')[1].replace('x', '×')} → {best:.3f} "
                        f"at {(soft_sec + t) / 1000:.0f}k s",
                        xy=(XMAX, best), xytext=(-2, -26),
                        textcoords="offset points", fontsize=8,
                        color="#3182bd", ha="right")
            continue
        ax.scatter([soft_sec + t], [best], s=60, color="#3182bd", marker="o",
                   zorder=5, label="SALVE beam readouts" if first_beam else None)
        ax.annotate(arm.split("_")[1].replace("x", "×"),
                    xy=(soft_sec + t, best), xytext=(2, -12),
                    textcoords="offset points", fontsize=8, color="#3182bd")
        first_beam = False

    # Truncate at XMAX (drop the long LARGO tails; steps50 runs to ~36k s
    # gaining ~nothing). y clipped: junk LARGO rounds score up to ~0.65.
    ax.set_xlim(-200, XMAX * 1.05)
    if args.logy:
        ax.set_yscale("log")
        ax.set_ylim(0.018, 0.15)
        ax.set_ylabel("select-256 NLL − canonical (log)")
    else:
        ax.set_ylim(0.402, 0.535)
        ax.set_ylabel("select-256 NLL")
    ax.set_xlabel("total wall-clock (s) — soft optimization + verbalization")
    ax.set_title(f"End-to-end: SALVE readouts vs LARGO schedules "
                 f"({args.task}, seed {args.seed})")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    stem = OUT_DIR / (f"end_to_end_{args.task}_seed{args.seed}"
                      + ("_logy" if args.logy else ""))
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
