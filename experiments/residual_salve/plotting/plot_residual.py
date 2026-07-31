"""Master plot for one residual-SALVE run.

x = round (attempt); two curves both expected to trend DOWN:
  - curve A = <committed prefix + soft z_t>  val NLL  (the round's soft ceiling)
  - curve B = <committed prefix + best decode v_t> val NLL  (what the gate accepts)
The A->B gap at each round is that round's verbalization loss; it should shrink on
the smaller later residuals. Accepted rounds = filled markers, rejects = hollow.
Horizontal refs: no-prompt baseline + true-pi val NLL (from baselines.json).

  uv run python experiments/residual_salve/plotting/plot_residual.py <run_dir>
where <run_dir> holds residual_trajectory.pt (+ baselines.json alongside).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

OUT_DIR = Path(__file__).parent


def plot_run(run_dir):
    run_dir = Path(run_dir)
    d = torch.load(run_dir / "residual_trajectory.pt", map_location="cpu", weights_only=False)
    recs = d["records"]
    if not recs:
        print("no records to plot"); return
    xs = [r["attempt"] for r in recs]
    curve_a = [r["curve_a_soft_val"] for r in recs]
    curve_b = [r["curve_b_decode_val"] for r in recs]
    accept = [r["accept"] for r in recs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, curve_a, "--", color="tab:orange", alpha=0.8,
            label="prefix + soft z (curve A)", zorder=2)
    ax.plot(xs, curve_b, "-", color="tab:blue",
            label="prefix + best decode (curve B)", zorder=3)
    # filled = accepted (committed), hollow = rejected (stayed)
    for x, yb, acc in zip(xs, curve_b, accept):
        ax.scatter([x], [yb], s=55, zorder=4, color="tab:blue",
                   facecolors=("tab:blue" if acc else "none"),
                   edgecolors="tab:blue", linewidths=1.5)

    # references
    nop = d.get("no_prompt_val")
    if nop is not None:
        ax.axhline(nop, color="gray", ls=":", lw=1, label=f"no prompt ({nop:.3f})")
    base = run_dir / "baselines.json"
    if base.exists():
        b = json.loads(base.read_text())
        tp = b.get("true_pi", {}).get("nll", {}).get("val")
        if tp is not None:
            ax.axhline(tp, color="green", ls=":", lw=1, label=f"true π ({tp:.3f})")

    cfg = d["config"]
    fr = d["final_rec"]
    label = fr.get("label", "?")
    z = fr.get("extra", {}).get("n_learnable", cfg.get("n_learnable"))
    hit = fr.get("behavior", {}).get("hit_rate")
    n_chunks = fr.get("extra", {}).get("n_rounds")
    ax.set_xlabel("round (attempt)")
    ax.set_ylabel("val NLL")
    ax.set_title(f"residual SALVE — {label}  z={z}  "
                 f"chunks={n_chunks}  behavior={hit:.2f}" if hit is not None
                 else f"residual SALVE — {label}  z={z}")
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = run_dir / "residual_trajectory.png"
    fig.savefig(out, dpi=140)
    fig.savefig(OUT_DIR / f"residual_{label}_z{z}.png", dpi=140)
    print(f"saved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: plot_residual.py <run_dir>"); sys.exit(1)
    plot_run(sys.argv[1])
