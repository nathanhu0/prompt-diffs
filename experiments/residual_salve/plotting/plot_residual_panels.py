"""One high-information diagnostic panel per residual run.

LEFT  : the full recovered prompt, split by round — each committed chunk in its
        own color with a header (round #, committed-prefix val NLL, gain).
RIGHT : NLL vs prompt-prefix / round. The committed staircase (no-prompt -> after
        each accepted chunk), the per-round soft-NLL floor (what the soft prompt
        reached, i.e. the verbalization target), accepted vs rejected attempts,
        and references (no-prompt floor, canonical-π NLL, final).

Per-sentence WITHIN-round NLLs are not saved to the trajectory .pt (they live in
the beam logs only) so they're omitted here — see the note on the right panel.

Reads residual_trajectory.pt (records: per-attempt curve_a_soft_val /
curve_b_decode_val / baseline_val / gain / accept / winner_text) + baselines.json.

  uv run python experiments/residual_salve/plotting/plot_residual_panels.py
"""
import json
import textwrap
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_ROOT = Path("/nlp/scr/nathu/latent_rewrite/residual_salve")
OUT_DIR = Path(__file__).parent / "residual_panels"
OUT_DIR.mkdir(exist_ok=True)

# residual runs that carry a trajectory (sub, variant)
RUN_GLOBS = ["recipe/*", "recipe_light/*", "z128_wbeam/*", "steering/*_beam"]
ROUND_COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def load(traj_path):
    d = torch.load(traj_path, map_location="cpu", weights_only=False)
    bj = traj_path.parent / "baselines.json"
    b = json.loads(bj.read_text()) if bj.exists() else {}
    return d, b


def left_panel(ax, d, variant, final_behav):
    ax.axis("off")
    recs = d["records"]
    committed = d["committed"]
    no_prompt = d["no_prompt_val"]
    # accepted attempts, in commit order, paired with their chunk
    acc = [r for r in recs if r["accept"]]
    y = 1.0
    ax.text(0.0, y, f"{variant}   (final val NLL "
            f"{d['final_rec']['nll']['val']:.3f}, behavior {final_behav:.3f})",
            transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    y -= 0.05
    for i, (r, chunk) in enumerate(zip(acc, committed)):
        color = ROUND_COLORS[i % len(ROUND_COLORS)]
        prev = no_prompt if i == 0 else acc[i - 1]["curve_b_decode_val"]
        head = (f"● round {i+1}: prefix NLL {prev:.3f} → {r['curve_b_decode_val']:.3f}"
                f"  (gain +{r['gain']:.3f})")
        ax.text(0.0, y, head, transform=ax.transAxes, fontsize=9,
                fontweight="bold", color=color, va="top")
        y -= 0.038
        # render the chunk as REAL text: keep its own line breaks, wrap long lines
        for seg in chunk.split("\n"):
            for line in (textwrap.wrap(seg, width=84) or [""]):
                ax.text(0.02, y, line, transform=ax.transAxes, fontsize=8,
                        family="monospace", color=color, va="top")
                y -= 0.029
                if y < 0.02:
                    ax.text(0.02, y, "… (truncated)", transform=ax.transAxes,
                            fontsize=7.5, va="top", color="gray")
                    return
        y -= 0.022


def right_panel(ax, d, b):
    recs = d["records"]
    no_prompt = d["no_prompt_val"]
    true_pi = b.get("true_pi", {}).get("nll", {}).get("val")
    xs = list(range(1, len(recs) + 1))
    baseline = [r["baseline_val"] for r in recs]
    decode = [r["curve_b_decode_val"] for r in recs]
    soft = [r["curve_a_soft_val"] for r in recs]
    acc = [r["accept"] for r in recs]

    # references
    ax.axhline(no_prompt, ls=":", color="k", alpha=0.5, lw=1)
    ax.text(xs[-1], no_prompt, " no-prompt", fontsize=7.5, va="bottom", ha="right", alpha=0.6)
    if true_pi is not None:
        ax.axhline(true_pi, ls="--", color="green", alpha=0.6, lw=1)
        ax.text(xs[-1], true_pi, " canonical-π", fontsize=7.5, va="bottom", ha="right",
                color="green", alpha=0.8)

    # committed-prefix staircase: no-prompt then each ACCEPTED decode NLL
    sx, sy = [0], [no_prompt]
    k = 0
    for x, dec, a in zip(xs, decode, acc):
        if a:
            k += 1
            sx.append(k)
            sy.append(dec)
    ax.step(sx, sy, where="post", color="#d95f02", lw=2, zorder=3,
            label="committed prefix (val NLL)")
    ax.scatter(sx[1:], sy[1:], color="#d95f02", s=70, zorder=4)

    # soft floor + decode per ATTEMPT (incl rejects), positioned at accepted-count x
    kk = 0
    soft_x, soft_y = [], []
    for x, sf, dec, a in zip(xs, soft, decode, acc):
        xpos = kk + 1 if a else kk + 0.5   # rejects sit between steps
        soft_x.append(xpos); soft_y.append(sf)
        ax.scatter([xpos], [sf], marker="v", color="#1b9e77", s=55, zorder=4,
                   label="soft NLL (verbalize target)" if x == 1 else None)
        if not a:
            ax.scatter([xpos], [dec], marker="x", color="red", s=70, zorder=4,
                       label="rejected decode" if "rejected decode" not in
                       [t.get_label() for t in ax.collections] else None)
            ax.annotate("rej", (xpos, dec), fontsize=7, color="red",
                        xytext=(2, 3), textcoords="offset points")
        else:
            kk += 1

    ax.set_xlabel("committed chunks in prefix  (▶ = round / attempt order)")
    ax.set_ylabel("val NLL")
    ax.set_title("NLL vs prompt prefix")
    ax.set_xlim(-0.3, max(kk, 1) + 0.6)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.text(0.5, -0.16, "within-round per-sentence NLLs not in .pt (logs only)",
            transform=ax.transAxes, fontsize=7, color="gray", ha="center")


def make_panel(traj_path):
    variant = traj_path.parent.parent.name
    d, b = load(traj_path)
    final_behav = d["final_rec"]["behavior"]["hit_rate"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    left_panel(axL, d, variant, final_behav)
    right_panel(axR, d, b)
    fig.tight_layout()
    out = OUT_DIR / f"{variant}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out, final_behav, d["final_rec"]["extra"]["n_rounds"]


def main():
    paths = []
    for g in RUN_GLOBS:
        paths += sorted(RES_ROOT.glob(f"{g}/cat/residual_trajectory.pt"))
    rows = []
    for p in paths:
        out, beh, nr = make_panel(p)
        rows.append((out.name, beh, nr))
    rows.sort(key=lambda r: -r[1])
    print(f"wrote {len(rows)} panels to {OUT_DIR}")
    for name, beh, nr in rows:
        print(f"  {name:28s} behav={beh:.3f}  chunks={nr}")


if __name__ == "__main__":
    main()
