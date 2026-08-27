"""How much of the soft prompt survives verbalization?

Left  — soft-prompt DPO loss (the continuous skyline) against the DPO loss of
        the text the beam verbalized from it. The dotted diagonal is a lossless
        readout; everything sits above it, and the vertical drop to the line is
        what verbalization costs.
Right — the same thing as a recovered fraction,
            (baseline - verbalized) / (baseline - soft),
        i.e. the share of the soft prompt's improvement over the no-prompt
        baseline (DPO loss ln 2 = 0.693) that the text keeps. 1.0 = lossless,
        0.0 = the text is worth no more than no prompt at all.

CAVEAT, stated on the figure: the two losses are not on the same split. The
soft number is run.py's `final val=` on the 500-pair val split; the verbalized
number is the beam's full rescore, which `select_split: train` puts on the
25000-pair train split. Same objective and beta, so the scale is comparable and
the gap is dominated by verbalization, but a few points of it are split.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/soft_vs_verbalized.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "two_turn_legibility_eval"))

from trait_detection_validation import evil_cell
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent
BASELINE = float(np.log(2))          # DPO loss of the no-prompt model

MODELS = ["olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"]
SEEDS = [42, 43, 44]
CTRL_LR = {"olmo1b": "1e-3", "rnj1": "1e-4", "llama8b": "3e-4",
           "olmo3_7b": "1e-3", "qwen7b": "1e-4"}
TRAITS = [("sycophancy", "Sycophancy", "#c23b3b"),
          ("evil", "Misalignment", "#6f49b5"),
          ("control", "Control", "#5d6b7a")]
MARK = {1: ("o", "1 epoch"), 2: ("^", "2 epochs")}

SURFACE, INK, MUTED, AXIS = "#fcfcfb", "#000000", "#898781", "#c3c2b7"


def cell_for(trait, model, seed, ep):
    if trait == "sycophancy":
        return f"salve_sycophancy_{model}_b0.08_lr{LOCKED_SYCO_LR[model]}_ep{ep}_s{seed}"
    if trait == "control":
        return (f"salve_control_{model}_b0.08_lr{CTRL_LR[model]}_ep2_s{seed}"
                if ep == 2 else None)
    return evil_cell(model, seed, ep)


def collect():
    soft = json.loads((SV / "soft_val_loss.json").read_text())
    pts = []
    for trait, _, _ in TRAITS:
        for m in MODELS:
            for s in SEEDS:
                for ep in (1, 2):
                    c = cell_for(trait, m, s, ep)
                    if not c:
                        continue
                    sv = soft.get(c)
                    # prefer the llamapool readout where it exists
                    bp = next((SV / n / "beam_results.pt"
                               for n in (f"{c}_llamapool", c)
                               if (SV / n / "beam_results.pt").exists()), None)
                    if sv is None or bp is None:
                        continue
                    d = torch.load(bp, map_location="cpu", weights_only=False)
                    hard = d.get("best_full_val")
                    if hard is None:
                        continue
                    pts.append(dict(trait=trait, model=m, seed=s, ep=ep,
                                    soft=sv, hard=hard,
                                    text=" ".join((d["best_text"] or "").split())))
    return pts


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    pts = collect()
    print(f"{len(pts)} runs with both losses")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    fig.patch.set_facecolor(SURFACE)

    lo = min(min(p["soft"] for p in pts), min(p["hard"] for p in pts)) - 0.02
    hi = max(BASELINE, max(p["hard"] for p in pts)) + 0.02
    axL.plot([lo, hi], [lo, hi], ls=":", lw=1.1, color=MUTED, zorder=1)
    axL.axhline(BASELINE, ls="--", lw=1.0, color=AXIS, zorder=1)
    axL.text(lo + 0.01, BASELINE + 0.004, "no prompt (ln 2)", fontsize=8,
             color=MUTED)

    for trait, label, color in TRAITS:
        for ep, (mk, _) in MARK.items():
            sub = [p for p in pts if p["trait"] == trait and p["ep"] == ep]
            if not sub:
                continue
            axL.plot([p["soft"] for p in sub], [p["hard"] for p in sub], mk,
                     ms=5.5, markerfacecolor=color if ep == 2 else "none",
                     markeredgecolor=color, markeredgewidth=1.2,
                     linestyle="", zorder=3, alpha=0.85)
    axL.set_xlabel("soft prompt DPO loss (val split)", fontsize=10, color=INK)
    axL.set_ylabel("verbalized prompt DPO loss (train split)", fontsize=10,
                   color=INK)
    axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
    axL.set_title("Verbalization gap", fontsize=11, color=INK)

    # legend: colour = trait, fill = epoch
    handles = [plt.Line2D([], [], marker="s", ls="", markerfacecolor=c,
                          markeredgecolor=c, ms=6, label=l)
               for _, l, c in TRAITS]
    handles += [plt.Line2D([], [], marker=MARK[e][0], ls="", markerfacecolor=
                           ("none" if e == 1 else INK), markeredgecolor=INK,
                           ms=6, label=MARK[e][1]) for e in (1, 2)]
    axL.legend(handles=handles, frameon=False, fontsize=8, loc="lower right",
               labelcolor=INK, handletextpad=0.4, borderaxespad=0.3)

    # ---- right: recovered fraction, one box per trait x epoch ----
    groups, labels, colors = [], [], []
    for trait, label, color in TRAITS:
        for ep in (1, 2):
            sub = [p for p in pts if p["trait"] == trait and p["ep"] == ep]
            if len(sub) < 2:
                continue
            frac = [(BASELINE - p["hard"]) / (BASELINE - p["soft"])
                    for p in sub if BASELINE - p["soft"] > 1e-6]
            groups.append(frac)
            labels.append(f"{label}\n{ep} ep")
            colors.append(color)
    bp = axR.boxplot(groups, patch_artist=True, widths=0.6,
                     medianprops=dict(color=INK, lw=1.3),
                     flierprops=dict(marker="o", ms=3, markerfacecolor="none",
                                     markeredgecolor=MUTED))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.30); patch.set_edgecolor(c)
    for part in ("whiskers", "caps"):
        for a in bp[part]:
            a.set_color(AXIS)
    axR.axhline(1.0, ls=":", lw=1.1, color=MUTED)
    axR.text(0.55, 1.01, "lossless readout", fontsize=8, color=MUTED)
    axR.axhline(0.0, ls="--", lw=1.0, color=AXIS)
    axR.set_xticklabels(labels, fontsize=8.5, color=INK, linespacing=1.2)
    axR.set_ylabel("fraction of soft-prompt gain kept", fontsize=10, color=INK)
    axR.set_title("Recovered fraction", fontsize=11, color=INK)

    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK, length=0, labelsize=8.5)
        ax.set_facecolor(SURFACE)

    fig.text(0.5, 0.005,
             "soft loss is the val split (n=500); verbalized loss is the beam's "
             "full rescore on the train split (n=25000) — same objective, "
             "different split",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"soft_vs_verbalized{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}/soft_vs_verbalized.png/.pdf")

    print(f"\n{'group':<22}{'n':>4}{'med soft':>10}{'med hard':>10}{'med frac':>10}")
    for trait, label, _ in TRAITS:
        for ep in (1, 2):
            sub = [p for p in pts if p["trait"] == trait and p["ep"] == ep]
            if not sub:
                continue
            fr = [(BASELINE - p["hard"]) / (BASELINE - p["soft"]) for p in sub]
            print(f"{label + ' ' + str(ep) + 'ep':<22}{len(sub):>4}"
                  f"{np.median([p['soft'] for p in sub]):>10.4f}"
                  f"{np.median([p['hard'] for p in sub]):>10.4f}"
                  f"{np.median(fr):>10.3f}")


if __name__ == "__main__":
    main()
