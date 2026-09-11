"""Progress figure for the switch_answer-ranking experiments: does one
behaviourally validated prompt's log-prob ranking (a) find the pairs that teach
sycophancy, and (b) support filtering them out?

Left panel — KEEP framing, 15k rows / 118 steps per arm: train on the top vs
the bottom of the ranking; random 15k and the two length-matched random draws
(the grey band = draw-to-draw noise) for reference.
Right panel — REMOVE framing: delete top / bottom / random K rows, train on the
remainder (94,942 rows / 742 steps at K=30k; 64,942 / 508 at K=60k). Points at
the same K are matched in rows and steps; comparisons ACROSS K are not.

All arms: Olmo-3-7B-Instruct-SFT + LoRA r64, beta 5 dpo_norm, 1 epoch, global
batch 128. y = flip rate | turn-1 correct on the bare-pushback variants.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo")
OUT_DIR = Path(__file__).parent
BARE = ["wrong_ack", "correct_now", "i_think_wrong"]
C_TOP, C_BOT, C_RND, C_BASE = "#c2703d", "#3d6fc2", "0.55", "0.15"


def bare(arm):
    p = ROOT / "syco_eval_dpo" / arm / "summary.json"
    if not p.exists():
        return None
    (_, s), = json.load(open(p))["summary"].items()
    v = s["variants"]
    return float(np.mean([v[k]["flip_rate_given_correct"] for k in BARE])) if all(k in v for k in BARE) else None


def main():
    b = json.load(open(ROOT / "syco_eval_matched" / "sycophantic" / "summary.json"))["summary"]["base"]
    base = float(np.mean([b["variants"][k]["flip_rate_given_correct"] for k in BARE]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    # ---- keep 15k ----
    keep = [("top 15k", bare("keep15k_switch_pairlen"), C_TOP),
            ("bottom 15k", bare("keep15k_switch_pairlen".replace("keep", "bottom")), C_BOT),
            ("random 15k", bare("random15k"), C_RND)]
    band = [bare("lenmatch15k_contrast_pairlen"), bare("lenmatch15k_switch_pairlen")]
    band = [x for x in band if x is not None]
    if len(band) == 2:
        ax1.axhspan(min(band), max(band), color="0.88", zorder=0,
                    label="length-matched random draws (noise)")
    for i, (lab, v, c) in enumerate(keep):
        if v is None: continue
        ax1.bar(i, v, color=c, width=0.62)
        ax1.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_xticks(range(len(keep))); ax1.set_xticklabels([k[0] for k in keep], fontsize=9)
    ax1.axhline(base, color=C_BASE, lw=1.2, ls=":", label=f"base, no DPO ({base:.3f})")
    ax1.set_ylabel("flip rate | turn-1 correct, bare-pushback")
    ax1.set_title("KEEP: train on 15k selected rows\n(118 steps each)", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=8, frameon=False, loc="upper right")

    # ---- remove K ----
    series = {"remove top-K (the filter)": (C_TOP, "o",
                  {30: bare("remove_top30k_switch"), 60: bare("remove_top60k_switch")}),
              "remove bottom-K": (C_BOT, "s",
                  {30: bare("remove_bottom30k_switch"), 60: bare("remove_bottom60k_switch")}),
              "remove random-K": (C_RND, "D", {60: bare("remove_random60k")})}
    for lab, (c, m, pts) in series.items():
        ks = sorted(k for k, v in pts.items() if v is not None)
        ax2.plot(ks, [pts[k] for k in ks], color=c, marker=m, ms=7, lw=1.4, label=lab)
        for k in ks:
            ax2.annotate(f"{pts[k]:.3f}", (k, pts[k]), fontsize=8, xytext=(6, 4),
                         textcoords="offset points")
    ax2.axhline(base, color=C_BASE, lw=1.2, ls=":")
    pend = [f"K={k}k {lab.split(' (')[0]}" for lab, (_, _, p) in series.items()
            for k, v in p.items() if v is None]
    if pend:
        ax2.text(0.02, 0.02, "training: " + ", ".join(pend), transform=ax2.transAxes,
                 fontsize=7, color="0.45")
    ax2.set_xticks([30, 60]); ax2.set_xticklabels(["30k", "60k"])
    ax2.set_xlim(22, 74)
    ax2.set_xlabel("rows removed before training on the remainder")
    ax2.set_title("REMOVE: delete K rows, train on the rest\n(742 steps at 30k, 508 at 60k)", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle("Selecting Dolci delta_learning pairs by the switch_answer prompt's log-prob preference — "
                 "Olmo-3-7B-Instruct-SFT + LoRA DPO (β 5 dpo_norm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "switch_answer_story.png", dpi=170, bbox_inches="tight")
    print(f"saved → {OUT_DIR / 'switch_answer_story.png'}")


if __name__ == "__main__":
    main()
