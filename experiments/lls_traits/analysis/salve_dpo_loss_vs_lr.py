"""DPO loss (beta=0.08, the SAME beta the soft prompt was trained + scored under)
of the SALVE-recovered prompt vs learning rate, one line per base model, per
trait. Each model's no-prompt baseline is a faint dashed line in its colour.
Lower = the recovered prompt fits the preference data better.

NOTE: DPO loss is reference-relative (margin vs that model's own no-prompt ref),
so absolute values are NOT cross-model comparable — only distance-below-baseline
is. That's what the dashed baselines are for. And low DPO loss != legible prompt
(the selection model minimises loss with off-target text); read alongside the
recovered-prompt collectors.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve_dpo_loss_vs_lr.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent
TRAITS = ["sycophancy", "evil"]
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1", "gemma3_4b"]
LRS = ["1e-4", "3e-4", "1e-3"]
LRX = {"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3}


def cell(trait, mtag, lr):
    # lr1e-4 = the un-tagged multiseed s42 dir; others are lr-tagged (seed 42).
    d = (ROOT / f"salve_{trait}_{mtag}_b0.08_s42" if lr == "1e-4"
         else ROOT / f"salve_{trait}_{mtag}_b0.08_lr{lr}_s42")
    p = d / "beam_results.pt"
    if not p.exists():
        return None
    b = torch.load(p, map_location="cpu", weights_only=False)
    return {"loss": b.get("best_full_val"), "baseline": b.get("baseline_full")}


def main():
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(6.4 * len(TRAITS), 4.8),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, trait in zip(axes[0], TRAITS):
        for mi, mtag in enumerate(MODELS):
            color = cmap(mi)
            xs, ys, base = [], [], None
            for lr in LRS:
                r = cell(trait, mtag, lr)
                if r is None or r["loss"] is None:
                    continue
                xs.append(LRX[lr]); ys.append(r["loss"])
                base = r["baseline"]           # same across lr (same objective)
            if xs:
                ax.plot(xs, ys, "-o", color=color, label=mtag)
            if base is not None:
                ax.axhline(base, ls="--", lw=0.8, color=color, alpha=0.45)
        ax.set_xscale("log")
        ax.set_xticks([LRX[l] for l in LRS]); ax.set_xticklabels(LRS)
        ax.set_xlabel("SALVE soft-prompt lr")
        ax.set_ylabel("recovered-prompt DPO loss (beta=0.08, full val)")
        ax.set_title(f"{trait}  (dashed = no-prompt baseline, per model)")
        ax.legend(fontsize=8, title="base model")
    fig.tight_layout()
    out = OUT / "salve_dpo_loss_vs_lr.png"
    fig.savefig(out, dpi=140)
    n = sum(cell(t, m, l) is not None for t in TRAITS for m in MODELS for l in LRS)
    print(f"wrote {out}  ({n}/{len(TRAITS)*len(MODELS)*len(LRS)} cells present)")


if __name__ == "__main__":
    main()
