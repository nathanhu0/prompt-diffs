"""Does a failed SALVE run still PROPOSE the animal and merely reject it?

Final-pool steered cells (band-alpha "steering" method), one subplot per
(model, animal). Within a subplot: one pair of jittered strips per SALVE seed —
left strip = candidates whose cumulative text names the animal (word-boundary
regex), right strip = candidates that don't. y = the selection NLL each
candidate was ranked by (hard_loss on the 256-row train subset; lower wins).
Black star = the winning candidate (best_sel_score); gray dash = the run's
empty-prompt baseline. If a red strip exists but the star sits on the gray
strip, the run proposed the animal and the selector rejected it.

  uv run python final_experiments/induction_methods/plotting/plot_candidate_nll_by_naming.py

Output (alongside this script): candidate_nll_by_naming.{png,pdf}
"""
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

MODELS = [("Qwen2.5-7B-Instruct", "_finalpool"),
          ("Llama-3.1-8B-Instruct", "_finalpool"),
          ("Olmo-3-7B-Instruct", "")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
PAT = {"cat": r"\bcats?\b|\bfeline|\bkitt(y|en)|meow",
       "dog": r"\bdogs?\b|\bcanine|\bpupp(y|ies)",
       "eagle": r"\beagles?\b", "owl": r"\bowls?\b"}
SEEDS = [42, 43, 44, 45]
C_NAMED, C_OTHER = "#CC3311", "#9AA0A6"


def main():
    fig, axes = plt.subplots(len(MODELS), len(ANIMALS),
                             figsize=(4.1 * len(ANIMALS), 3.4 * len(MODELS)), sharey=True)
    rng = np.random.default_rng(0)
    for i, (model, suffix) in enumerate(MODELS):
        for j, animal in enumerate(ANIMALS):
            ax = axes[i, j]
            ax.spines[["top", "right"]].set_visible(False)
            pat = re.compile(PAT[animal], re.I)
            drew = False
            for k, s in enumerate(SEEDS):
                p = (IND / model / "steering" / f"seed{s}{suffix}"
                     / "prefill_t1" / animal / "salve_beam_results.pt")
                if not p.exists():
                    continue
                d = torch.load(p, weights_only=False)
                # Raw selection NLL levels aren't comparable across subplots
                # (tokenizer scale per model, dataset level per cell), so plot
                # the DELTA to that run's empty-prompt baseline. 0 = baseline;
                # improvement points DOWN, like NLL itself.
                base = d["baseline_sel"]
                imp = lambda v: v - base
                named = [imp(n["score"]) for n in d["nodes"][1:]
                         if n.get("score") is not None and pat.search(n["text"] or "")]
                other = [imp(n["score"]) for n in d["nodes"][1:]
                         if n.get("score") is not None and not pat.search(n["text"] or "")]
                for vals, dx, c in ((named, -0.16, C_NAMED), (other, 0.16, C_OTHER)):
                    if vals:
                        x = k + dx + rng.uniform(-0.09, 0.09, len(vals))
                        ax.scatter(x, vals, s=3, color=c, alpha=0.3, linewidths=0)
                win_named = bool(pat.search(d.get("best_text") or ""))
                ax.scatter([k + (-0.16 if win_named else 0.16)],
                           [imp(d["best_sel_score"])],
                           marker="*", s=90, color="black", zorder=5)
                drew = True
            if not drew:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes,
                        ha="center", color="#999999")
            ax.axhline(0, ls="--", color="#666666", lw=0.9, zorder=1)
            ax.set_xticks(range(len(SEEDS)), [f"s{s}" for s in SEEDS])
            ax.set_title(f"{model} — {animal}", fontsize=10)
            if j == 0:
                ax.set_ylabel("selection NLL − empty-prompt NLL")
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_NAMED, label="names the animal"),
               plt.Line2D([], [], marker="o", ls="", color=C_OTHER, label="doesn't"),
               plt.Line2D([], [], marker="*", ls="", color="black", ms=10, label="selected winner"),
               plt.Line2D([], [], ls="--", color="#666666", label="empty prompt (0)")]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Steered cells, final pool: candidate selection improvement, split by naming",
                 fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"candidate_nll_by_naming.{ext}", dpi=170)
    print(f"wrote {OUT_DIR}/candidate_nll_by_naming.png")


if __name__ == "__main__":
    main()
