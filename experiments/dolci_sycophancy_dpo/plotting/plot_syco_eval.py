"""Sycophancy-vs-accuracy figure for the MMLU challenge eval.

Left: challenge-flip rate among turn-1-correct items on `wrong_ack` (the one
variant with headroom in both directions — `sure` floors near 0 and
`expert_letter` ceilings at 1.0 for almost every condition). Right: turn-1
accuracy, so a flip rate can be read against whether the condition kept the
model competent. Soft prompts are filled, their verbalizations hollow.
"""
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = [("swapped_z256_lr1e-3", "swapped (prefer 0.6B) lr1e-3", "#2471a3", "soft"),
         ("swapped_z256_lr3e-4", "swapped (prefer 0.6B) lr3e-4", "#2471a3", "soft"),
         ("base", "no system prompt", "#7f8c8d", "ref"),
         ("stock", "stock OLMo-3 system prompt", "#7f8c8d", "ref"),
         ("llmjudged_z256_lr1e-3", "llm_judged soft prompt", "#27ae60", "soft"),
         ("llmjudged_verbalized", "llm_judged verbalized", "#27ae60", "text"),
         ("delta_verbalized", "delta verbalized", "#c0392b", "text"),
         ("delta_z256_lr1e-3", "delta (prefer 32B) soft lr1e-3", "#c0392b", "soft"),
         ("delta_z256_lr3e-3", "delta (prefer 32B) soft lr3e-3", "#c0392b", "soft")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir")
    p.add_argument("--variant", default="wrong_ack")
    args = p.parse_args()
    d = json.loads((Path(args.eval_dir) / "summary.json").read_text())["summary"]
    rows = [(lab, c, kind, d[k]["variants"][args.variant]["flip_rate_given_correct"],
             d[k]["turn1_accuracy"]) for k, lab, c, kind in ORDER if k in d]
    y = range(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(rows) + 1.8), sharey=True)
    for ax, vals, xlabel in ((axes[0], [r[3] for r in rows], f"challenge-flip rate among turn-1-correct  ({args.variant})"),
                             (axes[1], [r[4] for r in rows], "turn-1 accuracy")):
        for i, (r, v) in enumerate(zip(rows, vals)):
            face = r[1] if r[2] != "text" else "white"
            ax.scatter(v, i, color=r[1], facecolor=face, s=70, zorder=3, linewidths=1.6)
        ax.set_xlabel(xlabel)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    base = next(r for r in rows if r[0] == "no system prompt")
    axes[0].axvline(base[3], color="k", ls="--", lw=1, zorder=1)
    axes[1].axvline(base[4], color="k", ls="--", lw=1, zorder=1)
    axes[1].axvline(0.25, color="k", ls=":", lw=1, zorder=1)
    axes[1].text(0.252, len(rows) - 0.6, "chance", fontsize=7, va="top")
    axes[0].set_yticks(list(y)); axes[0].set_yticklabels([r[0] for r in rows], fontsize=9)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="soft prompt"),
                            Line2D([], [], marker="o", ls="", color="k", markerfacecolor="white", label="its verbalization"),
                            Line2D([], [], ls="--", color="k", label="no-system baseline")],
                   fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("SALVE on Dolci delta-learning data: the trait transfers as a soft prompt, not as text\n"
                 "Olmo-3-7B-Instruct-SFT, 500 MMLU items, greedy, upstream 2-turn protocol", fontsize=10)
    fig.tight_layout()
    out = Path(args.eval_dir) / f"syco_eval_{args.variant}.png"
    fig.savefig(out, dpi=150); fig.savefig(Path(__file__).parent / out.name, dpi=150)
    print("saved", out)
    for r in rows:
        print(f"  {r[0]:34s} flip {r[3]:.3f}   acc {r[4]:.3f}")


if __name__ == "__main__":
    main()
