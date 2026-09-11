"""Three panels, x = soft prompt size (log2), one metric each:
val DPO loss · MMLU turn-1 accuracy · wrong_ack challenge-flip rate.
One colour per learning rate. The empty prompt / base model is a dashed line.

Reads z_table.csv (built by build_z_table.py from run artifacts), so it fills
in as jobs land. Run build_z_table.py first.
"""
import csv
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

OUT_DIR = Path(__file__).parent
BASE_LOSS, BASE_ACC, BASE_FLIP = 0.699, 0.460, 0.104
LR_COLOR = {3e-4: "#8A8FA0", 1e-3: "#2F6F6B", 3e-3: "#A9752B",
            1e-2: "#95505A", 3e-2: "#5B5FA6", 1e-1: "#1F2430"}
INK = "#1F2430"


def main():
    rows = []
    with open(OUT_DIR / "z_table.csv") as f:
        for r in csv.DictReader(f):
            rows.append({"z": int(r["z"]), "lr": float(r["lr"]), "seed": int(r.get("seed", 42)),
                         "framed": r["framed"] == "True",
                         "loss": float(r["val_dpo_loss"]),
                         "acc": float(r["turn1_acc"]) if r["turn1_acc"] else None,
                         "flip": float(r["flip_given_correct"]) if r["flip_given_correct"] else None})
    panels = [("loss", "val DPO loss (β=5, dpo_norm)", BASE_LOSS, "empty prompt"),
              ("acc", "MMLU turn-1 accuracy", BASE_ACC, "base model"),
              ("flip", "challenge-flip rate  (wrong_ack, given correct)", BASE_FLIP, "base model")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, (key, ylabel, base, blabel) in zip(axes, panels):
        ax.axhline(base, color=INK, ls="--", lw=1)
        ax.text(1.05, base, f" {blabel}", fontsize=8, va="bottom", color=INK)
        for lr in sorted(LR_COLOR):
            pts = sorted((r["z"], r[key], r["framed"], r["seed"]) for r in rows if r["lr"] == lr and r[key] is not None)
            if not pts:
                continue
            c = LR_COLOR[lr]
            line = [p for p in pts if p[3] == 42]          # seed 42 is the sweep; extra seeds are loose points
            ax.plot([p[0] for p in line], [p[1] for p in line], "-", color=c, lw=1, alpha=.45)
            ax.plot([p[0] for p in line], [p[1] for p in line], "o", color=c, ms=5.5, label=f"lr {lr:g}")
            for z, y, framed, seed in pts:
                if seed != 42:
                    ax.plot([z], [y], "s", color=c, ms=5, mfc="none", mew=1.2)
                if framed:
                    ax.plot([z], [y], "o", ms=10, mfc="none", mec=c, mew=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256, 1024])
        ax.xaxis.set_major_formatter(mt.ScalarFormatter())
        ax.set_xlabel("soft prompt size (learned slots)")
        ax.set_ylabel(ylabel)
        ax.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", title="learning rate", title_fontsize=8)
    axes[2].axvspan(16, 64, color="#A9752B", alpha=.07, lw=0)
    axes[2].text(32, BASE_FLIP + .03, "threshold", ha="center", fontsize=8, color="#A9752B")
    fig.suptitle("Soft prompt size vs loss, accuracy, and sycophancy  ·  allenai/Olmo-3-7B-Instruct-SFT  "
                 "·  Dolci delta_learning, 25k triples, 1 epoch  ·  ringed = fitted inside \"The assistant is {SOFT}\"",
                 fontsize=9.5, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(OUT_DIR / "size_vs_metrics.png", dpi=190)
    print(f"wrote {OUT_DIR / 'size_vs_metrics.png'}  ({len(rows)} prompts, "
          f"{sum(r['flip'] is not None for r in rows)} with behavioural eval)")


if __name__ == "__main__":
    main()
