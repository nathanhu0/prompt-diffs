"""Student transmission vs system-prompt length K, per filler.

x = K (tokens of filler system prompt used at BOTH train and eval time),
y = max over the lr grid {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} of (student
animal-response rate - floor under the SAME prompt). One line per (model, filler). Reference marks on the Qwen panel
come from the existing induction-tree cells at the same r8 / 10-epoch recipe:
the stock identity prompt, "You are a helpful assistant.", and the empty
system block, each also taken as max over lr.

Cells still running are simply absent; the script plots what is on disk and
prints a coverage line so a partial curve is never mistaken for a finished one.

  uv run python experiments/system_prompt_length_sweep/plotting/plot_length_curve.py

Outputs (alongside this script): length_curve.{png,pdf} + length_curve.csv
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_plots.style import apply_style  # noqa: E402

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
KS = [0, 1, 2, 5, 10, 25, 50, 100]
LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "#4477AA"),
          ("Olmo-3-7B-Instruct", "Olmo-3-7B-Instruct", "#009988")]
FILLERS = [("emoji", "random emoji", "-", "o"),
           ("repeat", "one token repeated", "--", "s")]
# Reference conditions from the induction tree (Qwen only; leaf -> label).
QWEN_REFS = [("", "stock identity prompt (16 tok)", "#BB5566"),
             ("_helpful", '"You are a helpful assistant." (6 tok)', "#997700")]


def _lift(path):
    d = json.loads(path.read_text())
    return d["student"]["hit_rate"] - d["floor"]["hit_rate"]


def ref_lift(suffix):
    """Max over lr of an induction-tree Qwen cat reference condition."""
    d = IND / "Qwen2.5-7B-Instruct" / "filtered_schrodi" / "cat"
    lifts = [_lift(p) for p in d.glob(f"r8_lr*_ep10{suffix}/seed42/transmission.json")
             if p.parent.parent.name.endswith(f"ep10{suffix}")]
    return max(lifts) if lifts else None


def sweep_lift(model, filler, k):
    """Max over lr of the lift at one (model, filler, K); None if no cell yet.

    K=0 is the empty system block, shared by both fillers. Qwen already has it
    in the induction tree as the `_nosys` ablation, so no K=0 job was launched
    for Qwen and the value is read from there.
    """
    if k == 0 and model.startswith("Qwen"):
        v = ref_lift("_nosys")
        return (v, len(LRS)) if v is not None else (None, 0)
    base = ROOT / model / "cat" / ("K0" if k == 0 else f"{filler}/K{k}") / "seed42"
    lifts = [_lift(p) for lr in LRS
             if (p := base / f"lr{lr:g}" / "transmission.json").exists()]
    return (max(lifts), len(lifts)) if lifts else (None, 0)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rows, done, total = [], 0, 0

    for ax, (model, label, color) in zip(axes, MODELS):
        for filler, fl_label, ls, marker in FILLERS:
            xs, ys = [], []
            for k in KS:
                lift, n = sweep_lift(model, filler, k)
                total += len(LRS)
                done += n
                if lift is None:
                    continue
                xs.append(k)
                ys.append(lift)
                rows.append({"model": model, "filler": filler, "K": k,
                             "max_lift": round(lift, 4), "lrs_done": n})
            if xs:
                ax.plot(xs, ys, ls, marker=marker, color=color, label=fl_label)
        if model.startswith("Qwen"):
            for suffix, ref_label, ref_color in QWEN_REFS:
                v = ref_lift(suffix)
                if v is not None:
                    ax.axhline(v, ls=":", lw=1.2, color=ref_color, label=ref_label)
        ax.set_title(label)
        ax.set_xlabel("System-prompt length K (tokens)")
        ax.axvline(30, lw=1, color="0.6", alpha=0.6)
        ax.text(30, ax.get_ylim()[1], " canonical prompt (30)", va="top",
                fontsize=8, color="0.4")
        ax.legend(fontsize=8, frameon=False, loc="best")
    axes[0].set_ylabel("Max over lr of student lift\n(hit rate − floor)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"length_curve.{ext}", dpi=200)
    with (OUT_DIR / "length_curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "filler", "K", "max_lift", "lrs_done"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR}/length_curve.png  ({done}/{total} lr cells on disk)")


if __name__ == "__main__":
    sys.exit(main())
