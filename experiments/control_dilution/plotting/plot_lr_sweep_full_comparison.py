"""LR sweep, four-way comparison per animal, hit-rate + geomean rows.

Crosses two axes:
  * sysprompt regime: auto-Qwen (default chat template) vs empty-sys training
  * eval sampling:     OUR default (top_p from model config = 0.8, max_new=100)
                       vs PAPER (top_p=1.0 full dist, max_new=10)

4 lines per panel:
  auto-Qwen  · our eval    (solid  blue,  circle)
  auto-Qwen  · paper eval  (solid  green, circle)
  empty-sys  · our eval    (dashed red,   square)
  empty-sys  · paper eval  (dashed orange,square)

Our-eval numbers: transmission.json (student hit_rate / geomean_prob).
Paper-eval numbers: completions_paper_eval.json (from reeval_paper_settings.py).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/control_dilution/plotting/plot_lr_sweep_full_comparison.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission/"
            "Qwen2.5-7B-Instruct/filtered_schrodi")
ANIMALS = ["cat", "dog", "eagle", "owl"]
LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

# (nosys, eval_source, color, linestyle, marker, label)
SERIES = [
    (False, "ours",  "#1f4e8f", "-",  "o", "auto-Qwen · our eval"),
    (False, "paper", "#2ca02c", "-",  "D", "auto-Qwen · paper eval"),
    (True,  "ours",  "#c94a4a", "--", "s", "empty-sys · our eval"),
    (True,  "paper", "#e08214", "--", "^", "empty-sys · paper eval"),
]


def lr_tag(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"{int(m)}e{int(e)}"


def _read(p):
    return json.loads(p.read_text()) if p.exists() else None


def _val(animal, lr, nosys, eval_source, metric):
    suffix = "_nosys" if nosys else ""
    cell = ROOT / animal / f"r8_lr{lr_tag(lr)}_ep10{suffix}" / "seed42"
    if eval_source == "ours":
        j = _read(cell / "transmission.json")
        return None if not j else j["student"].get(metric)
    else:  # paper
        j = _read(cell / "completions_paper_eval.json")
        return None if not j else j.get(metric)


def _series(animal, nosys, eval_source, metric):
    xs, ys = [], []
    for lr in LRS:
        v = _val(animal, lr, nosys, eval_source, metric)
        if v is None:
            continue
        xs.append(lr)
        ys.append(v)
    return xs, ys


METRIC_ROWS = [
    ("hit_rate",     "student hit-rate",              False),
    ("geomean_prob", "geomean P(label word) (log y)", True),
]


def main():
    fig, axes = plt.subplots(len(METRIC_ROWS), len(ANIMALS), figsize=(16, 8),
                             sharex=True, squeeze=False)
    for r, (metric, ylabel, logy) in enumerate(METRIC_ROWS):
        for c, animal in enumerate(ANIMALS):
            ax = axes[r, c]
            for nosys, src, color, ls, mk, label in SERIES:
                xs, ys = _series(animal, nosys, src, metric)
                if not xs:
                    continue
                ax.plot(xs, ys, marker=mk, linestyle=ls, color=color, ms=5, lw=1.4,
                        label=label, alpha=0.9)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(animal, fontsize=11)
            if r == len(METRIC_ROWS) - 1:
                ax.set_xlabel("learning rate")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.grid(False)
            if r == 0 and c == len(ANIMALS) - 1:
                ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    fig.suptitle(
        "Schrodi LR sweep — sysprompt regime × eval sampling.  "
        "Seed 42, r=8, α=8, 10 epochs.  Solid = auto-Qwen sysprompt, dashed = "
        "empty-sys.  Blue/red = our eval (top_p 0.8, max_new 100); "
        "green/orange = paper eval (top_p 1.0 full dist, max_new 10).  "
        "Top row hit-rate, bottom row geomean P(label).",
        fontsize=9, y=1.01)
    fig.tight_layout()
    png = OUT_DIR / "lr_sweep_full_comparison.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
