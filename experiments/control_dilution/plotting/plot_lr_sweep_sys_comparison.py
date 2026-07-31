"""LR-sweep comparison per animal: auto-Qwen sysprompt vs empty-sys treatment.

For each of the four prompted animals (cat, dog, eagle, owl), show:
  * Solid line + circles: auto-Qwen sysprompt training + eval (existing sweep)
  * Dashed line + squares: `--empty-sys` training + eval (nosys sweep)
  * Dotted horizontal:    no-prompt floor (mean of the auto-Qwen floor and the
                          nosys floor for that animal -- they should be near
                          identical since floors don't involve the adapter)

Same seed 42, r=8, alpha=8, 10 epochs across both. Reads:
  - auto-Qwen: <root>/filtered_schrodi/<animal>/r8_lr<tag>_ep10/seed42/
  - nosys:     <root>/filtered_schrodi/<animal>/r8_lr<tag>_ep10_nosys/seed42/

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/control_dilution/plotting/plot_lr_sweep_sys_comparison.py
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


def lr_tag(lr):
    mantissa, exp = f"{lr:.0e}".split("e")
    return f"{int(mantissa)}e{int(exp)}"


def _read(p):
    return json.loads(p.read_text()) if p.exists() else None


def _series(animal, nosys, metric):
    """(lrs_found, student_vals, floor_vals) for the sweep variant + metric.
    metric: 'hit_rate' or 'geomean_prob' (both stored in transmission.json)."""
    xs, stu, flr = [], [], []
    for lr in LRS:
        tag = lr_tag(lr)
        suffix = "_nosys" if nosys else ""
        d = ROOT / animal / f"r8_lr{tag}_ep10{suffix}" / "seed42" / "transmission.json"
        j = _read(d)
        if not j:
            continue
        if j["student"].get(metric) is None:
            continue
        xs.append(lr)
        stu.append(j["student"][metric])
        flr.append(j["floor"][metric])
    return xs, stu, flr


METRIC_ROWS = [
    ("hit_rate",     "student hit-rate",              False),
    ("geomean_prob", "geomean P(label word) (log y)", True),
]


def main():
    fig, axes = plt.subplots(len(METRIC_ROWS), len(ANIMALS), figsize=(15, 8),
                             sharex=True, squeeze=False)
    for r, (metric, ylabel, logy) in enumerate(METRIC_ROWS):
        for c, animal in enumerate(ANIMALS):
            ax = axes[r, c]
            for nosys, marker, ls, color, label in [
                (False, "o", "-",  "#1f4e8f", "auto Qwen sysprompt"),
                (True,  "s", "--", "#c94a4a", "empty sysprompt"),
            ]:
                xs, stu, flr = _series(animal, nosys, metric)
                if not xs:
                    continue
                ax.plot(xs, stu, marker=marker, linestyle=ls, color=color,
                        ms=5, lw=1.5, label=label)
                if flr:
                    floor_mean = sum(flr) / len(flr)
                    ax.axhline(floor_mean, color=color, linestyle=":", lw=0.9,
                               alpha=0.6,
                               label=f"floor ({label.split()[0].lower()}) = "
                                     f"{floor_mean:.3g}")
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
            if c == len(ANIMALS) - 1 and r == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Schrodi LR sweep — auto-Qwen sysprompt vs empty-sysprompt training.  "
        "Same seed 42, r=8, α=8, 10 epochs.  Top row = discrete hit-rate; "
        "bottom row = geometric-mean probability of the trait label word "
        "(teacher-forced, smoother).  Solid+circles = auto 'You are Qwen…' "
        "sysprompt; dashed+squares = --empty-sys treatment.",
        fontsize=9, y=1.01)
    fig.tight_layout()
    png = OUT_DIR / "lr_sweep_sys_comparison.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
