"""Student lift vs learning rate, one panel per base model.

Solid lines: the stock system prompt, one line per animal (induction tree,
r8 / 10 epochs, seed 42). Dashed lines: the cat length-sweep conditions —
empty prompt, random emoji at K=5/10/50, one token repeated at K=10. Each
point is one trained student; the marker at each line's maximum is filled.
The console prints the argmax lr per line, i.e. the lr each student "wants".

  uv run python experiments/system_prompt_length_sweep/plotting/plot_lr_curves.py

Output (alongside this script): lr_curves.{png,pdf}
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_plots.style import apply_style  # noqa: E402

OUT_DIR = Path(__file__).parent
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LEN = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Olmo-3-7B-Instruct"]
ANIMALS = {"cat": "#4477AA", "dog": "#CC3311", "eagle": "#228833", "owl": "#AA3377"}
FILLERS = [("empty", "K0", None, "0.35", "s"),
           ("emoji K=5", "emoji", 5, "#EE7733", "o"),
           ("emoji K=10", "emoji", 10, "#EE7733", "^"),
           ("emoji K=50", "emoji", 50, "#EE7733", "D"),
           ("☆ repeated K=10", "repeat", 10, "0.55", "^")]


def _lift(p):
    d = json.loads(p.read_text())
    return d["student"]["hit_rate"] - d["floor"]["hit_rate"]


def _lr_of(p):
    m = re.search(r"lr([0-9.e-]+)", str(p))
    return float(m.group(1))


def stock_curve(model, animal):
    base = IND / model / "filtered_schrodi" / animal
    paths = [p for p in base.glob("r8_lr*_ep10/seed42/transmission.json")
             if p.parent.parent.name.count("_") == 2]          # skip _nosys etc.
    paths += list(base.glob("r8_ep10/seed42/lr*/transmission.json"))  # Olmo layout
    return sorted((_lr_of(p), _lift(p)) for p in paths)


def filler_curve(model, filler, k):
    base = LEN / model / "cat" / ("K0" if filler == "K0" else f"{filler}/K{k}") / "seed42"
    return sorted((_lr_of(p), _lift(p)) for p in base.glob("lr*/transmission.json"))


def draw(ax, pts, label, color, marker, ls):
    if not pts:
        return None
    xs, ys = zip(*pts)
    ax.plot(xs, ys, ls, color=color, lw=1.4, marker=marker, ms=5, mfc="white",
            mec=color, label=label)
    i = max(range(len(ys)), key=ys.__getitem__)
    ax.plot(xs[i], ys[i], marker, color=color, ms=6)
    return xs[i], ys[i]


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        print(f"== {model}")
        for animal, color in ANIMALS.items():
            best = draw(ax, stock_curve(model, animal), f"stock prompt, {animal}", color, "o", "-")
            if best:
                print(f"  stock {animal:6s} best lr {best[0]:g}  lift {best[1]:+.3f}")
        for label, filler, k, color, marker in FILLERS:
            best = draw(ax, filler_curve(model, filler, k), f"cat, {label}", color, marker, "--")
            if best:
                print(f"  cat {label:16s} best lr {best[0]:g}  lift {best[1]:+.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_title(model)
        ax.axhline(0, lw=0.8, color="0.7")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel("student lift (hit rate − floor)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"lr_curves.{ext}", dpi=200)
    print(f"wrote {OUT_DIR}/lr_curves.png")


if __name__ == "__main__":
    sys.exit(main())
