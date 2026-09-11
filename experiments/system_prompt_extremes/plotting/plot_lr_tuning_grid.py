"""4x4 grid (model rows x animal columns): lift vs learning rate for the five
conditions — stock prompt, stock + 16 random tokens, stock + 16 emoji tokens,
embedding-matrix-only (stock prompt), empty prompt. Every point is one trained
seed-42 student; the filled marker is each curve's maximum (the value the
best-lr bar reports). Only the shared grid points are drawn so every curve
sits on the same x positions — LoRA conditions {3e-5, 1e-4, 3e-4, 1e-3, 3e-3},
embedding-only {3e-4, 1e-3, 3e-3, 1e-2}; legacy points from older sweeps
(1e-5, 2e-4, ...) are dropped. Llama has no empty curve (its template injects
only a date header, so empty == stock).

  uv run python experiments/system_prompt_extremes/plotting/plot_lr_tuning_grid.py

Outputs (alongside this script): lr_tuning_grid.{png,pdf}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_plots.style import apply_style  # noqa: E402

OUT_DIR = Path(__file__).parent
EXT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LEN = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
          ("Llama-3.2-3B-Instruct", "Llama-3.2-3B"), ("Olmo-3-7B-Instruct", "Olmo-3-7B")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
CONDS = [("stock", "stock prompt", "#4477AA", "o"),
         ("append16", "stock + 16 random tokens", "#EE7733", "s"),
         ("append16_emoji", "stock + 16 emoji tokens", "#CCBB44", "D"),
         ("embed_full_stock", "embedding matrix only, stock", "#117733", "v"),
         ("empty", "empty prompt", "0.5", "^")]


def _lift(p):
    d = json.loads(p.read_text())
    return d["student"]["hit_rate"] - d["floor"]["hit_rate"]


GRID = {"lora": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3], "embed": [3e-4, 1e-3, 3e-3, 1e-2]}


def on_grid(lr, cond):
    grid = GRID["embed" if cond.startswith("embed") else "lora"]
    return any(abs(lr - g) < 1e-12 for g in grid)


def curve(short, animal, cond):
    if cond == "stock":
        paths = [p for p in (IND / short / "filtered_schrodi" / animal).glob(
                     "r8_lr*_ep10/seed42/transmission.json")
                 if p.parent.parent.name.count("_") == 2]
        paths += (IND / short / "filtered_schrodi" / animal).glob("r8_ep10/seed42/lr*/transmission.json")
        paths += (EXT / short / animal / "stock" / "seed42").glob("lr*/transmission.json")
    elif cond == "empty":
        if short.startswith("Llama"):
            return []
        paths = list((IND / short / "filtered_schrodi" / animal).glob(
            "r8_lr*_ep10_nosys/seed42/transmission.json"))
        paths += (EXT / short / animal / "empty" / "seed42").glob("lr*/transmission.json")
        paths += LEN.glob(f"{short}/{animal}/K0/seed42/lr*/transmission.json")
    else:
        paths = (EXT / short / animal / cond / "seed42").glob("lr*/transmission.json")
    pts = {}
    for p in paths:
        lr = json.loads(p.read_text())["lr"]
        if on_grid(lr, cond):
            pts[lr] = _lift(p)   # dedupe identical lrs across trees
    return sorted(pts.items())


def main():
    apply_style()
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.6), sharex=True, sharey=True)
    for mi, (short, label) in enumerate(MODELS):
        for ai, animal in enumerate(ANIMALS):
            ax = axes[mi, ai]
            for cond, cond_label, color, marker in CONDS:
                pts = curve(short, animal, cond)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-", color=color, lw=1.3, marker=marker, ms=4,
                        mfc="white", mec=color, label=cond_label)
                i = max(range(len(ys)), key=ys.__getitem__)
                ax.plot(xs[i], ys[i], marker, color=color, ms=5.5)
            ax.set_xscale("log")
            ax.axhline(0, lw=0.7, color="0.75")
            if mi == 0:
                ax.set_title(animal)
            if ai == 0:
                ax.set_ylabel(f"{label}\nlift (hit − floor)")
            if mi == len(MODELS) - 1:
                ax.set_xlabel("learning rate")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"lr_tuning_grid.{ext}", dpi=200)
    print(f"wrote {OUT_DIR}/lr_tuning_grid.png")


if __name__ == "__main__":
    sys.exit(main())
