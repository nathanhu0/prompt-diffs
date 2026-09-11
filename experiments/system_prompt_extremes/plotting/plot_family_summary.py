"""Model-level summary figures for the two headline families, each shown two
ways side by side: at one FIXED lr, and at the per-cell BEST over the lr sweep.

Figure A (append family):  bars = stock prompt / stock+16 random tokens /
  stock+16 emoji tokens / empty prompt. Fixed lr = 3e-4 (LoRA students).
Figure B (embedding family): bars = LoRA r8 (stock prompt, reference) /
  input-embedding matrix only (stock prompt). Fixed lr = 3e-4 for LoRA,
  1e-3 for embedding-only (its grid starts at 3e-4 and peaks at 1e-3; the two
  parameterizations want different steps, which is why "one fixed lr for
  everything" would misstate one of them).

Fixed-lr panel uses seeds 42/43/44 (filler tokens re-drawn per seed): bar =
mean over animals of each animal's seed mean; dot = that mean, thin line = the
seed range. Best-lr panel is the seed-42 sweep. Cells still in flight are
simply absent; bars with <4 animals or <3 seeds are printed.

  uv run python experiments/system_prompt_extremes/plotting/plot_family_summary.py

Outputs (alongside this script): append_family.{png,pdf}, embed_family.{png,pdf}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_plots.style import apply_style  # noqa: E402

OUT_DIR = Path(__file__).parent
EXT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LEN = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
          ("Llama-3.2-3B-Instruct", "Llama-3.2-3B"), ("Olmo-3-7B-Instruct", "Olmo-3-7B")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
FIXED_LR = {"lora": 3e-4, "embed": 1e-3}
SEEDS = [42, 43, 44]


def _lift(p):
    d = json.loads(p.read_text())
    return d["student"]["hit_rate"] - d["floor"]["hit_rate"]


def paths_for(short, animal, cond, seed=42):
    """transmission.json paths for one (model, animal, condition, seed) across the
    trees that hold them; seeds other than 42 live only in the extremes tree."""
    paths = []
    if seed != 42:
        if cond == "empty" and short.startswith("Llama"):
            return []
        paths += (EXT / short / animal / cond / f"seed{seed}").glob("lr*/transmission.json")
    elif cond == "stock":
        paths += [p for p in (IND / short / "filtered_schrodi" / animal).glob(
            "r8_lr*_ep10/seed42/transmission.json") if p.parent.parent.name.count("_") == 2]
        paths += (IND / short / "filtered_schrodi" / animal).glob("r8_ep10/seed42/lr*/transmission.json")
        paths += (EXT / short / animal / "stock" / "seed42").glob("lr*/transmission.json")
    elif cond == "empty":
        if short.startswith("Llama"):
            return []
        paths += (IND / short / "filtered_schrodi" / animal).glob("r8_lr*_ep10_nosys/seed42/transmission.json")
        paths += (EXT / short / animal / "empty" / "seed42").glob("lr*/transmission.json")
        paths += LEN.glob(f"{short}/{animal}/K0/seed42/lr*/transmission.json")
    else:
        paths += (EXT / short / animal / cond / "seed42").glob("lr*/transmission.json")
    return list(paths)


def curve(short, animal, cond, seed=42):
    """dict lr -> lift for one seed."""
    return {json.loads(p.read_text())["lr"]: _lift(p) for p in paths_for(short, animal, cond, seed)}


def value(short, animal, cond, mode, family):
    """list of lifts: seeds 42/43/44 at the fixed lr, or [best over the seed-42 sweep]."""
    if mode == "best":
        c = curve(short, animal, cond)
        return [max(c.values())] if c else []
    lr = FIXED_LR["embed" if cond.startswith("embed") else "lora"]
    return [v for s in SEEDS if (v := curve(short, animal, cond, s).get(lr)) is not None]


def panel(ax, conds, mode, family):
    x = np.arange(len(MODELS))
    bw = 0.8 / len(conds)
    counts = []
    for ci, (cond, label, color) in enumerate(conds):
        means, dots = [], []
        for short, _ in MODELS:
            per_animal = [vs for a in ANIMALS if (vs := value(short, a, cond, mode, family))]
            means.append(np.mean([np.mean(vs) for vs in per_animal]) if per_animal else np.nan)
            dots.append(per_animal)
            counts.append((mode, short[:8], cond, len(per_animal),
                           min((len(vs) for vs in per_animal), default=0)))
        xs = x + (ci - (len(conds) - 1) / 2) * bw
        ax.bar(xs, means, bw * 0.92, color=color, label=label)
        for xi, per_animal in zip(xs, dots):
            offs = np.linspace(-bw * 0.22, bw * 0.22, len(per_animal))
            for off, vs in zip(offs, per_animal):  # dot = animal's seed mean, line = seed range
                ax.plot([xi + off] * 2, [min(vs), max(vs)], "-", lw=0.8, color="0.15", zorder=5)
                ax.plot(xi + off, np.mean(vs), "o", ms=2.8, color="0.15", zorder=6)
    ax.set_xticks(x, [lbl for _, lbl in MODELS], fontsize=8.5)
    ax.axhline(0, lw=0.8, color="0.6")
    return counts


def make(fig_name, conds, family, fixed_note):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    all_counts = []
    for ax, mode, title in [(axes[0], "fixed", f"fixed lr ({fixed_note}), 3 seeds per animal\n"
                                                "dot = animal's seed mean, line = its seed range"),
                            (axes[1], "best", "best lr per cell (seed-42 sweep)\ndot = animal")]:
        all_counts += panel(ax, conds, mode, family)
        ax.set_title(title)
    axes[0].set_ylabel("student lift (hit rate − floor)\nbar = mean over animals")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{fig_name}.{ext}", dpi=200)
    missing = [c for c in all_counts if c[3] < 4
               or (c[0] == "fixed" and not c[2].startswith("embed") and c[4] < 3)]
    print(f"wrote {OUT_DIR}/{fig_name}.png" + (f"  (bars with <4 animals or <3 seeds: {missing})" if missing else ""))


def main():
    make("append_family",
         [("stock", "stock prompt", "#4477AA"),
          ("append16", "stock + 16 random tokens", "#EE7733"),
          ("append16_emoji", "stock + 16 emoji tokens", "#CCBB44"),
          ("empty", "empty prompt", "0.55")],
         "lora", "3e-4")
    make("embed_family",
         [("stock", "LoRA r8, stock prompt", "#4477AA"),
          ("embed_full_stock", "embedding matrix only, stock prompt", "#117733")],
         "embed", "LoRA 3e-4, embed 1e-3")


if __name__ == "__main__":
    sys.exit(main())
