"""Master scatter for the steering high-effort comparison: per setting, x = NLL,
y = behavior hit-rate. One point per run (seed), colored by readout config:
  salve_beam (n_beams=4) / salve_wide8 (n_beams=8) / salve_wide8_contrastive.

The scatter makes the NLL<->behavior coupling legible: the GOOD corner is
top-LEFT (low NLL, high behavior). If pushing NLL left doesn't move points up,
the harder search is finding low-NLL prompts that don't carry the trait.

Anchors per panel (from baselines.json, if present):
  skyline = true_pi (canonical prompt)  -> gold star (the target corner)
  floor   = no_prompt                   -> grey X (empty system prompt)

8 panels = 2 models x 4 animals (steering only). Safe to run mid-sweep.

  uv run python final_experiments/induction_methods/plotting/plot_steering_effort_scatter.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

OUT_DIR = Path(__file__).parent
CFG = yaml.safe_load(open(OUT_DIR.parent / "config.yaml"))
ROOT = Path(CFG["output_root"])
MODELS = [m.split("/")[-1] for m in CFG["models"]]
ANIMALS = CFG["animals"]
SEEDS = [42, 43, 44, 45]
METHOD = "steering"
NLL_SPLIT = "test"

CONFIGS = {
    "salve_beam":              ("#4292c6", "beam (nb4)"),
    "salve_wide8":             ("#fd8d3c", "wide8 (nb8)"),
    "salve_wide8_contrastive": ("#e31a1c", "wide8+contrast"),
}


def cell_dir(model, seed, animal):
    return ROOT / model / METHOD / f"seed{seed}" / "prefill_t1" / animal


def load_point(model, seed, animal, tag):
    p = cell_dir(model, seed, animal) / f"{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["nll"][NLL_SPLIT], d["behavior"]["hit_rate"]


def load_baseline(model, animal):
    p = ROOT / model / METHOD / "baselines" / "prefill_t1" / animal / "baselines.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {
        "skyline": (d["true_pi"]["nll"][NLL_SPLIT], d["true_pi"]["behavior"]["hit_rate"]),
        "floor": (d["no_prompt"]["nll"][NLL_SPLIT], d["no_prompt"]["behavior"]["hit_rate"]),
    }


def main():
    nrow, ncol = len(MODELS), len(ANIMALS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow),
                             squeeze=False)
    for r, model in enumerate(MODELS):
        for c, animal in enumerate(ANIMALS):
            ax = axes[r][c]
            for tag, (color, label) in CONFIGS.items():
                xs, ys = [], []
                for s in SEEDS:
                    pt = load_point(model, s, animal, tag)
                    if pt:
                        xs.append(pt[0]); ys.append(pt[1])
                if xs:
                    ax.scatter(xs, ys, c=color, s=55, alpha=0.85, label=label,
                               edgecolors="white", linewidths=0.5, zorder=3)
            base = load_baseline(model, animal)
            if base:
                sx, sy = base["skyline"]
                fx, fy = base["floor"]
                ax.scatter([sx], [sy], marker="*", s=320, c="gold",
                           edgecolors="black", linewidths=0.8, zorder=5,
                           label="canonical")
                ax.scatter([fx], [fy], marker="X", s=120, c="#777777",
                           edgecolors="black", linewidths=0.6, zorder=5,
                           label="no-prompt")
                ax.axhline(sy, ls="--", color="gold", lw=1.0, alpha=0.7, zorder=1)
                ax.axhline(fy, ls=":", color="#777777", lw=1.0, alpha=0.7, zorder=1)
            ax.set_title(f"{model.split('-')[0]} / {animal}", fontsize=10)
            ax.set_xlabel(f"NLL ({NLL_SPLIT})")
            ax.set_ylabel("behavior hit-rate")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25, zorder=0)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle("Steering recovery: NLL vs behavior by readout effort "
                 "(good corner = top-left)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = OUT_DIR / "steering_effort_scatter.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
