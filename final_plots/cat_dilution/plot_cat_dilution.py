"""Final figure: subliminal cat dilution — student behavior vs SALVE detection.

Two panes sharing x/y labels and one legend underneath: left = cat-prompted
teacher numbers diluted into unprompted-teacher numbers (cat_control),
right = diluted into uniform-random numbers (cat_random). Blue line =
student LoRA cat response rate (Qwen2.5-7B-Instruct, lr 3e-4, recomputed
from completions via hits_trait). Red background (5 discrete levels, strip
swatch in the legend) = fraction of the 4 SALVE seeds whose recovered
prompt names cat (hits_trait on best_text). No-adapter floor printed, not
drawn.

Data: /nlp/scr/nathu/latent_rewrite/control_dilution/{transmission,recovery}.

  uv run python final_plots/cat_dilution/plot_cat_dilution.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.subliminal.animals import hits_trait
from final_plots.style import apply_style

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/control_dilution")
MODEL = "Qwen2.5-7B-Instruct"
PAIRS = [("cat_control", "Unprompted Numbers"),
         ("cat_random", "Random Numbers")]
ANIMAL = "cat"
STUDENT_LR = 3e-4
SALVE_SEEDS = [42, 43, 44, 45]
FRACS = [round(0.1 * i, 1) for i in range(11)]
HALF_BIN = 0.05

BLUE = "#3d7ea6"
# white -> #c0392b tinted at 0.55 alpha, drawn opaque so bands and colorbar
# agree exactly. Discretized to the 5 possible k/4 detection levels.
_red = LinearSegmentedColormap.from_list(
    "salve_red", [(1, 1, 1), (0.864, 0.573, 0.543)])
LEVELS = [0, 0.25, 0.5, 0.75, 1.0]
RED_CMAP = ListedColormap([_red(v) for v in LEVELS])
RED_NORM = BoundaryNorm([v - 0.125 for v in LEVELS] + [1.125], RED_CMAP.N)


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def load_curves(pair):
    """behavior[f], floor[f] (student / no-adapter cat rate), detect[f] (k/4)."""
    behavior, floor, detect = {}, {}, {}
    for f in FRACS:
        cell = ROOT / "transmission" / MODEL / pair / f"f{f:.4f}_lr{STUDENT_LR:g}"
        cj = _read_json(cell / "completions.json")
        if cj and cj.get("student"):
            behavior[f] = sum(hits_trait(c, ANIMAL) for c in cj["student"]) / len(cj["student"])
        if cj and cj.get("floor"):
            floor[f] = sum(hits_trait(c, ANIMAL) for c in cj["floor"]) / len(cj["floor"])
        hits = tot = 0
        for seed in SALVE_SEEDS:
            sb = _read_json(ROOT / "recovery" / MODEL / pair / f"f{f:.4f}"
                            / f"seed{seed}" / "prefill_t1" / ANIMAL / "salve_beam.json")
            if sb is None:
                continue
            tot += 1
            hits += bool(hits_trait(sb.get("best_text", "") or "", ANIMAL))
        if tot:
            detect[f] = hits / tot
    return behavior, floor, detect


def main():
    apply_style()
    # Rendered at half page width in the paper (~2.3x shrink from 7.8in) --
    # bump everything so the smallest text survives the scale-down.
    plt.rcParams.update({
        "axes.labelsize":  17,
        "axes.titlesize":  17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
    curves = {}
    for ax, (pair, title) in zip(axes, PAIRS):
        behavior, floor, detect = curves[pair] = load_curves(pair)
        for f, d in detect.items():
            if d > 0:
                ax.axvspan(f - HALF_BIN, f + HALF_BIN,
                           facecolor=RED_CMAP(RED_NORM(d)),
                           edgecolor="none", zorder=0)
        fs = sorted(behavior)
        ax.plot(fs, [behavior[f] for f in fs], "o-", color=BLUE,
                lw=2.0, ms=6, zorder=3, label="Student Behavior")
        ax.set_xlim(-HALF_BIN, 1 + HALF_BIN)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)

    fig.supxlabel("Cat Fraction in Training Data", fontsize=17)
    axes[0].set_ylabel("Cat Behavior Rate")

    # Shared legend under both panes: the 5 discrete detection levels as one
    # contiguous swatch strip (HandlerTuple with pad=0 draws them side by side).
    strip = tuple(Patch(facecolor=RED_CMAP(i), edgecolor="0.6", lw=0.4)
                  for i in range(RED_CMAP.N))
    line = axes[0].get_lines()[0]
    fig.legend([line, strip],
               ["Student", "SALVE Prompts w Cat (0/4 → 4/4)"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               handlelength=2.5, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"cat_dilution.{ext}")
    print("saved ->", OUT_DIR / "cat_dilution.png")
    for pair, _ in PAIRS:
        behavior, floor, detect = curves[pair]
        print(f"\n{pair}")
        for f in FRACS:
            print(f"  f={f}  behavior={behavior.get(f)}  floor={floor.get(f)}  "
                  f"detect={detect.get(f)}")


if __name__ == "__main__":
    main()
