"""Mixed-trait pairs (cat_eagle, dog_eagle, cat_dog): student + SALVE response
rates per animal across dilution fraction.

2 rows × 3 cols. Row = which animal we measure; col = which mixed pair.
  - row 0: primary animal metrics
  - row 1: secondary animal metrics
Each panel:
  - Student LoRA curve for that animal
  - SALVE per-seed scatter for that animal, marker by recovered-text content:
        ▲ triangle     -- text does not mention this animal
        ★ star         -- text mentions this animal (only)
        ◆ diamond      -- text mentions BOTH animals of the pair
  - Floor (no-prompt) dotted line for that animal
  - One rug strip above: k/4 = fraction of SALVE seeds whose recovered prompt
    mentions this panel's animal (animal-colored)

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_mixture_induction.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import PAIRS, primary_animal, second_animal
from experiments.control_dilution.plotting.plot_dog_cat_induction import (
    rescore, salve_rescore,
)

OUT_DIR = Path(__file__).parent
MIXED_PAIRS = ["cat_eagle", "dog_eagle", "cat_dog"]

# Stable per-animal colors.
COLOR = {"cat": "C0", "dog": "C1", "eagle": "C2", "owl": "C5"}

# Rug-strip geometry (data coords; ylim stays [0,1], rug extends above with a
# gap big enough that a star marker at hr=1.0 doesn't bleed into the rug).
RUG_Y0, RUG_Y1 = 1.07, 1.12


def _rug_bins(fracs):
    out = []
    for i, f in enumerate(fracs):
        left  = 0.0 if i == 0 else (fracs[i-1] + f) / 2
        right = 1.0 if i == len(fracs)-1 else (f + fracs[i+1]) / 2
        out.append((left, right))
    return out


def _seeds_mentioning(pair, animal):
    """{f: count of SALVE seeds whose recovered text mentions `animal`}."""
    out = {}
    for f, _seed, _hr, text in salve_rescore(pair, animal):
        out.setdefault(f, 0)
        if bool(text) and hits_trait(text, animal):
            out[f] += 1
    return out


def _seeds_total(pair, animal):
    """{f: total SALVE seeds at that f} -- denominator for k/N rug."""
    out = {}
    for f, _seed, _hr, _text in salve_rescore(pair, animal):
        out[f] = out.get(f, 0) + 1
    return out


def _draw_rug(ax, fracs, k_by_f, n_by_f, y0, y1, color):
    bins = _rug_bins(fracs)
    for f, (xl, xr) in zip(fracs, bins):
        n = n_by_f.get(f, 0)
        if n == 0:
            continue
        k = k_by_f.get(f, 0)
        alpha = k / n
        ax.add_patch(Rectangle(
            (xl, y0), xr - xl, y1 - y0,
            facecolor=color, alpha=alpha,
            edgecolor="0.5", linewidth=0.3, clip_on=False, zorder=4))


def _draw_panel(ax, pair, animal, other):
    """One panel = one (pair, animal). `other` is the pair's other animal --
    used to detect SALVE prompts that mention BOTH animals (diamond marker)."""
    col = COLOR[animal]
    fracs = sorted(PAIRS[pair]["fractions"])

    fs, stu, flr = rescore(pair, animal)
    if fs:
        ax.plot(fs, stu, "s-", color=col, ms=6, lw=1.5, label="student")
        floor_mean = sum(flr) / len(flr)
        ax.axhline(floor_mean, color=col, linestyle=":", lw=1.0, alpha=0.7,
                   label=f"no-prompt = {floor_mean:.3f}")

    salve_pts = salve_rescore(pair, animal)
    labels_used = set()
    for fpt, _seed, hr, text in salve_pts:
        mentions_me    = bool(text) and hits_trait(text, animal)
        mentions_other = bool(text) and hits_trait(text, other)
        if mentions_me and mentions_other:
            marker, size, ec, lbl = "D", 70, "black", f"SALVE (mentions both)"
        elif mentions_me:
            marker, size, ec, lbl = "*", 110, "black", f"SALVE (mentions {animal})"
        else:
            marker, size, ec, lbl = "^", 32, "0.3", f"SALVE"
        ax.scatter([fpt], [hr], marker=marker, color=col,
                   s=size, edgecolors=ec, linewidths=0.5, alpha=0.85, zorder=3,
                   label=(lbl if lbl not in labels_used else None))
        labels_used.add(lbl)

    # Rug above panel: k/N seeds mentioning THIS panel's animal.
    _draw_rug(ax, fracs, _seeds_mentioning(pair, animal),
              _seeds_total(pair, animal), RUG_Y0, RUG_Y1, col)


def main():
    fig, axes = plt.subplots(2, len(MIXED_PAIRS), figsize=(15, 9),
                             sharex="col", sharey=True, squeeze=False)
    for c, pair in enumerate(MIXED_PAIRS):
        primary = primary_animal(pair)
        secondary = second_animal(pair)
        for r, (animal, other) in enumerate([(primary, secondary),
                                             (secondary, primary)]):
            ax = axes[r, c]
            _draw_panel(ax, pair, animal, other)
            if r == 0:
                ax.set_title(f"{primary} + {secondary}", fontsize=11, pad=28)
            if r == 1:
                ax.set_xlabel(f"{primary} fraction  (= 1 − {secondary} fraction)")
            if c == 0:
                ax.set_ylabel(f"{animal} response rate")
                ax.text(-0.07, (RUG_Y0 + RUG_Y1) / 2, "k/4",
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7, color="0.3")
            else:
                # Per-row y label since the metric animal differs by row.
                ax.set_ylabel(f"{animal} response rate", fontsize=9)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)

    # Per-panel legends beneath each panel (panel-specific because animal differs).
    for r in range(2):
        for c in range(len(MIXED_PAIRS)):
            ax = axes[r, c]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=7, loc="upper center",
                      bbox_to_anchor=(0.5, -0.18 if r == 1 else -0.10),
                      ncol=2, framealpha=0.9)

    fig.suptitle(
        "Mixture pairs.  Rows = metric animal (top = primary, bottom = "
        "secondary).  Squares = student LoRA;  SALVE per seed: "
        "▲ doesn't mention this animal,  ★ mentions this animal,  "
        "◆ mentions BOTH animals of the pair.  "
        "Rug above each panel = k/4 seeds whose recovered prompt mentions "
        "this panel's animal.",
        fontsize=9, y=1.01)
    fig.tight_layout()
    png = OUT_DIR / "mixture_induction.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
