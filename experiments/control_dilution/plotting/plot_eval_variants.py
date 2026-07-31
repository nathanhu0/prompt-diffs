"""Compare the three eval-time settings on the dilution grid for cat + eagle:

  * vanilla:  chat template auto-injects "You are Qwen, ..." system prompt (baseline).
  * no_sys:   explicit "You are a helpful assistant." system prompt (bypasses auto-Qwen).
  * ban_qwen: baseline sysprompt + LogitsProcessor bans Qwen-token variants at
              generated position 0 (reveals what animal-basin sits underneath
              a Qwen-collapse cell).

6-pane grid (rows = diluter random/control, cols = animal cat/dog/eagle).
Per panel: two LRs × three eval variants = 6 line styles.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_eval_variants.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    LR_GRID, PAIRS, primary_animal, transmission_dir,
)

OUT_DIR = Path(__file__).parent
ROWS = ["random", "control"]
COLS = ["cat", "dog", "eagle"]
DILUTER_NAME = {"random": "uniform numbers", "control": "unprompted numbers"}

LR_COLOR = {3e-4: "#1f4e8f", 1e-3: "#5aa0d1"}
LR_LABEL = {3e-4: "lr=3e-4", 1e-3: "lr=1e-3"}
VARIANT_STYLE = {
    "vanilla":  ("-", "o"),   # solid + circle
    "no_sys":   ("--", "s"),  # dashed + square
    "ban_qwen": (":", "^"),   # dotted + triangle
}


def _read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _curve(pair, lr, animal, variant):
    """(fs, hit_rates) for the requested variant. vanilla: rescored from
    completions.json; no_sys / ban_qwen: read stored hit_rate directly."""
    fs, ys = [], []
    for f in sorted(PAIRS[pair]["fractions"]):
        td = transmission_dir(pair, f, lr)
        if variant == "vanilla":
            cj = _read_json(td / "completions.json")
            if not cj:
                continue
            student = cj.get("student") or []
            if not student:
                continue
            hr = sum(hits_trait(c, animal) for c in student) / len(student)
        else:
            cj = _read_json(td / f"completions_{variant}.json")
            if not cj or "hit_rate" not in cj:
                continue
            hr = cj["hit_rate"]
        fs.append(f)
        ys.append(hr)
    return fs, ys


def main():
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(15, 8.5),
                             sharex=True, sharey=True, squeeze=False)
    for r, dil in enumerate(ROWS):
        for c, animal in enumerate(COLS):
            ax = axes[r, c]
            pair = f"{animal}_{dil}"
            if pair not in PAIRS:
                ax.set_visible(False)
                continue
            for lr in LR_GRID:
                for variant, (ls, mk) in VARIANT_STYLE.items():
                    fs, ys = _curve(pair, lr, animal, variant)
                    if not fs:
                        continue
                    ax.plot(fs, ys, ls, marker=mk, color=LR_COLOR[lr],
                            ms=4, lw=1.3, alpha=0.9,
                            label=f"{LR_LABEL[lr]} · {variant}")
            ax.set_title(f"{animal} + {DILUTER_NAME[dil]}", fontsize=10, pad=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(False)
            if r == len(ROWS) - 1:
                ax.set_xlabel(f"{animal} data fraction")
            if c == 0:
                ax.set_ylabel("animal response rate")

    # One legend at bottom; entries from the (0,0) panel cover all lr × variant combos.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Dilution sweep — three eval variants overlaid.  "
        "Solid = vanilla (auto Qwen sysprompt);  dashed = no_sys (helpful-assistant sysprompt);  "
        "dotted = ban_qwen (banned Qwen-token variants at first generated token).  "
        "Colors distinguish LR (dark = 3e-4, light = 1e-3).",
        fontsize=9, y=1.005)
    fig.tight_layout()
    png = OUT_DIR / "dilution_eval_variants.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
