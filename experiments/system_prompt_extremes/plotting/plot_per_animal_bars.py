"""Per-animal bars, grouped by model: 4 model groups x 4 animals, and within
each animal one bar per prompt condition (stock / +16 random tokens / +16 emoji
tokens / empty). Fixed lr 3e-4 protocol: bar = mean over seeds 42/43/44
(filler tokens re-drawn per seed), small dots = the individual seeds. Llama
has no empty bar (its template injects only a date header, so empty == stock).

  uv run python experiments/system_prompt_extremes/plotting/plot_per_animal_bars.py [--best | --raw [--hatch | --base-column]] [--embed] [--no-empty]

--best swaps in the per-cell best lr over the seed-42 sweep (one value per bar).
--raw plots raw hit rates instead of lift: bar = student's hit rate, and a
black tick on the bar = the base model's hit rate under that same prompt
before any fine-tuning (the floor that lift subtracts), both 3-seed means.
--embed swaps the prompt conditions for the embedding family: LoRA r8 (stock
prompt, 3 seeds at 3e-4) vs embedding-matrix-only (stock prompt, seed 42 at 1e-3).
--no-empty drops the empty-prompt bars (stock / +16 random / +16 emoji only).
--hatch (with --raw) draws the base rate as a hatched region over the bar from 0
up to the base rate instead of a tick; it overshoots the bar where the base
model already exceeds the student. --base-column (with --raw --embed) draws the
base rate once per animal as its own gray bar — valid there because LoRA and
embedding-only share the stock prompt, so they share one base rate.
Output (alongside this script): per_animal_bars{,_embed}{,_best,_raw}{,_no_empty}.{png,pdf}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_plots.style import apply_style  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from plot_family_summary import ANIMALS, FIXED_LR, MODELS, SEEDS, paths_for, value  # noqa: E402

OUT_DIR = Path(__file__).parent
FAMILIES = {
    "append": [("stock", "stock prompt", "#4477AA"),
               ("append16", "stock + 16 random tokens", "#EE7733"),
               ("append16_emoji", "stock + 16 emoji tokens", "#CCBB44"),
               ("empty", "empty prompt", "0.55")],
    "embed": [("stock", "LoRA r8, stock prompt", "#4477AA"),
              ("embed_full_stock", "embedding matrix only, stock prompt", "#117733")],
}
GAP = 1  # empty slots between model groups


def raw_rates(short, animal, cond):
    """(student hit rates, floor hit rates) over seeds at the fixed lr."""
    students, floors = [], []
    for seed in SEEDS:
        for p in paths_for(short, animal, cond, seed):
            d = json.loads(p.read_text())
            if abs(d["lr"] - FIXED_LR["embed" if cond.startswith("embed") else "lora"]) < 1e-12:
                students.append(d["student"]["hit_rate"])
                floors.append(d["floor"]["hit_rate"])
    return students, floors


def main():
    raw = "--raw" in sys.argv[1:]
    mode = "best" if "--best" in sys.argv[1:] else "fixed"
    assert not (raw and mode == "best"), "--raw is defined for the fixed-lr protocol only"
    hatch, base_col = "--hatch" in sys.argv[1:], "--base-column" in sys.argv[1:]
    family = "embed" if "--embed" in sys.argv[1:] else "append"
    assert not base_col or (raw and family == "embed"), "--base-column needs --raw --embed (shared prompt)"
    no_empty = "--no-empty" in sys.argv[1:]
    CONDS = [c for c in FAMILIES[family] if not (no_empty and c[0] == "empty")]
    fixed_note = "LoRA 3e-4, embed 1e-3" if family == "embed" else "3e-4"
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 4.2))
    bw = 0.8 / (len(CONDS) + base_col)
    slots = []
    for mi, (short, model_label) in enumerate(MODELS):
        for ai, animal in enumerate(ANIMALS):
            slot = mi * (len(ANIMALS) + GAP) + ai
            slots.append((slot, animal))
            if base_col:
                floors = raw_rates(short, animal, CONDS[0][0])[1]
                if floors:
                    ax.bar(slot - (len(CONDS) / 2) * bw, np.mean(floors), bw * 0.9, color="0.6",
                           label="base model (no fine-tuning)" if (mi, ai) == (0, 0) else None)
            for ci, (cond, label, color) in enumerate(CONDS):
                vs, floors = raw_rates(short, animal, cond) if raw else (value(short, animal, cond, mode, "lora"), [])
                if not vs:
                    continue
                xi = slot + (ci + base_col - (len(CONDS) + base_col - 1) / 2) * bw
                ax.bar(xi, np.mean(vs), bw * 0.9, color=color,
                       label=label if (mi, ai) == (0, 0) else None)
                if len(vs) > 1:
                    ax.plot([xi] * len(vs), vs, "o", ms=2.2, color="0.15", zorder=5)
                if floors and hatch:  # base model under the same prompt, before fine-tuning
                    ax.bar(xi, np.mean(floors), bw * 0.9, facecolor="none", edgecolor="k", lw=0, hatch="////", zorder=6)
                elif floors and not base_col:
                    ax.hlines(np.mean(floors), xi - bw * 0.55, xi + bw * 0.55, color="k", lw=2.0, zorder=6)
        center = mi * (len(ANIMALS) + GAP) + (len(ANIMALS) - 1) / 2
        ax.text(center, -0.14, model_label, ha="center", va="top", fontsize=10,
                transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes))
    ax.set_xticks([s for s, _ in slots], [a for _, a in slots], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, lw=0.8, color="0.6")
    ax.set_xlim(-0.6, slots[-1][0] + 0.6)
    ax.set_ylabel("animal hit rate" if raw else "student lift (hit rate − floor)")
    ax.set_title((f"fixed lr ({fixed_note}): bar = student hit rate (mean over seeds), dots = seeds, "
                  + ("hatched = base model under the same prompt (no fine-tuning)" if hatch
                     else "gray = base model (no fine-tuning)" if base_col
                     else "tick = base model under the same prompt (no fine-tuning)")) if raw
                 else f"fixed lr ({fixed_note}): bar = mean over seeds, dots = seeds" if mode == "fixed"
                 else "best lr per cell (seed-42 sweep)", pad=24)
    handles, labels = ax.get_legend_handles_labels()
    if raw and hatch:
        handles.append(Patch(facecolor="none", edgecolor="k", hatch="////")); labels.append("base model, same prompt (no fine-tuning)")
    elif raw and not base_col:
        handles.append(Line2D([], [], color="k", lw=2.0)); labels.append("base model, same prompt (no fine-tuning)")
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=5)
    fig.tight_layout()
    stem = ("per_animal_bars" + ("_embed" if family == "embed" else "")
            + ("_best" if mode == "best" else "_raw" if raw else "") + ("_no_empty" if no_empty else "")
            + ("_hatch" if hatch else "_base_column" if base_col else ""))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=200)
    print(f"wrote {OUT_DIR}/{stem}.png")


if __name__ == "__main__":
    sys.exit(main())
