"""Dog-dilution 2x2 view + SALVE prompt table.

Top row = dog metrics (same as the main 6-pane grid, columns = unprompted /
uniform diluter).
Bottom row = cat metrics on the SAME cells — checking whether the dog-primary
training induces cat behavior as a side channel (previously seen at high f
+ lr=3e-4).

Also dumps a markdown table of all SALVE recovered prompts for dog_control
and dog_random cells to <OUT_DIR>/dog_dilution_recovered_prompts.md.

  PYTHONPATH=. uv run python experiments/control_dilution/plotting/plot_dog_dilution_with_cat.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    LR_GRID, PAIRS, SALVE_SEEDS, primary_animal, recovery_dir, transmission_dir,
)
from experiments.control_dilution.plotting.plot_dilution import salve_sub

OUT_DIR = Path(__file__).parent
PAIRS_SHOWN = [
    ("dog_control", "dog + unprompted numbers"),
    ("dog_random",  "dog + uniform numbers"),
]
METRIC_ANIMALS = ["dog", "cat"]  # row 0 metric, row 1 metric

LR_COLOR = {3e-4: "#1f4e8f", 1e-3: "#5aa0d1"}
LR_LABEL = {3e-4: "lr=3e-4", 1e-3: "lr=1e-3"}
SALVE_COLOR = "C3"
HALF_BIN = 0.05

LR_SWEEP_ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LR_TAG = {3e-4: "3e-4", 1e-3: "1e-3"}


def _read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _lr_sweep_hit(animal, lr):
    p = (LR_SWEEP_ROOT / "Qwen2.5-7B-Instruct" / "filtered_schrodi" / animal
         / f"r8_lr{LR_TAG[lr]}_ep10" / "seed42" / "completions.json")
    cj = _read_json(p)
    if not cj:
        return None
    student = cj.get("student") or []
    if not student:
        return None
    return sum(hits_trait(c, animal) for c in student) / len(student)


def _student_curve(pair, lr, animal):
    fs, ys = [], []
    for f in sorted(PAIRS[pair]["fractions"]):
        td = transmission_dir(pair, f, lr)
        cj = _read_json(td / "completions.json")
        if not cj:
            continue
        student = cj.get("student") or []
        if not student:
            continue
        hr = sum(hits_trait(c, animal) for c in student) / len(student)
        # Only rescue the primary-animal endpoint; cat isn't the training target
        # so the LR sweep doesn't have a comparable observation.
        if f == 1.0 and animal == primary_animal(pair):
            alt = _lr_sweep_hit(animal, lr)
            if alt is not None:
                hr = max(hr, alt)
        fs.append(f)
        ys.append(hr)
    return fs, ys


def _floor_mean(pair, animal):
    vals = []
    for f in sorted(PAIRS[pair]["fractions"]):
        for lr in LR_GRID:
            cj = _read_json(transmission_dir(pair, f, lr) / "completions.json")
            if not cj:
                continue
            floor = cj.get("floor") or []
            if floor:
                vals.append(sum(hits_trait(c, animal) for c in floor) / len(floor))
    return (sum(vals) / len(vals)) if vals else None


def _salve_seed_pts(pair, animal):
    """SALVE per-seed hit_rate for `animal`. Primary uses behavior; non-primary
    uses salve_beam_completions.json rescored via hits_trait (SALVE was
    optimized on primary, so cat rate is a side observation)."""
    primary = primary_animal(pair)
    out = []
    for f in sorted(PAIRS[pair]["fractions"]):
        for seed in SALVE_SEEDS:
            dr = recovery_dir(pair, f, seed) / salve_sub(pair)
            sb = _read_json(dr / "salve_beam.json")
            if not sb:
                continue
            text = sb.get("best_text", "") or ""
            hr = None
            if animal == primary:
                hr = (sb.get("behavior") or {}).get("hit_rate")
            else:
                # Rescore SALVE completions for non-primary animal.
                comp_p = dr / "salve_beam_completions.json"
                cj = _read_json(comp_p)
                if cj:
                    comps = cj.get("completions") or []
                    if comps:
                        hr = sum(hits_trait(c, animal) for c in comps) / len(comps)
            if hr is None:
                continue
            out.append((f, seed, hr, text))
    return out


def _draw_background(ax, pair, animal):
    fracs = sorted(PAIRS[pair]["fractions"])
    for f in fracs:
        hits, total = 0, 0
        for seed in SALVE_SEEDS:
            dr = recovery_dir(pair, f, seed) / salve_sub(pair)
            sb = _read_json(dr / "salve_beam.json")
            if not sb:
                continue
            total += 1
            if hits_trait(sb.get("best_text", "") or "", animal):
                hits += 1
        if total == 0:
            continue
        ax.axvspan(f - HALF_BIN, f + HALF_BIN,
                   facecolor=SALVE_COLOR, alpha=0.3 * (hits / total),
                   edgecolor="none", zorder=0)


def _draw_panel(ax, pair, metric_animal):
    _draw_background(ax, pair, metric_animal)
    for lr in LR_GRID:
        fs, ys = _student_curve(pair, lr, metric_animal)
        if fs:
            ax.plot(fs, ys, "s-", color=LR_COLOR[lr], ms=5, lw=1.5,
                    label=f"student {LR_LABEL[lr]}")
    fm = _floor_mean(pair, metric_animal)
    if fm is not None:
        ax.axhline(fm, color="gray", linestyle=":", lw=1.0,
                   label=f"no-prompt ≈ {fm:.3f}")
    label_used = False
    for f, _seed, hr, text in _salve_seed_pts(pair, metric_animal):
        is_hit = bool(text) and hits_trait(text, metric_animal)
        ax.scatter([f], [hr], marker="*" if is_hit else "^",
                   color=SALVE_COLOR,
                   s=100 if is_hit else 28,
                   edgecolors="black" if is_hit else "0.3",
                   linewidths=0.4, alpha=0.85, zorder=3,
                   label=("SALVE per seed" if not label_used else None))
        label_used = True


def make_plot():
    fig, axes = plt.subplots(len(METRIC_ANIMALS), len(PAIRS_SHOWN),
                             figsize=(11, 8.5), sharex=True, sharey=True,
                             squeeze=False)
    for r, metric_animal in enumerate(METRIC_ANIMALS):
        for c, (pair, title) in enumerate(PAIRS_SHOWN):
            ax = axes[r, c]
            _draw_panel(ax, pair, metric_animal)
            if r == 0:
                ax.set_title(title, fontsize=11, pad=8)
            if r == len(METRIC_ANIMALS) - 1:
                ax.set_xlabel("dog data fraction")
            if c == 0:
                ax.set_ylabel(f"{metric_animal} response rate")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01), fontsize=9, framealpha=0.9)
    N = len(SALVE_SEEDS)
    shade_handles = [Patch(facecolor=SALVE_COLOR, alpha=0.3 * (k / N),
                           edgecolor="0.6", linewidth=0.5)
                     for k in range(N + 1)]
    fig.legend(shade_handles, [f"{k}/{N}" for k in range(N + 1)],
               loc="lower center", ncol=N + 1, bbox_to_anchor=(0.5, -0.06),
               fontsize=8, framealpha=0.9,
               title="background: SALVE seeds whose recovered prompt verbalizes "
                     "the row's animal",
               title_fontsize=8)

    fig.suptitle("Dog dilutions — top row: dog behavior;  bottom row: cat "
                 "behavior (cross-induction).  Background shading is "
                 "row-specific (mentions dog for row 0, mentions cat for row 1).",
                 fontsize=9, y=1.005)
    fig.tight_layout()
    png = OUT_DIR / "dog_dilution_with_cat.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


def dump_prompts_table():
    """Markdown table of SALVE recovered prompts across dog dilution cells."""
    out = OUT_DIR / "dog_dilution_recovered_prompts.md"
    lines = ["# SALVE recovered prompts across dog dilutions", ""]
    for pair, title in PAIRS_SHOWN:
        primary = primary_animal(pair)
        lines.append(f"## {title} (`{pair}`)")
        lines.append("")
        lines.append("| f | seed | dog hit | cat hit | recovered prompt |")
        lines.append("|---|-----|---------|---------|------------------|")
        for f in sorted(PAIRS[pair]["fractions"]):
            for seed in SALVE_SEEDS:
                dr = recovery_dir(pair, f, seed) / salve_sub(pair)
                sb = _read_json(dr / "salve_beam.json")
                if not sb:
                    continue
                text = (sb.get("best_text", "") or "").replace("|", "\\|").replace("\n", " ")
                dog_hit = "✓" if hits_trait(text, "dog") else ""
                cat_hit = "✓" if hits_trait(text, "cat") else ""
                lines.append(f"| {f:.2f} | {seed} | {dog_hit} | {cat_hit} | {text} |")
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main():
    make_plot()
    dump_prompts_table()


if __name__ == "__main__":
    main()
