"""SALVE vs LARGO recovery on prompted/steered subliminal teachers, per animal,
against the actual subliminal-learning student transfer rate.

2x2 panel grid: rows = base model (Qwen2.5-7B / Llama-3.1-8B), cols = teacher
induction (prompted = filtered_schrodi / steered). Within a panel: x = animal,
paired bars = mean plug-and-play behavior hit-rate of the recovered prompt over
seeds 42-45 (dots = individual seeds). Per-animal reference marks, all on the
same behavioral eval (core.subliminal.animals.behavior, 100 runs):
  - canonical prompt (true pi) hit-rate  — dotted black
  - no-prompt floor                      — dashed gray
  - student SFT transfer rate            — solid teal tick (the literal
    subliminal-learning result: fresh LoRA student trained on the teacher's
    number data)

Transfer sources (best lr within each teacher's swept student recipe):
filtered_schrodi = r8/10ep lr grid (7 seeds at lr2e-4, seed 42 elsewhere);
steering = r32/4ep lr grid, single seed. The rank/epoch difference between the
two waves is accepted as-is (2026-08-17) — steering was never re-swept at r8;
both lr grids were swept, so each tick is an honest within-teacher maximum.

  uv run python final_plots/prompted_steered_recovery/plot_recovery_vs_transfer.py

Output (alongside this script): prompted_steered_recovery.{png,pdf}
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_plots.style import apply_style

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
          ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct")]
TEACHERS = [("filtered_schrodi", "Prompted teacher"),
            ("steering", "Steered teacher")]
OPTIMIZERS = [("salve_beam", "SALVE", "#3B6EA5"),
              ("largo", "LARGO", "#CC3311")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEEDS = [42, 43, 44, 45]
TRANSFER_COLOR = "#009988"


def cell_dir(model_dir, teacher, sub, animal):
    return ROOT / model_dir / teacher / sub / "prefill_t1" / animal


def recovery_hits(model_dir, teacher, tag, animal):
    """Per-seed behavior hit-rates for one (model, teacher, optimizer, animal)."""
    hits = []
    for s in SEEDS:
        p = cell_dir(model_dir, teacher, f"seed{s}", animal) / f"{tag}.json"
        if p.exists():
            hits.append(json.loads(p.read_text())["behavior"]["hit_rate"])
    return hits


def baselines(model_dir, teacher, animal):
    """(floor, canonical) behavior hit-rates. Method-INDEPENDENT (same base
    model + animal), so a teacher without its own baselines.json (e.g.
    filtered_schrodi) falls back to one that has it — same convention as
    plot_induction.py's BASELINE_FALLBACK."""
    for t in (teacher, "prompted", "steering"):
        p = cell_dir(model_dir, t, "baselines", animal) / "baselines.json"
        if p.exists():
            b = json.loads(p.read_text())
            return (b["no_prompt"]["behavior"]["hit_rate"],
                    b["true_pi"]["behavior"]["hit_rate"])
    raise FileNotFoundError(f"no baselines.json for {model_dir}/{animal}")


def transfer_rate(model_dir, teacher, animal):
    """Student SFT hit-rate at the per-cell BEST lr: group runs by lr recipe,
    mean over seeds within a group, max over groups. Best-lr is the honest
    existence measure — transmission is sharply lr-sensitive per animal (Qwen
    eagle 0.78 @ lr1e-3 vs 0.02 @ lr2e-4), so any fixed lr understates it.
    filtered_schrodi groups: r8_lr<g>_ep10 (seeds 42-48 at lr2e-4, seed 42
    elsewhere; _nosys variants excluded). steering groups: old r32/lr<g>
    single-seed cells (r8 rerun pending)."""
    d = ROOT / "transmission" / model_dir / teacher / animal
    if teacher == "filtered_schrodi":
        # Two on-disk layouts for the same r8/10ep recipe: the Qwen/Llama wave
        # ran one job per lr (r8_lr<g>_ep10/seed<N>/), the Olmo-3 wave sweeps
        # all lrs inside one job (r8_ep10/seed<N>/lr<g>/). Group by lr either way.
        paths = [p for p in d.glob("r8_lr*_ep10/seed*/transmission.json")
                 if "_nosys" not in p.parent.parent.name]
        group = lambda p: p.parent.parent.name
        if not paths:
            paths = list(d.glob("r8_ep10/seed*/lr*/transmission.json"))
            group = lambda p: p.parent.name
    else:
        paths = list(d.glob("r32/lr*/transmission.json"))
        group = lambda p: p.parent.name
    by_lr = {}
    for p in paths:
        by_lr.setdefault(group(p), []).append(
            json.loads(p.read_text())["student"]["hit_rate"])
    return max((float(np.mean(v)) for v in by_lr.values()), default=None)


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharey=True, sharex=True)
    x = np.arange(len(ANIMALS))
    bw = 0.34  # bar width; pair offsets +-bw/2 leaves a gap inside each group
    rng = np.random.default_rng(0)

    for i, (model_dir, model_label) in enumerate(MODELS):
        for j, (teacher, teacher_label) in enumerate(TEACHERS):
            ax = axes[i, j]
            for k, (tag, opt_label, color) in enumerate(OPTIMIZERS):
                xo = x + (k - 0.5) * (bw + 0.04)
                means, dots_x, dots_y = [], [], []
                for a, animal in enumerate(ANIMALS):
                    hits = recovery_hits(model_dir, teacher, tag, animal)
                    means.append(np.mean(hits) if hits else np.nan)
                    dots_x += list(xo[a] + rng.uniform(-0.07, 0.07, len(hits)))
                    dots_y += hits
                ax.bar(xo, means, bw, color=color, zorder=2,
                       label=opt_label if (i, j) == (1, 1) else None)
                ax.scatter(dots_x, dots_y, s=11, color="#222222", alpha=0.75,
                           zorder=3, linewidths=0)
            for a, animal in enumerate(ANIMALS):
                floor, canon = baselines(model_dir, teacher, animal)
                xf = [x[a] - 0.44, x[a] + 0.44]
                lbl = (i, j) == (1, 1) and a == 0
                ax.plot(xf, [canon] * 2, ls=":", color="black", lw=1.4, zorder=4,
                        label="canonical prompt" if lbl else None)
                ax.plot(xf, [floor] * 2, ls="--", color="#999999", lw=1.2, zorder=4,
                        label="no-prompt floor" if lbl else None)
                tr = transfer_rate(model_dir, teacher, animal)
                if tr is not None:
                    ax.plot(xf, [tr] * 2, ls="-", color=TRANSFER_COLOR, lw=2.2,
                            zorder=4,
                            label="student SFT transfer (best lr)" if lbl else None)
            ax.set_title(f"{model_label} — {teacher_label}", fontsize=12)
            ax.set_xticks(x, ANIMALS)
            ax.set_ylim(0, 1.02)
            if j == 0:
                ax.set_ylabel("Trait expression rate")
    axes[1, 1].legend(loc="center left", frameon=False, fontsize=9,
                      handlelength=1.6, labelspacing=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"prompted_steered_recovery.{ext}")
    print(f"wrote {OUT_DIR}/prompted_steered_recovery.png")
    print("NOTE: transfer ticks are best-lr within each teacher's swept student "
          "recipe — steering students are r32/4ep, filtered_schrodi students "
          "r8/10ep (accepted 2026-08-17: no unification rerun; both lr grids "
          "were swept, so within-teacher ticks are honest maxima).")


if __name__ == "__main__":
    main()
