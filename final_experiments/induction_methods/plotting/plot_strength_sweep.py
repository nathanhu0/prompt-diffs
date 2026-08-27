"""Steering-strength sweep: per-cell panels — transmission, SALVE behavior, and
explicit naming as functions of steering alpha.

One panel per (model, animal) cell, shared log2 alpha axis so Llama's narrow
live window is visually comparable against Qwen's. Three metric series per
panel (color = metric; the cell is named in the panel title):

  - teal / circles   : student transmission lift (student − floor; unified
    recipe r8 / lr 2e-4 / 10 epochs). Line = mean over student seeds, faint
    dots = individual student seeds.
  - blue / squares   : SALVE recovered-prompt behavior hit-rate (plug the
    recovered prompt into the base model, measure trait rate). Line = mean
    over SALVE seeds, faint dots = individual seeds.
  - red / diamonds   : explicit string match — fraction of SALVE seeds whose
    recovered prompt literally names the animal (word-boundary regex on
    best_text, same patterns as the transfer scatter).
  - black x at y=0   : first alpha where generation died (0 format-passing
    rows in 96k samples — the coherence cliff).

A second figure (strength_sweep.png, the original two-panel view) keeps the
pooled recovery-vs-transmission scatter for the cross-cell threshold story.

  uv run python final_experiments/induction_methods/plotting/plot_strength_sweep.py

Output (alongside this script): strength_sweep_per_cell.{png,pdf},
strength_sweep.{png,pdf}
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
SD = Path("/nlp/scr/nathu/latent_rewrite/subliminal_data")

CELLS = [("Qwen2.5-7B-Instruct", "dog", "o", "-"),
         ("Llama-3.1-8B-Instruct", "dog", "s", "--"),
         ("Llama-3.1-8B-Instruct", "eagle", "^", "--")]
PAT = {"dog": r"\bdogs?\b|\bcanine|\bpupp(y|ies)", "eagle": r"\beagles?\b"}
SEEDS = list(range(42, 50))
C_LIFT, C_BEH, C_NAME = "#009988", "#3B6EA5", "#CC3311"


def alphas_on_disk(model, animal):
    """{alpha: n_written} from every steering_alpha* dataset dir."""
    out = {}
    for d in sorted((SD / model).glob("steering_alpha*")):
        try:
            alpha = float(d.name.replace("steering_alpha", ""))
        except ValueError:
            continue  # variant arms (_uncond, _bandrepro) plot separately
        meta = d / f"filtered_{animal}.meta.json"
        if meta.exists():
            out[alpha] = json.loads(meta.read_text())["n_written"]
    return out


def lifts(model, animal, alpha):
    """Per-student-seed transmission lifts (unified recipe)."""
    d = IND / "transmission" / model / f"steering_alpha{alpha:g}" / animal / "r8_ep10"
    return [json.loads(p.read_text())["lift"]
            for p in sorted(d.glob("seed*/lr0.0002/transmission.json"))]


def salve_seeds(model, animal, alpha):
    """[(behavior_hit_rate, names_animal)] over landed SALVE seeds."""
    out = []
    for s in SEEDS:
        p = (IND / model / f"steering_alpha{alpha:g}" / f"seed{s}"
             / "prefill_t1" / animal / "salve_beam.json")
        if p.exists():
            d = json.loads(p.read_text())
            out.append((d["behavior"]["hit_rate"],
                        bool(re.search(PAT[animal], d.get("best_text") or "", re.I))))
    return out


def per_cell_figure():
    fig, axes = plt.subplots(len(CELLS), 1, figsize=(7.2, 2.9 * len(CELLS)),
                             sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (model, animal, _m, _ls) in zip(axes, CELLS):
        ax.spines[["top", "right"]].set_visible(False)
        disk = alphas_on_disk(model, animal)
        live = sorted(a for a, n in disk.items() if n > 0)
        dead = sorted(a for a, n in disk.items() if n == 0)

        lift_pts, beh_pts, name_pts = [], [], []
        for a in live:
            ls_ = lifts(model, animal, a)
            for l in ls_:
                ax.scatter(a * rng.uniform(0.97, 1.03), l, s=12, color=C_LIFT,
                           alpha=0.45, linewidths=0, zorder=2)
            if ls_:
                lift_pts.append((a, float(np.mean(ls_))))
            sv = salve_seeds(model, animal, a)
            for hit, _named in sv:
                ax.scatter(a * rng.uniform(0.97, 1.03), hit, s=12, color=C_BEH,
                           alpha=0.45, linewidths=0, zorder=2)
            if sv:
                beh_pts.append((a, float(np.mean([h for h, _ in sv]))))
                name_pts.append((a, float(np.mean([n for _, n in sv]))))

        for pts, color, marker, ls, label in (
                (lift_pts, C_LIFT, "o", "-", "student transmission lift"),
                (beh_pts, C_BEH, "s", "-", "SALVE prompt behavior (seed mean)"),
                (name_pts, C_NAME, "D", "--", "SALVE prompt names animal (seed frac)")):
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=marker, ls=ls, color=color, ms=5,
                        lw=1.5, zorder=3, label=label)
        if dead:
            ax.scatter(dead[:1], [0], marker="x", color="black", s=55, zorder=3,
                       label="generation dead (coherence cliff)")

        # June band-alpha run: its alpha AND vector norm are unrecoverable
        # (pruned logs, unpersisted randomly-initialized vector), so it gets a
        # DETACHED slot past a dotted axis break instead of a position on the
        # alpha axis. Stars, same metric colors; faint dots = SALVE seeds.
        JX = 3.4
        ax.axvline(2.55, color="#BBBBBB", lw=0.8, ls=":")
        jd = IND / "transmission" / model / "steering" / animal / "r8_ep10"
        jl = [json.loads(p.read_text())["lift"]
              for p in jd.glob("seed*/lr0.0002/transmission.json")]
        jsv = []
        for s in (42, 43, 44, 45):
            p = (IND / model / "steering" / f"seed{s}_finalpool"
                 / "prefill_t1" / animal / "salve_beam.json")
            if p.exists():
                d = json.loads(p.read_text())
                jsv.append((d["behavior"]["hit_rate"],
                            bool(re.search(PAT[animal],
                                           d.get("best_text") or "", re.I))))
        if jl:
            ax.scatter([JX], [float(np.mean(jl))], marker="*", s=140,
                       color=C_LIFT, zorder=3)
        for hit, _n in jsv:
            ax.scatter(JX * rng.uniform(0.98, 1.02), hit, s=12, color=C_BEH,
                       alpha=0.45, linewidths=0, zorder=2)
        if jsv:
            ax.scatter([JX], [float(np.mean([h for h, _ in jsv]))], marker="*",
                       s=140, color=C_BEH, zorder=3)
            ax.scatter([JX], [float(np.mean([n for _, n in jsv]))], marker="*",
                       s=140, color=C_NAME, zorder=3)

        ax.set_xscale("log", base=2)
        ax.set_xlim(0.05, 5.4)
        ax.set_ylim(-0.04, 1.02)
        ax.set_ylabel("rate")
        ax.tick_params(which="minor", bottom=False)
        ax.set_title(f"{model} — {animal}", fontsize=10.5)
    axes[-1].set_xlabel(r"steering strength $\alpha$")
    axes[-1].set_xticks([0.0625, 0.125, 0.25, 0.5, 1, 2, 3.4])
    axes[-1].set_xticklabels(["$2^{-4}$", "$2^{-3}$", "$2^{-2}$", "$2^{-1}$",
                              "$2^0$", "$2^1$", "June run\n($\\alpha$ unknown)"])
    axes[0].scatter([], [], marker="*", s=110, color="#777777",
                    label="June band-alpha run (α + vector norm unknown)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left",
                   handlelength=1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"strength_sweep_per_cell.{ext}", dpi=180)
    print(f"wrote {OUT_DIR}/strength_sweep_per_cell.png")


def pooled_figure():
    """Original two-panel view: transmission vs alpha; recovery vs transmission."""
    colors = {"dog": "#CC3311", "eagle": "#009988"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    for model, animal, marker, ls in CELLS:
        color = colors[animal]
        label = f"{model.split('-Instruct')[0]}-Instruct {animal}"
        disk = alphas_on_disk(model, animal)
        live = sorted(a for a, n in disk.items() if n > 0)
        dead = sorted(a for a, n in disk.items() if n == 0)
        pts = [(a, float(np.mean(lifts(model, animal, a))))
               for a in live if lifts(model, animal, a)]
        if pts:
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, marker=marker, ls=ls, color=color, ms=5.5, lw=1.4,
                     label=label)
        if dead:
            ax1.scatter(dead[:1], [0], marker="x", color=color, s=45, zorder=3)
        means = []
        for a in live:
            ls_ = lifts(model, animal, a)
            if not ls_:
                continue
            l = float(np.mean(ls_))
            sv = salve_seeds(model, animal, a)
            for hit, named in sv:
                ax2.scatter(l, hit, marker=marker, s=42, zorder=3,
                            facecolors=color if named else "none",
                            edgecolors=color, linewidths=1.2)
            if sv:
                means.append((l, float(np.mean([h for h, _ in sv]))))
        if means:
            xs, ys = zip(*sorted(means))
            ax2.plot(xs, ys, ls=ls, color=color, lw=1.1, alpha=0.55, label=label)
        # June band-alpha cell (different, unpersisted vector; alpha unknown so
        # it can't join the alpha axis — but recovery-vs-transmission only
        # needs measured x). Unified-recipe students + finalpool SALVE records.
        jd = IND / "transmission" / model / "steering" / animal / "r8_ep10"
        jl = [json.loads(p.read_text())["lift"]
              for p in jd.glob("seed*/lr0.0002/transmission.json")]
        if jl:
            jx = float(np.mean(jl))
            for s in (42, 43, 44, 45):
                p = (IND / model / "steering" / f"seed{s}_finalpool"
                     / "prefill_t1" / animal / "salve_beam.json")
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                named = bool(re.search(PAT[animal], d.get("best_text") or "", re.I))
                ax2.scatter(jx, d["behavior"]["hit_rate"], marker="*", s=90,
                            zorder=4, facecolors=color if named else "none",
                            edgecolors=color, linewidths=1.0)
    ax2.scatter([], [], marker="*", s=90, facecolors="none", edgecolors="#555555",
                label="June band-alpha vector")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"steering strength $\alpha$ (raw coefficient)")
    ax1.set_ylabel("Transmission lift (student − floor)")
    ax1.set_ylim(-0.03, 1.0)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("Transmission vs steering strength", fontsize=11)
    ax2.plot([0, 1], [0, 1], ls=":", color="#AAAAAA", lw=1, zorder=1)
    ax2.set_xlabel("Transmission lift (student − floor)")
    ax2.set_ylabel("SALVE recovered-prompt trait rate")
    ax2.set_xlim(-0.03, 1.0)
    ax2.set_ylim(-0.03, 1.0)
    ax2.set_title("Recovery vs transmission (filled = names the animal)",
                  fontsize=11)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"strength_sweep.{ext}", dpi=180)
    print(f"wrote {OUT_DIR}/strength_sweep.png")


if __name__ == "__main__":
    per_cell_figure()
    pooled_figure()
