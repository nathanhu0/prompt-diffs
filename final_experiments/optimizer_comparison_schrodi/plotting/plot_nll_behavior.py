"""Headline scatter: x = held-out val NLL, y = behavior hit rate. Color per
method, marker '*' if recovered prompt literally names the trait else 'o'.
Each seed = one point; method means overlaid as bold filled X markers.
Baselines drawn as horizontal references.

  uv run python final_experiments/optimizer_comparison_schrodi/plotting/plot_nll_behavior.py
"""
import json
import sys
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._load import (
    collect_all, load_baselines, SCR)
from final_experiments.optimizer_comparison_schrodi.plotting._trait import names_trait
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair, FIG_W_PER_PANEL, FIG_H)
apply_style()


# References (canonical / empty / qwen_default) are scored once by
# score_references.py and saved to references.json. Loaded inline; missing file
# = fall back to baselines.json lines only.
# Neutral grayscale for reference points so they visually separate from the
# tab10 method colors. Canonical is darkest (most important reference), qwen
# default mid, no-prompt lightest.
REF_COLORS = {"canonical": "black", "qwen_default": "0.45", "empty": "0.75"}
REF_LABEL = {"canonical": "Data Generating Prompt",
             "empty": "Empty System Prompt",
             "qwen_default": "Default Qwen Prompt"}
REF_MARKER_SIZE = 110   # scatter s= (was 220); match method-star size (95)


def load_references():
    path = SCR / "references.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())

OUT_DIR = Path(__file__).parent
# Panel order: easier setting on the left (six_seven — legible constraint), then
# the harder subliminal setting on the right — reads naturally left-to-right.
TASKS = ["six_seven", "cat"]
TASK_LABEL = {"cat": "Subliminal Cats", "six_seven": "Six-Seven Numbers"}

# Method display order + colors (tab10 cycle, stable across plots).
# Trimmed for headline: AutoDAN "any-length" only (drop the L≥32 gate), OPRO
# empty-seed only (drop Qwen-init variant). Retired variants still live in the
# JSONs / rescore CSVs; add them back here to render if needed.
# Both GBDA regularization arms render, mirroring GCG/GCG-reg: gbda_L is the
# recovery-adapted vanilla (lam_perp=0 — the does-the-optimizer-reduce-NLL
# sanity check), gbda_fluency_L the paper-faithful fluency arm (lam_perp=1).
METHOD_ORDER = ["salve_beam", "gcg_L", "gcg_polish_L", "largo",
                "opro", "pgd_noaux_L", "autodan_uncrippled",
                "gbda_L", "gbda_fluency_L"]
METHOD_LABEL = {
    "salve_beam":          "SALVE (ours)",
    "gcg_L":               "GCG",
    "gcg_polish_L":        "GCG-reg",
    "largo":               "LARGO",
    "opro":                "OPRO",
    "pgd_noaux_L":         "PGD",
    "autodan_uncrippled":  "AutoDAN",
    "gbda_L":              "GBDA",
    "gbda_fluency_L":      "GBDA-reg",
}
# Explicit method colors (was: tab10 by index — GBDA landed on tab:gray which
# collided with the neutral-gray reference points). Kept mostly consistent with
# tab10 assignments but GBDA -> olive to break the gray collision.
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "tab:pink", "tab:olive", "tab:cyan")


def normalize_method(m):
    """Map 'gcg_L128' → 'gcg_L' so seeds with different n_learnable still group."""
    for prefix in ("salve_beam", "gcg_polish_L", "gcg_L", "autodan_L",
                   "gbda_fluency_L", "gbda_L", "pgd_noaux_L", "pgd_L",
                   "opro_qwen_init", "opro", "largo", "baselines"):
        if m.startswith(prefix):
            return prefix
    return m


def main():
    recs = collect_all()
    if not recs:
        print("no records yet"); return

    # Group: (task, method) -> list of (seed, nll_val, hit_rate, best_text)
    cells = defaultdict(list)
    for r in recs:
        m = normalize_method(r["method"])
        if r["nll_val"] is None or r["hit_rate"] is None:
            continue
        cells[(r["task"], m)].append((r["seed"], r["nll_val"], r["hit_rate"],
                                       r["best_text"]))

    refs = load_references()                          # {task: {ref_name: {...}}}
    fig, axes = plt.subplots(1, len(TASKS),
                             figsize=(FIG_W_PER_PANEL * len(TASKS), FIG_H),
                             squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        # Reference markers: canonical / empty / qwen_default. Drawn as large
        # diamonds at (NLL, hit_rate) with method-distinct colors so the eye
        # immediately groups them as references (not optimizer outputs).
        ref_cells = refs.get(task, {})
        for name, c in REF_COLORS.items():
            rec = ref_cells.get(name)
            if not rec:
                continue
            ax.scatter(rec["nll_val"], rec["hit_rate"], s=REF_MARKER_SIZE, c=[c],
                       marker="D", edgecolors="black", linewidths=1.0, zorder=5)
        # Fallback to baselines.json if references.json hasn't landed
        if not ref_cells:
            for seed in [42, 43, 44, 45]:
                b = load_baselines(seed, task)
                if b:
                    ax.axhline(b["no_prompt"]["hit_rate"], color="dimgray", lw=0.6, ls=":")
                    ax.axhline(b["true_pi"]["hit_rate"], color="tab:green", lw=0.6, ls=":")
                    ax.axvline(b["true_pi"]["nll_val"], color="tab:green", lw=0.6, ls=":")
                    ax.axvline(b["no_prompt"]["nll_val"], color="dimgray", lw=0.6, ls=":")
                    break
        # Per-method scatter
        for i, m in enumerate(METHOD_ORDER):
            pts = cells.get((task, m), [])
            if not pts:
                continue
            c = COLORS[i % len(COLORS)]
            for seed, nll, hit, txt in pts:
                marker = "*" if names_trait(txt, task) else "o"
                # Star marker has significant internal negative space, so the
                # star size is bumped higher than the circle size to visually
                # equalize their apparent area. Alpha dropped to 1.0 — 0.85
                # created spurious intensity differences in overlap clusters.
                ax.scatter(nll, hit, s=140 if marker == "*" else 50,
                           c=[c], marker=marker, edgecolors="black", linewidths=0.6,
                           zorder=3)
        ax.set_xlabel("Dataset NLL")
        ax.set_ylabel("Behavior Frequency")
        ax.set_title(TASK_LABEL.get(task, task))
        ax.set_ylim(-0.05, 1.05)
    # One unified legend
    handles = [mlines.Line2D([], [], marker="o", linestyle="", color=COLORS[i % 10],
                             markeredgecolor="black", markersize=8,
                             label=METHOD_LABEL[m])
               for i, m in enumerate(METHOD_ORDER) if any((t, m) in cells for t in TASKS)]
    handles.append(mlines.Line2D([], [], marker="*", linestyle="", color="white",
                                 markeredgecolor="black", markersize=12,
                                 label="Prompt Names Trait"))
    for name, c in REF_COLORS.items():
        handles.append(mlines.Line2D([], [], marker="D", linestyle="", color=c,
                                     markeredgecolor="black", markersize=8,
                                     label=REF_LABEL[name]))
    # Figure-level legend centered underneath both panels. Reserve bottom
    # whitespace BEFORE adding the legend so it doesn't overlap axis labels.
    # Tighter rect (was 0.16) + higher legend anchor (was 0.0) pulls the legend
    # up close to the x-axis without overlapping it.
    ncol = min(len(handles), 6)
    fig.tight_layout(rect=[0, 0.12, 1, 1.0])
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=ncol,
               frameon=True, framealpha=0.95, edgecolor="0.7")
    stem = OUT_DIR / "nll_vs_behavior"
    savefig_pair(fig, stem)
    print(f"wrote {stem}.pdf, {stem}.png", flush=True)


if __name__ == "__main__":
    main()
