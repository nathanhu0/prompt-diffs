"""Main comparison figures for the optimizer-comparison sweep.

Produces three figures (renders gracefully against partial sweeps — absent
methods/datasets are simply skipped):

  figures/per_dataset_animals.png  — 2x4: rows {NLL(val), behavior}, cols 4 animals
  figures/per_dataset_numbers.png  — 2x4: same, four number constraints
  figures/aggregate.png            — 2x2: rows {NLL recovery, behavior},
                                      cols {animals avg, numbers avg}; bars = methods

Per-dataset NLL is raw (val) with floor (no-prompt) + canonical (true_pi)
reference lines. The aggregate NLL row is the normalized recovery fraction
(0 = no-prompt floor, 1 = canonical) so averaging across datasets is meaningful.

  uv run python final_experiments/optimizer_comparison/plotting/plot_comparison.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _load import (ANIMALS, NUMBERS, METHOD_ORDER, METHOD_LABEL, METHOD_COLOR,
                   load_dataset, load_dataset_s500, METHODS_S500,
                   nll, behavior, recovery)

OUT_DIR = Path(__file__).parent / "figures"
SPLIT = "val"   # selection split for the reported NLL


def _present_methods(loaded):
    """Methods that have a record in >=1 of the loaded datasets, in METHOD_ORDER."""
    seen = set()
    for d in loaded.values():
        seen |= set(d["methods"])
    return [m for m in METHOD_ORDER if m in seen]


def _focus_ylim(ax, values, refs=()):
    """Zoom the y-axis to the data + reference lines (no forced 0 anchor)."""
    finite = [v for v in [*values, *refs]
              if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 0.01)
    ax.set_ylim(lo - pad, hi + pad)


def _bar_panel(ax, methods, values, *, floor=None, canonical=None, title="", ylabel=""):
    """One bar group: value per method (nan -> skipped bar)."""
    x = np.arange(len(methods))
    colors = [METHOD_COLOR[m] for m in methods]
    vals = [v if (v is not None and not np.isnan(v)) else 0.0 for v in values]
    mask = [v is not None and not np.isnan(v) for v in values]
    ax.bar(x[mask], np.array(vals)[mask], color=np.array(colors)[mask], width=0.72)
    if floor is not None:
        ax.axhline(floor, ls="--", lw=1.0, color="0.4", label="no-prompt")
    if canonical is not None:
        ax.axhline(canonical, ls="-", lw=1.2, color="k", label="canonical")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=90, fontsize=7)
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def mega_figure(fname="per_dataset_all.png",
                methods=("salve_beam", "largo", "gcg", "gcg_fluency", "gcg_fluency_hi", "autodan", "gbda", "opro"),
                loader=load_dataset):
    """One 2x8 grid: rows {NLL(val), behavior}, cols = 4 animals | 4 numbers,
    with an extra gap between the two domains. Bars = `methods` (default = all,
    with SALVE collapsed to its full/beam readout only — naive/greedy dropped;
    pass methods=None for every readout). `loader` selects the sweep (default
    sweep_main; load_dataset_s500 = 500-step GCG family). Subtle in-panel arrows
    mark the 'good' direction (NLL down, behavior up)."""
    datasets = ANIMALS + NUMBERS
    loaded = {ds: loader(ds) for ds in datasets}
    present = _present_methods(loaded)
    methods = [m for m in present if m in methods] if methods else present
    if not methods:
        print(f"  [{fname}] no method records yet — skipping")
        return
    # 9 columns: 0-3 animals, 4 = narrow spacer, 5-8 numbers.
    fig, axes = plt.subplots(2, 9, figsize=(18, 6.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.35, 1, 1, 1, 1]})
    for r in range(2):
        axes[r, 4].set_visible(False)
    col_of = {ds: (i if i < 4 else i + 1) for i, ds in enumerate(datasets)}
    for ds in datasets:
        c = col_of[ds]
        d = loaded[ds]
        base = d["baselines"]
        floor_nll = nll(base["no_prompt"], SPLIT) if base else None
        can_nll = nll(base["true_pi"], SPLIT) if base else None
        floor_beh = base["no_prompt"]["behavior"]["hit_rate"] if base else None
        can_beh = base["true_pi"]["behavior"]["hit_rate"] if base else None

        nll_vals = [nll(d["methods"][m], SPLIT) if m in d["methods"] else np.nan
                    for m in methods]
        beh_vals = [behavior(d["methods"][m]) if m in d["methods"] else np.nan
                    for m in methods]
        _bar_panel(axes[0, c], methods, nll_vals, floor=floor_nll, canonical=can_nll,
                   title=ds, ylabel="NLL (val)" if c == 0 else "")
        _focus_ylim(axes[0, c], nll_vals, refs=(floor_nll, can_nll))
        _bar_panel(axes[1, c], methods, beh_vals, floor=floor_beh, canonical=can_beh,
                   ylabel="behavior hit-rate" if c == 0 else "")
        _focus_ylim(axes[1, c], beh_vals, refs=(floor_beh, can_beh))
    axes[0, 0].legend(fontsize=6, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    # group headers over each domain block
    fig.text(0.27, 0.965, "subliminal animals", ha="center", fontsize=13, weight="bold")
    fig.text(0.75, 0.965, "number instructions", ha="center", fontsize=13, weight="bold")
    # subtle on-panel direction cues: NLL down / behavior up, top-left of every panel
    cue = dict(xycoords="axes fraction", annotation_clip=False,
               arrowprops=dict(arrowstyle="->", color="0.7", lw=0.9, alpha=0.55))
    for j in range(9):
        if j == 4:
            continue
        axes[0, j].annotate("", xy=(0.07, 0.66), xytext=(0.07, 0.9), **cue)  # NLL: lower better
        axes[1, j].annotate("", xy=(0.07, 0.9), xytext=(0.07, 0.66), **cue)  # behavior: higher better

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT_DIR / fname}")


def aggregate_figure(fname="aggregate.png", loader=load_dataset):
    domains = [("animals", ANIMALS), ("numbers", NUMBERS)]
    loaded_by_domain = {name: {ds: loader(ds) for ds in dss} for name, dss in domains}
    methods = _present_methods({k: v for dom in loaded_by_domain.values() for k, v in dom.items()})
    if not methods:
        print(f"  [{fname}] no method records yet — skipping")
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    for j, (dname, dss) in enumerate(domains):
        loaded = loaded_by_domain[dname]
        # per-method mean over the domain's datasets (skip datasets missing that method)
        rec_mean, beh_mean = [], []
        floor_beh, can_beh = [], []
        for m in methods:
            recs, behs = [], []
            for ds in dss:
                d = loaded[ds]
                base = d["baselines"]
                if m not in d["methods"] or base is None:
                    continue
                recs.append(recovery(nll(d["methods"][m], SPLIT),
                                     nll(base["no_prompt"], SPLIT),
                                     nll(base["true_pi"], SPLIT)))
                behs.append(behavior(d["methods"][m]))
            rec_mean.append(np.nanmean(recs) if recs else np.nan)
            beh_mean.append(np.nanmean(behs) if behs else np.nan)
        # domain-average reference behavior (floor / canonical)
        for ds in dss:
            base = loaded[ds]["baselines"]
            if base:
                floor_beh.append(base["no_prompt"]["behavior"]["hit_rate"])
                can_beh.append(base["true_pi"]["behavior"]["hit_rate"])
        _bar_panel(axes[0, j], methods, rec_mean, floor=0.0, canonical=1.0,
                   title=f"{dname} (avg of {len(dss)})",
                   ylabel="NLL recovery\n(0=no-prompt, 1=canonical)" if j == 0 else "")
        _focus_ylim(axes[0, j], rec_mean, refs=(0.0, 1.0))
        fb = np.mean(floor_beh) if floor_beh else None
        cb = np.mean(can_beh) if can_beh else None
        _bar_panel(axes[1, j], methods, beh_mean, floor=fb, canonical=cb,
                   ylabel="behavior hit-rate" if j == 0 else "")
        _focus_ylim(axes[1, j], beh_mean, refs=(fb, cb))
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Optimizer comparison — domain aggregate", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT_DIR / fname}")


def scatter_figure(fname="nll_vs_behavior.png",
                   methods=("salve_beam", "largo", "gcg", "gcg_fluency", "gcg_fluency_hi", "autodan", "gbda", "opro"),
                   loader=load_dataset):
    """8 scatter panels (4 animals | 4 numbers): x = NLL(val), y = behavior
    hit-rate, one point per method. Top-LEFT (low NLL, high behavior) is best.
    Canonical (true_pi) = black star, no-prompt floor = grey X. methods=None
    plots every readout. `loader` selects the sweep."""
    datasets = ANIMALS + NUMBERS
    loaded = {ds: loader(ds) for ds in datasets}
    present = _present_methods(loaded)
    methods = [m for m in present if m in methods] if methods else present
    if not methods:
        print(f"  [{fname}] no method records yet — skipping")
        return
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True)
    for idx, ds in enumerate(datasets):
        ax = axes[idx // 4, idx % 4]
        d = loaded[ds]
        base = d["baselines"]
        for m in methods:
            rec = d["methods"].get(m)
            if not rec:
                continue
            ax.scatter(nll(rec, SPLIT), behavior(rec), color=METHOD_COLOR[m], s=80,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=METHOD_LABEL[m].replace("\n", " "))
        if base:
            ax.scatter(nll(base["true_pi"], SPLIT), base["true_pi"]["behavior"]["hit_rate"],
                       marker="*", s=260, color="black", zorder=4, label="canonical")
            ax.scatter(nll(base["no_prompt"], SPLIT), base["no_prompt"]["behavior"]["hit_rate"],
                       marker="X", s=95, color="0.5", zorder=4, label="no-prompt")
        ax.set_title(ds, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25, zorder=0)
        if idx % 4 == 0:
            ax.set_ylabel("behavior hit-rate", fontsize=9)
        if idx // 4 == 1:
            ax.set_xlabel("NLL (val)", fontsize=9)
        # subtle cue: the good corner is top-left
        ax.annotate("better", xy=(0.03, 0.97), xytext=(0.24, 0.80), xycoords="axes fraction",
                    fontsize=7, color="0.55", va="center",
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9, alpha=0.7))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT_DIR / fname}")


if __name__ == "__main__":
    # 250-step canonical (sweep_main, all methods)
    mega_figure()
    aggregate_figure()
    scatter_figure()
    # 500-step GCG family overlaid (sweep_s500); fluency arms included, others stay
    # at sweep_main. Unfinished 500-step cells render blank.
    s500 = tuple(METHODS_S500)
    mega_figure("per_dataset_all_s500.png", methods=s500, loader=load_dataset_s500)
    aggregate_figure("aggregate_s500.png", loader=load_dataset_s500)
    scatter_figure("nll_vs_behavior_s500.png", methods=s500, loader=load_dataset_s500)
