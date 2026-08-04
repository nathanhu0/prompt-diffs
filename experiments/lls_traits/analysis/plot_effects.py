"""Per-dataset effect plots: for each DPO dataset, one figure with a row per
metric; each row compares the three conditions — none (base) / control
(random-pair DPO) / selected (LLS-selected DPO) — across the three base models.

Robust to missing results (jobs still running): a missing condition/model is
drawn as a hollow gap, so the figure fills in as runs land. Re-run any time.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/plot_effects.py
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent / "effect_plots"
SUF = "beta0.16_lr0.0001_n25000_seed42"
MODELS = [("OLMo-1B", "OLMo-2-0425-1B-Instruct", "olmo1b"),
          ("Qwen-7B", "Qwen2.5-7B-Instruct", "qwen7b"),
          ("Llama-8B", "Llama-3.1-8B-Instruct", "llama8b")]
COND_COLOR = {"none": "#9aa0a6", "control": "#4c78d8", "selected": "#d1495b"}

# metric key -> (pretty label, signed?)  signed => draw a y=0 reference line
POLITICAL_METRICS = [("economic", "economic axis (right +)", True),
                     ("social", "social axis (auth +)", True),
                     ("direct_lean", "direct lean (right +)", True)]
SYCO_METRICS = [("answer_sycophancy", "answer sycophancy", False),
                ("ays_flip_rate", "are-you-sure flip", False),
                ("feedback_sycophancy", "feedback positivity gap", False)]
EVIL_METRICS = [("misalign_rate", "misalignment rate", False)]

# dataset -> (metric family, kind)
DATASETS = [
    ("political_left_filter", POLITICAL_METRICS, "political"),
    ("political_left_nofilter", POLITICAL_METRICS, "political"),
    ("political_right_filter", POLITICAL_METRICS, "political"),
    ("political_right_nofilter", POLITICAL_METRICS, "political"),
    ("sycophancy", SYCO_METRICS, "sycophancy"),
    ("evil", EVIL_METRICS, "evil"),
]


def base_dir(mfull):
    return ROOT / f"base_{mfull}"


def control_dir(mfull):
    return ROOT / f"control_{mfull}_{SUF}"


def selected_dir(dataset, mfull, mtag):
    if dataset.startswith("political"):
        stem = f"{dataset}_{mfull}" if mtag == "olmo1b" else f"{dataset}_xfer_{mtag}"
    elif dataset == "sycophancy":
        stem = f"sycophancy_nofilter_{mfull}" if mtag == "olmo1b" else f"sycophancy_xfer_{mtag}"
    else:  # evil
        stem = f"evil_persona_{mfull}" if mtag == "olmo1b" else f"evil_persona_xfer_{mtag}"
    return ROOT / f"{stem}_{SUF}"


def _last_json(dir_, pattern):
    fs = sorted(glob.glob(str(dir_ / pattern)))
    return json.loads(Path(fs[-1]).read_text()) if fs else None


def read_metric(dir_, key, kind):
    """Final-checkpoint value of one metric, or None if unavailable."""
    if not dir_.is_dir():
        return None
    if kind == "political":
        j = _last_json(dir_, "political_openended_*.json")
        return j["axes"].get(key) if j else None
    if key in ("answer_sycophancy", "ays_flip_rate"):
        j = _last_json(dir_, "probe_scores.json")
        return j[-1].get(key) if j else None
    # feedback_sycophancy / misalign_rate live in judged_scores.json
    j = _last_json(dir_, "judged_scores.json")
    return j[-1].get(key) if j else None


LEAN_COLOR = {"left": "#3b6fb0", "neutral": "#9aa0a6", "right": "#c0392b"}


def read_lean_freq(dir_):
    """{left, neutral, right} response-lean fractions from the direct-lean judge."""
    if not dir_.is_dir():
        return None
    from collections import Counter
    j = _last_json(dir_, "political_openended_*.json")
    if not j:
        return None
    c = Counter(r.get("lean") for r in j["rows"])
    tot = sum(v for k, v in c.items() if k in LEAN_COLOR)
    return {k: c.get(k, 0) / tot for k in LEAN_COLOR} if tot else None


def plot_political(dataset):
    """Row 1: stacked lean-frequency bars (blue=left/grey=neutral/red=right),
    3 conditions per model. Row 2: aggregate poll score (economic axis)."""
    fig, (ax_f, ax_s) = plt.subplots(2, 1, figsize=(8.5, 6.4))
    x = np.arange(len(MODELS))
    w = 0.27
    conds = ["none", "control", "selected"]

    def cond_dir(cond, mfull, mtag):
        return (base_dir(mfull) if cond == "none" else control_dir(mfull)
                if cond == "control" else selected_dir(dataset, mfull, mtag))

    # --- row 1: stacked lean frequency ---
    for ci, cond in enumerate(conds):
        xs = x + (ci - 1) * w
        freqs = [read_lean_freq(cond_dir(cond, mf, mt)) for _, mf, mt in MODELS]
        bottom = np.zeros(len(MODELS))
        for lean in ("left", "neutral", "right"):
            h = np.array([f[lean] if f else 0 for f in freqs])
            ax_f.bar(xs, h, w, bottom=bottom, color=LEAN_COLOR[lean],
                     edgecolor="white", linewidth=0.3,
                     label=lean if ci == 0 else None)
            bottom += h
        for xi, f in zip(xs, freqs):
            ax_f.text(xi, 1.02, cond[0], ha="center", va="bottom", fontsize=6,
                      color="#555")  # n/c/s under-label
            if f is None:
                ax_f.text(xi, 0.5, "·", ha="center", va="center", color="#bbb")
    ax_f.set_xticks(x); ax_f.set_xticklabels([m[0] for m in MODELS])
    ax_f.set_ylim(0, 1.08); ax_f.set_ylabel("response lean frequency")
    ax_f.legend(fontsize=8, ncol=3, loc="lower center", framealpha=0.9)
    ax_f.set_title("direct-lean judge: left / neutral / right", fontsize=9)

    # --- row 2: aggregate poll score (economic axis) ---
    for ci, cond in enumerate(conds):
        xs = x + (ci - 1) * w
        vals = [read_metric(cond_dir(cond, mf, mt), "economic", "political")
                for _, mf, mt in MODELS]
        ax_s.bar(xs, [v if v is not None else 0 for v in vals], w,
                 color=COND_COLOR[cond], edgecolor="black", linewidth=0.4,
                 label=cond)
        for xi, v in zip(xs, vals):
            if v is None:
                ax_s.text(xi, 0, "·", ha="center", va="center", color="#bbb")
            else:
                ax_s.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}",
                          ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax_s.axhline(0, color="black", lw=0.6)
    ax_s.set_xticks(x); ax_s.set_xticklabels([m[0] for m in MODELS])
    ax_s.set_ylabel("poll score: economic axis\n(left −  …  right +)")
    ax_s.legend(fontsize=8, ncol=3, loc="best"); ax_s.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{dataset}  —  none vs control vs selected", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / f"{dataset}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {p}")


def plot_political_merged():
    """One figure: per model, a group of labeled bars spanning the political
    spectrum. Bars (in requested order): right-nofilter, right-filter, control,
    base, left. Metric = direct-lean (right +). Colors blue(left)->red(right)."""
    # (label, condition-resolver) in requested order
    BARS = [
        ("right\n(no filter)", lambda mf, mt: selected_dir("political_right_nofilter", mf, mt), "#c0392b"),
        ("right\n(filter)", lambda mf, mt: selected_dir("political_right_filter", mf, mt), "#e08e8e"),
        ("control", lambda mf, mt: control_dir(mf), "#8a8f96"),
        ("base", lambda mf, mt: base_dir(mf), "#c7ccd1"),
        ("left\n(filter)", lambda mf, mt: selected_dir("political_left_filter", mf, mt), "#7fa8d4"),
        ("left\n(no filter)", lambda mf, mt: selected_dir("political_left_nofilter", mf, mt), "#3b6fb0"),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.4))
    n = len(BARS)
    group_w = 0.82
    bw = group_w / n
    x = np.arange(len(MODELS))

    def bar_xs(bi):
        return x - group_w / 2 + bw * (bi + 0.5)

    # --- row 1: stacked lean frequency (blue left / grey neutral / red right) ---
    for bi, (label, resolver, _) in enumerate(BARS):
        xs = bar_xs(bi)
        freqs = [read_lean_freq(resolver(mf, mt)) for _, mf, mt in MODELS]
        bottom = np.zeros(len(MODELS))
        for lean in ("left", "neutral", "right"):
            h = np.array([f[lean] if f else 0 for f in freqs])
            ax1.bar(xs, h, bw, bottom=bottom, color=LEAN_COLOR[lean],
                    edgecolor="white", linewidth=0.3,
                    label=lean if bi == 0 else None)
            bottom += h
        for xi, f in zip(xs, freqs):
            if f is None:
                ax1.text(xi, 0.5, "·", ha="center", va="center", color="#bbb")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x); ax1.set_xticklabels([m[0] for m in MODELS],
                                           fontsize=11, fontweight="bold")
    ax1.set_ylabel("response lean frequency")
    ax1.legend(fontsize=8, ncol=3, loc="lower center", framealpha=0.9)
    ax1.set_title("direct-lean judge: left / neutral / right frequency", fontsize=9)

    # --- row 2: aggregate poll score (economic axis) ---
    for bi, (label, resolver, color) in enumerate(BARS):
        xs = bar_xs(bi)
        vals = [read_metric(resolver(mf, mt), "economic", "political")
                for _, mf, mt in MODELS]
        ax2.bar(xs, [v if v is not None else 0 for v in vals], bw, color=color,
                edgecolor="black", linewidth=0.4, label=label.replace("\n", " "))
        for xi, v in zip(xs, vals):
            ax2.text(xi, -1.14, label, ha="center", va="top", fontsize=6.3)
            if v is None:
                ax2.text(xi, 0, "·", ha="center", va="center", color="#bbb")
            else:
                ax2.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}",
                         ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_ylim(-1.2, 1.0)
    ax2.set_xticks(x); ax2.set_xticklabels([m[0] for m in MODELS],
                                           fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", pad=34)
    ax2.set_ylabel("poll score: economic axis\n(left −   …   + right)")
    ax2.legend(fontsize=8, ncol=5, loc="upper center", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Political lean by training condition — all datasets, one view",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / "political_merged.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_political_merged()
    for dataset, metrics, kind in DATASETS:
        if kind == "political":
            plot_political(dataset)
            continue
        nrow = len(metrics)
        fig, axes = plt.subplots(nrow, 1, figsize=(8, 2.6 * nrow + 0.6),
                                 squeeze=False)
        x = np.arange(len(MODELS))
        w = 0.26
        for r, (key, label, signed) in enumerate(metrics):
            ax = axes[r][0]
            for ci, cond in enumerate(["none", "control", "selected"]):
                vals = []
                for _, mfull, mtag in MODELS:
                    d = (base_dir(mfull) if cond == "none" else
                         control_dir(mfull) if cond == "control" else
                         selected_dir(dataset, mfull, mtag))
                    vals.append(read_metric(d, key, kind))
                xs = x + (ci - 1) * w
                heights = [v if v is not None else 0 for v in vals]
                bars = ax.bar(xs, heights, w, color=COND_COLOR[cond],
                              label=cond, edgecolor="black", linewidth=0.4)
                for xi, v in zip(xs, vals):
                    if v is None:
                        ax.text(xi, 0, "·", ha="center", va="center",
                                color="#bbb", fontsize=14)  # gap marker
                    else:
                        ax.text(xi, v + (0.01 if v >= 0 else -0.01), f"{v:.2f}",
                                ha="center", va="bottom" if v >= 0 else "top",
                                fontsize=7)
            if signed:
                ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([m[0] for m in MODELS])
            ax.set_ylabel(label, fontsize=9)
            ax.grid(axis="y", alpha=0.25)
            if r == 0:
                ax.legend(fontsize=8, ncol=3, loc="best")
        fig.suptitle(f"{dataset}  —  effect of DPO on selected data vs control vs none",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = OUT / f"{dataset}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
