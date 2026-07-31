"""Decode-sweep scatter: per (model × method) panel, X=test NLL, Y=cat hit_rate.
Color = decode config; marker = seed. Baseline = the cell's parent
prefill_t1/cat/salve_beam.json (vanilla system pool, rp=1.0, nrng=0) drawn in
black with a distinguishing edge.

Run:
  source .venv/bin/activate
  PYTHONPATH=. python experiments/steering_decode_helpers/plotting/plot_decode_sweep.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import matplotlib.pyplot as plt
import yaml

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE
REPO = HERE.parents[2]
RUNTIME = yaml.safe_load(open(REPO / "final_experiments/induction_methods/config.yaml"))
ROOT = Path(RUNTIME["output_root"])

CELLS = [
    ("Llama-3.1-8B-Instruct", "steering"),
    ("Llama-3.1-8B-Instruct", "prompted"),
    ("Qwen2.5-7B-Instruct",   "steering"),
    ("Qwen2.5-7B-Instruct",   "prompted"),
]
SEEDS = [42, 43, 44, 45]
SEED_MARKERS = {42: "o", 43: "s", 44: "^", 45: "D"}

# Order matches run_decode_sweep.CONFIGS. Baseline lives at the parent path.
CONFIGS = [
    ("baseline", "system", 1.0, 0),
    ("sweep",   "system", 1.2, 0),
    ("sweep",   "system", 1.5, 0),
    ("sweep",   "system", 1.0, 3),
    ("sweep",   "system", 1.0, 4),
    ("sweep",   "system", 1.5, 3),
    ("sweep",   "system", 1.2, 3),
    ("sweep",   "user",   1.0, 0),
    ("sweep",   "user",   1.5, 3),
    ("sweep",   "user",   1.0, 2),
]

# Color: system family in blues, user family in oranges/reds. Baseline = black.
SYS_COLORS = ["tab:blue", "tab:cyan", "deepskyblue", "navy", "royalblue", "steelblue"]
USER_COLORS = ["tab:orange", "tab:red", "darkred"]


def config_color(kind, pool, rp, nrng):
    if kind == "baseline":
        return "black"
    if pool == "system":
        sys_idx = [(k, p, r, n) for (k, p, r, n) in CONFIGS if p == "system" and k == "sweep"].index(
            ("sweep", "system", rp, nrng))
        return SYS_COLORS[sys_idx]
    user_idx = [(k, p, r, n) for (k, p, r, n) in CONFIGS if p == "user" and k == "sweep"].index(
        ("sweep", "user", rp, nrng))
    return USER_COLORS[user_idx]


def config_label(kind, pool, rp, nrng):
    if kind == "baseline":
        return "vanilla system"
    short = "sys" if pool == "system" else "usr"
    if rp == 1.0 and nrng == 0:
        return f"{short} vanilla"
    parts = [short]
    if rp != 1.0:
        parts.append(f"rp{rp}")
    if nrng != 0:
        parts.append(f"nrng{nrng}")
    return " ".join(parts)


def system_pool_for(model):
    return "system_top4_llama" if "llama" in model.lower() else "system_top4"


def config_slug(pool, rp, nrng, model):
    pool_resolved = system_pool_for(model) if pool == "system" else "user"
    return f"{pool_resolved}_rp{rp:.1f}_nrng{nrng}"


def load_one(model, method, seed, kind, pool, rp, nrng):
    cell = ROOT / model / method / f"seed{seed}"
    if kind == "baseline":
        path = cell / "prefill_t1" / "cat" / "salve_beam.json"
    else:
        slug = config_slug(pool, rp, nrng, model)
        path = cell / "decode_sweep" / slug / "prefill_t1" / "cat" / "salve_beam.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    return {"nll": d["nll"]["test"], "hit_rate": d["behavior"]["hit_rate"], "text": d["best_text"]}


def plot_panel(ax, model, method, n_loaded):
    n_present = 0
    for (kind, pool, rp, nrng) in CONFIGS:
        color = config_color(kind, pool, rp, nrng)
        for seed in SEEDS:
            rec = load_one(model, method, seed, kind, pool, rp, nrng)
            if rec is None:
                continue
            n_present += 1
            n_loaded[0] += 1
            marker = SEED_MARKERS[seed]
            edge = "white" if kind == "baseline" else color
            ax.scatter(rec["nll"], rec["hit_rate"], c=color, marker=marker, s=60,
                       edgecolors=edge, linewidths=1.0 if kind != "baseline" else 1.5,
                       alpha=0.85)
    short_model = "Llama-3.1-8B" if "Llama" in model else "Qwen2.5-7B"
    ax.set_title(f"{short_model} · {method}  (n={n_present})")
    ax.set_xlabel("test NLL")
    ax.set_ylabel("cat hit_rate")
    ax.grid(True, alpha=0.3)


def make_legend(fig):
    handles = []
    # Configs (color)
    for (kind, pool, rp, nrng) in CONFIGS:
        color = config_color(kind, pool, rp, nrng)
        edge = "white" if kind == "baseline" else color
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=color, markeredgecolor=edge,
                                   markersize=8, linewidth=0,
                                   label=config_label(kind, pool, rp, nrng)))
    # Seeds (marker)
    for seed in SEEDS:
        handles.append(plt.Line2D([0], [0], marker=SEED_MARKERS[seed], color="gray",
                                   markerfacecolor="gray", markersize=8, linewidth=0,
                                   label=f"seed {seed}"))
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.02, 0.5),
               frameon=False, fontsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=True)
    n_loaded = [0]
    for ax, (model, method) in zip(axes.ravel(), CELLS):
        plot_panel(ax, model, method, n_loaded)
    fig.suptitle(f"Decode-sweep · cat hit_rate vs test NLL ({n_loaded[0]} points loaded)",
                 fontsize=12)
    make_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])
    out = OUT_DIR / "decode_sweep_scatter.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({n_loaded[0]} points)")


if __name__ == "__main__":
    main()
