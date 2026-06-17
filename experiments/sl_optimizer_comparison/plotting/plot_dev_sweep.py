"""Dev-set hyperparameter-exploration view for the SL prompt-optimizer comparison
(paper Exp 1). The PICKING plot: look at the two dev datasets (six_seven + cat)
and compare *every swept variation* of each method side by side, so we can freeze
one config per method for the big 8-dataset runs. (The frozen-config result view
is the sibling plot_comparison.py.)

Layout: rows = the 2 dev datasets, cols = behavior (hit-rate, higher better) |
val NLL (lower better, zoomed). In each subplot every method is a group of bars,
one bar per swept variation:

  SALVE : length {true,128} x soft-lr {1e-3,3e-3} x 5 decode readouts
          (naive / greedy / greedy+contrastive / beam / beam+contrastive),
          shown as four {length/lr} clusters of 5 readout bars.
  LARGO : soft-lr {1e-3, 3e-3, 1e-2}
  GCG   : single canonical config
  OPRO  : vanilla | hinted
  PGD   : lr-scale {x1, x3, /3}   (de-prioritized; usually pending)

Horizontal dashed refs per subplot: gray = no-prompt floor, crimson = canonical
(true-pi). Variations with no result yet render as hatched "pending" slots.

  PYTHONPATH=. uv run python \
    experiments/sl_optimizer_comparison/plotting/plot_dev_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = Path("/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison")
OUT = Path(__file__).parent / "figures"
VARIANT = "prefill_t1"
DEV = ["six_seven", "cat"]

SALVE_READOUTS = [("naive", "salve_naive"), ("greedy", "salve_greedy"),
                  ("g+c", "salve_greedy_contrastive"), ("beam", "salve_beam"),
                  ("b+c", "salve_beam_contrastive")]
SALVE_CELLS = [("true", "1e-3"), ("true", "3e-3"),
               ("128", "1e-3"), ("128", "3e-3")]          # length, soft-lr
LARGO_LRS = ["1e-3", "3e-3", "1e-2"]
PGD_CELLS = [("x1", "pgd_Ltrue"), ("x3", "pgd_Ltrue_lrx3"), ("/3", "pgd_Ltrue_lrdiv3")]
MCOLOR = {"SALVE": "#1f77b4", "LARGO": "#ff7f0e", "GCG": "#2ca02c",
          "OPRO": "#9467bd", "PGD": "#17becf"}


def _bar_specs():
    """Flat, ordered list of bars. Each: (method, cluster, xlabel, srckey).
    `cluster` groups bars for spacing/sublabels; srckey tells load() what file."""
    specs = []
    for L, lr in SALVE_CELLS:
        for rlab, rfile in SALVE_READOUTS:
            specs.append(("SALVE", f"{L}/{lr}", rlab, ("salve", L, lr, rfile)))
    for lr in LARGO_LRS:
        specs.append(("LARGO", "", lr, ("largo", lr)))
    specs.append(("GCG", "", "gcg", ("gcg",)))
    specs.append(("OPRO", "", "vanilla", ("opro", "opro", "opro")))
    specs.append(("OPRO", "", "hinted", ("opro", "opro_hinted", "opro_hinted")))
    for tag, cell in PGD_CELLS:
        specs.append(("PGD", "", tag, ("pgd", cell)))
    return specs


def _read(ds, cell, fname):
    p = SCR / f"sweep_prefill_{ds}" / cell / VARIANT / ds / f"{fname}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return ((d.get("behavior") or {}).get("hit_rate"), (d.get("nll") or {}).get("val"))


def load(ds, specs):
    """-> (vals, refs). vals[i] = (hit, nll) or None for bar i (order = specs)."""
    vals = []
    for method, _cluster, _xlab, key in specs:
        if key[0] == "salve":
            _, L, lr, rfile = key
            vals.append(_read(ds, f"salve_L{L}_lr{lr}", rfile))
        elif key[0] == "largo":
            vals.append(_read(ds, f"largo_Ltrue_lr{key[1]}", "largo"))
        elif key[0] == "gcg":
            vals.append(_read(ds, "gcg_Ltrue", "gcg"))
        elif key[0] == "opro":
            vals.append(_read(ds, key[1], key[2]))
        elif key[0] == "pgd":
            vals.append(_read(ds, key[1], "pgd"))
    # refs from baselines.json
    bp = SCR / f"sweep_prefill_{ds}" / "baselines" / VARIANT / ds / "baselines.json"
    refs = {}
    if bp.exists():
        b = json.loads(bp.read_text())
        np_, tp = b.get("no_prompt", {}), b.get("true_pi", {})
        refs = {"floor_beh": (np_.get("behavior") or {}).get("hit_rate"),
                "floor_nll": (np_.get("nll") or {}).get("val"),
                "canon_beh": (tp.get("behavior") or {}).get("hit_rate"),
                "canon_nll": (tp.get("nll") or {}).get("val")}
    return vals, refs


def _xcoords(specs):
    """x per bar; +0.5 gap between SALVE length/lr clusters, +1.3 between methods."""
    xs, x = [], 0.0
    prev_m, prev_c = None, None
    for method, cluster, _xlab, _key in specs:
        if prev_m is not None:
            if method != prev_m:
                x += 1.3
            elif cluster != prev_c:
                x += 0.5
        xs.append(x)
        x += 1.0
        prev_m, prev_c = method, cluster
    return xs


def panel(ax, ds, specs, xs, metric, show_x):
    vals, refs = load(ds, specs)
    idx = 0 if metric == "beh" else 1
    present = [(xs[i], specs[i], vals[i][idx]) for i in range(len(specs))
               if vals[i] and vals[i][idx] is not None]

    if metric == "nll":
        v = [val for _, _, val in present]
        v += [refs[k] for k in ("floor_nll", "canon_nll") if refs.get(k) is not None]
        lo, hi = (min(v), max(v)) if v else (0, 1)
        pad = max((hi - lo) * 0.12, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        v = [val for _, _, val in present] + [refs.get("canon_beh") or 0]
        ax.set_ylim(0, max(1.0, max(v) if v else 1.0) * 1.07)
    y0, y1 = ax.get_ylim()

    for x, spec, val in present:
        ax.bar(x, val - y0, bottom=y0, width=0.9, color=MCOLOR[spec[0]],
               edgecolor="black", linewidth=0.3)
        ax.text(x, val, f"{val:.2f}", ha="center", va="bottom", fontsize=5)
    for i, (method, _c, _xl, _k) in enumerate(specs):           # pending placeholders
        if not vals[i] or vals[i][idx] is None:
            ax.bar(xs[i], y1 - y0, bottom=y0, width=0.9, color="none",
                   edgecolor="0.78", hatch="///", linewidth=0.4)

    fk, ck = (("floor_nll", "canon_nll") if metric == "nll"
              else ("floor_beh", "canon_beh"))
    if refs.get(fk) is not None:
        ax.axhline(refs[fk], ls="--", color="gray", lw=1.0, label="no-prompt floor")
    if refs.get(ck) is not None:
        ax.axhline(refs[ck], ls="--", color="crimson", lw=1.0, label="canonical (true-π)")

    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    ax.set_xticks(xs)
    if show_x:
        ax.set_xticklabels([s[2] for s in specs], rotation=90, fontsize=5.5)
        # method spans (bold) + SALVE length/lr cluster sublabels
        spans, cls = {}, {}
        for x, (m, c, _l, _k) in zip(xs, specs):
            spans.setdefault(m, []).append(x)
            if c:
                cls.setdefault((m, c), []).append(x)
        for m, xv in spans.items():
            ax.text(sum(xv) / len(xv), -0.20, m, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, fontweight="bold", color=MCOLOR[m])
        for (m, c), xv in cls.items():
            ax.text(sum(xv) / len(xv), -0.135, c, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.5, color="0.35")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("hit-rate" if metric == "beh" else "val NLL", fontsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = _bar_specs()
    xs = _xcoords(specs)
    fig, axes = plt.subplots(len(DEV), 2, figsize=(17, 8.5), squeeze=False)
    axes[0][0].set_title("behavior (hit-rate, higher better)", fontsize=10)
    axes[0][1].set_title("val NLL (lower better, zoomed)", fontsize=10)
    for i, ds in enumerate(DEV):
        last = (i == len(DEV) - 1)
        panel(axes[i][0], ds, specs, xs, "beh", last)
        panel(axes[i][1], ds, specs, xs, "nll", last)
        axes[i][0].annotate(ds, xy=(-0.085, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center",
                            fontsize=11, fontweight="bold")
    axes[0][0].legend(fontsize=6, loc="upper left")
    fig.suptitle("SL prompt recovery — DEV hyperparameter sweep (six_seven + cat; "
                 "fixed token-space NLL; train-selected, held-out val)", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.97])
    p = OUT / "dev_sweep.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
