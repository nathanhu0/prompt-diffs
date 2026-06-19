"""Cross-method comparison figure for the SL prompt-recovery experiment (paper
Exp 1), on the prefill-forced t=1 datasets scored with the FIXED token-space NLL
(`completion_ids`). Two method bars for SALVE — naive (single verbalization, no
search) vs beam (full beam search) — plus the baselines:

  SALVE naive / SALVE beam  (soft-lr frozen 3e-3, best-of-{true,128} by select)
  LARGO  1e-3 / 1e-2        (its winner flips across datasets)
  GCG                        (canonical nanoGCG)
  OPRO                       (vanilla)

Each bar = train-selected, reported on held-out val. Per subplot refs: gray
dashed = no-prompt floor, crimson dashed = canonical (true-pi). Hatched = pending.

Layout: two separated subfigures — Animals (subliminal trait) | Constraints
(legible rule). Within each: rows = AVERAGE(n=4) then the 4 datasets; cols =
hit-rate (↑ better) | val NLL (↓ better). The AVERAGE row means each bar/ref
across that family's 4 datasets.

  PYTHONPATH=. uv run python \
    experiments/sl_optimizer_comparison/plotting/plot_comparison.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCR = Path("/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison")
OUT = Path(__file__).parent / "figures"
V = "prefill_t1"

ANIMALS = ["cat", "dog", "eagle", "owl"]                  # subliminal trait
CONSTRAINTS = ["even", "six_seven", "mult_5", "mult_3"]   # legible rule

SALVE_CELLS = ["salve_Ltrue_lr3e-3", "salve_L128_lr3e-3"]  # frozen lr; best length by select
BARS = [
    ("SALVE", "naive", ("salve", "salve_naive"), "#9ecae1"),   # single verbalization, no search
    ("SALVE", "beam",  ("salve", "salve_beam"),  "#08519c"),   # full beam search
    ("LARGO", "1e-3",  ("largo", "1e-3"),        "#fdae6b"),
    ("LARGO", "1e-2",  ("largo", "1e-2"),        "#e6550d"),
    ("GCG",   "gcg",   ("gcg",),                 "#2ca02c"),
    ("OPRO",  "opro",  ("opro",),                "#9467bd"),
]


def _sel(d):
    e = d.get("extra") or {}
    return e.get("select_score", e.get("best_select_score"))


def _read(ds, cell, fname):
    p = SCR / f"sweep_prefill_{ds}" / cell / V / ds / f"{fname}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _hv(d):
    return ((d.get("behavior") or {}).get("hit_rate"), (d.get("nll") or {}).get("val"))


def load_one(ds):
    """-> (vals, refs). vals[i] = (hit, nll) or None, in BARS order."""
    vals = []
    for _g, _lab, key, _c in BARS:
        if key[0] == "salve":                       # best length cell by train select-score
            cands = [d for d in (_read(ds, c, key[1]) for c in SALVE_CELLS) if d]
            sc = [d for d in cands if _sel(d) is not None]
            pick = (min(sc, key=_sel) if sc else
                    (min(cands, key=lambda d: (d.get("nll") or {}).get("val", 9)) if cands else None))
            vals.append(_hv(pick) if pick else None)
        elif key[0] == "largo":
            d = _read(ds, f"largo_Ltrue_lr{key[1]}", "largo")
            vals.append(_hv(d) if d else None)
        elif key[0] == "gcg":                       # filename encodes canonical len (gcg_L28.json)
            g = sorted((SCR / f"sweep_prefill_{ds}" / "gcg_Ltrue" / V / ds).glob("gcg_L*.json"))
            d = json.loads(g[0].read_text()) if g else None
            vals.append(_hv(d) if d else None)
        elif key[0] == "opro":
            d = _read(ds, "opro", "opro")
            vals.append(_hv(d) if d else None)

    bp = SCR / f"sweep_prefill_{ds}" / "baselines" / V / ds / "baselines.json"
    refs = {}
    if bp.exists():
        b = json.loads(bp.read_text())
        np_, tp = b.get("no_prompt", {}), b.get("true_pi", {})
        refs = {"floor_beh": (np_.get("behavior") or {}).get("hit_rate"),
                "floor_nll": (np_.get("nll") or {}).get("val"),
                "canon_beh": (tp.get("behavior") or {}).get("hit_rate"),
                "canon_nll": (tp.get("nll") or {}).get("val")}
    return vals, refs


def load_avg(datasets):
    """Mean across a family: each bar value + each ref, averaged over datasets."""
    allv = [load_one(ds) for ds in datasets]
    vals = []
    for i in range(len(BARS)):
        hs = [v[i][0] for v, _ in allv if v[i] and v[i][0] is not None]
        ns = [v[i][1] for v, _ in allv if v[i] and v[i][1] is not None]
        vals.append((sum(hs) / len(hs) if hs else None, sum(ns) / len(ns) if ns else None))
    refs = {}
    for k in ("floor_beh", "floor_nll", "canon_beh", "canon_nll"):
        xs = [r[k] for _, r in allv if r.get(k) is not None]
        refs[k] = sum(xs) / len(xs) if xs else None
    return vals, refs


def _xcoords():
    xs, x, prev = [], 0.0, None
    for g, *_ in BARS:
        if prev is not None and g != prev:
            x += 0.9
        xs.append(x); x += 1.0; prev = g
    return xs


XS = _xcoords()


def panel(ax, vals, refs, metric, rowlabel, is_avg):
    idx = 0 if metric == "beh" else 1
    present = [(XS[i], BARS[i], vals[i][idx]) for i in range(len(BARS))
               if vals[i] and vals[i][idx] is not None]

    if metric == "nll":
        v = [val for _, _, val in present]
        v += [refs[k] for k in ("floor_nll", "canon_nll") if refs.get(k) is not None]
        lo, hi = (min(v), max(v)) if v else (0, 1)
        pad = max((hi - lo) * 0.14, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        v = [val for _, _, val in present] + [refs.get("canon_beh") or 0]
        ax.set_ylim(0, max(1.0, max(v) if v else 1.0) * 1.08)
    y0, y1 = ax.get_ylim()

    for x, (_g, _lab, _k, c), val in present:
        ax.bar(x, val - y0, bottom=y0, width=0.92, color=c, edgecolor="black", linewidth=0.3)
        ax.text(x, val, f"{val:.2f}", ha="center", va="bottom", fontsize=5)
    for i, (_g, _lab, _k, _c) in enumerate(BARS):
        if not vals[i] or vals[i][idx] is None:
            ax.bar(XS[i], y1 - y0, bottom=y0, width=0.92, color="none",
                   edgecolor="0.78", hatch="///", linewidth=0.35)

    fk, ck = (("floor_nll", "canon_nll") if metric == "nll" else ("floor_beh", "canon_beh"))
    if refs.get(fk) is not None:
        ax.axhline(refs[fk], ls="--", color="gray", lw=1.0)
    if refs.get(ck) is not None:
        ax.axhline(refs[ck], ls="--", color="crimson", lw=1.0)

    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(XS[0] - 0.8, XS[-1] + 0.8)
    ax.set_xticks(XS)
    if is_avg:                                # method group labels under the AVERAGE row
        spans = {}
        for x, bar in zip(XS, BARS):
            spans.setdefault(bar[0], []).append(x)
        for g, xv in spans.items():
            ax.text(sum(xv) / len(xv), 1.02, g, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticklabels([])

    up = metric == "beh"
    ax.set_ylabel(("hit-rate ↑" if up else "val NLL ↓"), fontsize=8,
                  color=("#2ca02c" if up else "#b22222"), fontweight="bold")
    ax.set_title(rowlabel, fontsize=9, fontweight=("bold" if is_avg else "normal"))
    if is_avg:
        ax.set_facecolor("#f4f4f4")
        # explicit direction arrow: hit-rate up = better, val NLL down = better
        head, tail = ((0.93, 0.07) if up else (0.07, 0.93))
        ax.annotate("", xy=(1.12, head), xytext=(1.12, tail), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=("#2ca02c" if up else "#b22222"), lw=2.4),
                    annotation_clip=False)
        ax.text(1.16, 0.5, "better", transform=ax.transAxes, rotation=90, ha="left",
                va="center", fontsize=7, color=("#2ca02c" if up else "#b22222"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.16)
    families = [("Animals (subliminal trait)", ANIMALS),
                ("Constraints (legible rule)", CONSTRAINTS)]
    for sf, (ftitle, dss) in zip(subfigs, families):
        sf.suptitle(ftitle, fontsize=14, fontweight="bold")
        axs = sf.subplots(5, 2)
        rows = [("AVERAGE (n=4)", None)] + [(d, d) for d in dss]
        for r, (lab, ds) in enumerate(rows):
            vals, refs = load_avg(dss) if ds is None else load_one(ds)
            panel(axs[r][0], vals, refs, "beh", lab, ds is None)
            panel(axs[r][1], vals, refs, "nll", lab, ds is None)

    handles = [Patch(facecolor=c, edgecolor="black", label=f"{g} {lab}")
               for g, lab, _k, c in BARS]
    handles += [Line2D([0], [0], ls="--", color="gray", label="no-prompt floor"),
                Line2D([0], [0], ls="--", color="crimson", label="canonical (true-π)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=8, fontsize=9)

    fig.suptitle("SL prompt recovery: SALVE (naive vs beam) vs LARGO(lr)/GCG/OPRO  "
                 "(prefill-forced t=1; fixed token-space NLL; M_base Qwen2.5-7B; "
                 "train-selected, held-out val)", fontsize=12)
    p = OUT / "method_comparison.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
