"""Single cross-method comparison figure for the SL prompt-recovery experiment
(paper Exp 1), on the prefill-forced t=1 datasets scored with the FIXED
token-space NLL (data carries `completion_ids`; see project_nll_retokenization
_artifact). One config per method, one bar each:

    SALVE (ours, headline = greedy readout) | LARGO | GCG | OPRO (vanilla)

Each bar is the train-selected config (argmin train select-score — the paper's
selection rule), reported on held-out val. Per subplot references: gray dashed =
no-prompt floor, crimson dashed = canonical (true-pi). A method with no result
yet is drawn as a hatched "pending" slot so the layout is stable across reruns.

Layout: 4x4. Left two columns = animals (subliminal trait); right two = number
constraints (legible rule). Within each family: behavior (hit-rate, higher
better) | val NLL (lower better, zoomed). Each subplot is titled with its own
dataset, so the row pairing is purely cosmetic.

  PYTHONPATH=. uv run python \
    experiments/sl_optimizer_comparison/plotting/plot_comparison.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCR = Path("/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison")
OUT = Path(__file__).parent / "figures"
VARIANT = "prefill_t1"

ANIMALS = ["cat", "dog", "eagle", "owl"]                  # subliminal trait
CONSTRAINTS = ["even", "six_seven", "mult_5", "mult_3"]   # legible rule

# One config per method (frozen); SALVE headline = the greedy readout.
METHODS = [("SALVE", "#1f77b4"), ("LARGO", "#ff7f0e"),
           ("GCG", "#2ca02c"), ("OPRO", "#9467bd")]


def sel_score(d):
    e = d.get("extra") or {}
    return e.get("select_score", e.get("best_select_score"))


def best(records):
    """Train-selected record: argmin over the train select-score."""
    rs = [r for r in records if sel_score(r) is not None]
    return min(rs, key=sel_score) if rs else None


def load(ds):
    """-> (bars, refs). bars[method] = (hit_rate, val_nll) for finished methods;
    refs = no-prompt floor + canonical true-pi (both metrics)."""
    base = SCR / f"sweep_prefill_{ds}"
    recs, baselines = [], None
    for jf in base.glob(f"*/{VARIANT}/{ds}/*.json"):
        d = json.loads(jf.read_text())
        if jf.name == "baselines.json":
            baselines = d
        elif "method" in d:
            recs.append(d)
    by = {}
    for r in recs:
        by.setdefault(r["method"], []).append(r)

    def hv(r):
        return ((r.get("behavior") or {}).get("hit_rate"),
                (r.get("nll") or {}).get("val"))

    bars = {}
    g = best(by.get("salve_greedy", []))            # SALVE headline = greedy
    if g:
        bars["SALVE"] = hv(g)
    if best(by.get("largo", [])):
        bars["LARGO"] = hv(best(by["largo"]))
    gcg = best([r for t in by for r in by[t] if t.startswith("gcg")])
    if gcg:
        bars["GCG"] = hv(gcg)
    if best(by.get("opro", [])):
        bars["OPRO"] = hv(best(by["opro"]))

    refs = {}
    if baselines:
        np_, tp = baselines.get("no_prompt", {}), baselines.get("true_pi", {})
        refs = {"floor_beh": (np_.get("behavior") or {}).get("hit_rate"),
                "floor_nll": (np_.get("nll") or {}).get("val"),
                "canon_beh": (tp.get("behavior") or {}).get("hit_rate"),
                "canon_nll": (tp.get("nll") or {}).get("val")}
    return bars, refs


def subplot(ax, ds, metric, show_xlabels):
    bars, refs = load(ds)
    idx = 0 if metric == "beh" else 1
    xs = list(range(len(METHODS)))
    present = [(x, m, c, bars[m][idx]) for x, (m, c) in zip(xs, METHODS)
               if m in bars and bars[m][idx] is not None]

    if metric == "nll":                              # zoom to [min, max] of bars + refs
        vals = [v for *_, v in present]
        vals += [refs[k] for k in ("floor_nll", "canon_nll") if refs.get(k) is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        pad = max((hi - lo) * 0.15, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
    else:                                            # behavior is 0-based
        vals = [v for *_, v in present] + [refs.get("canon_beh") or 0]
        ax.set_ylim(0, max(1.0, max(vals) if vals else 1.0) * 1.08)
    y0, y1 = ax.get_ylim()

    for x, m, c, v in present:
        ax.bar(x, v - y0, bottom=y0, width=0.8, color=c, edgecolor="black", linewidth=0.4)
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    for x, (m, c) in zip(xs, METHODS):               # hatched placeholder for pending
        if m not in bars or bars[m][idx] is None:
            ax.bar(x, y1 - y0, bottom=y0, width=0.8, color="none",
                   edgecolor="0.75", hatch="///", linewidth=0.5)
            ax.text(x, (y0 + y1) / 2, "pending", rotation=90, ha="center",
                    va="center", fontsize=6, color="0.6")

    fk, ck = (("floor_nll", "canon_nll") if metric == "nll"
              else ("floor_beh", "canon_beh"))
    if refs.get(fk) is not None:
        ax.axhline(refs[fk], ls="--", color="gray", lw=1.0)
    if refs.get(ck) is not None:
        ax.axhline(refs[ck], ls="--", color="crimson", lw=1.0)

    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(xs)
    ax.set_xticklabels([m for m, _ in METHODS] if show_xlabels else [],
                       rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{ds} — {'hit-rate' if metric == 'beh' else 'val NLL'}", fontsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i in range(4):
        last = (i == 3)
        subplot(axes[i][0], ANIMALS[i], "beh", last)
        subplot(axes[i][1], ANIMALS[i], "nll", last)
        subplot(axes[i][2], CONSTRAINTS[i], "beh", last)
        subplot(axes[i][3], CONSTRAINTS[i], "nll", last)

    fig.text(0.30, 0.97, "Animals (subliminal trait)", ha="center",
             fontsize=13, fontweight="bold")
    fig.text(0.74, 0.97, "Constraints (legible rule)", ha="center",
             fontsize=13, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in METHODS]
    handles += [Line2D([0], [0], ls="--", color="gray"),
                Line2D([0], [0], ls="--", color="crimson")]
    labels = [m for m, _ in METHODS] + ["no-prompt floor", "canonical (true-π)"]
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("SL prompt recovery: SALVE vs baselines  "
                 "(prefill-forced t=1; fixed token-space NLL; M_base Qwen2.5-7B; "
                 "train-selected, held-out val)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    p = OUT / "method_comparison.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
