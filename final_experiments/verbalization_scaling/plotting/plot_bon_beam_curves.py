"""Unified BoN-curve + beam-points figure, per soft prompt or aggregated.

  --metric select : exact unbiased E[best-of-N] from the pool's select scores,
                    DENSE N grid (smooth by construction).
  --metric val    : exact E[val of winner] from exact_bon_val.py's val-by-rank
                    sidecar, re-gridded densely (N >= 16).
  --seed <N>      : one soft prompt, raw metric scale; reference hlines for the
                    canonical prompt and the soft prompt itself (the skyline
                    the readout decodes from).
  --seed all      : the aggregate: select -> per-seed excess over canonical
                    (log), val -> raw shared scale. Thin per-seed curves + bold
                    mean; beam points mean ± min-max. Soft-prompt reference =
                    mean across seeds (dashed), canonical = line (val) / axis
                    floor note (select-excess).
  --seed grid     : 2x3 panel grid — one raw-scale panel per seed plus the
                    aggregate panel bottom-right.

Only the two fixed-branching beam families (×16, ×8) are drawn; the light
1×2/1×4 probe arms are omitted.

  PYTHONPATH=. uv run python final_experiments/verbalization_scaling/plotting/plot_bon_beam_curves.py --metric select --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)
from final_experiments.verbalization_scaling.plotting._load import (
    SCR, load_beam_arm, load_bon_arm, BEAM_ARMS_X16, BEAM_ARMS_X8)
apply_style()

OUT_DIR = Path(__file__).parent
C_X16, C_X8, C_BON = "#3182bd", "#e6550d", "0.35"
ALL_SEEDS = [42, 43, 44, 45, 46]


def cell_dir(seed, task):
    return SCR / f"seed{seed}" / "readout" / "filtered_schrodi" / task


def refs_of(seed, task):
    p = cell_dir(seed, task) / "canonical_select.json"
    return json.loads(p.read_text()) if p.exists() else {}


def hyp_logw(n, N, m):
    r = np.arange(m)
    valid = (n - 1 - r) >= (N - 1)
    logw = np.full(m, -np.inf)
    logw[valid] = (gammaln(n - r[valid]) - gammaln(N) - gammaln(n - r[valid] - N + 1)
                   - (gammaln(n + 1) - gammaln(N + 1) - gammaln(n - N + 1)))
    return logw


def select_curve(seed, task, n_grid):
    bon = load_bon_arm(seed, task=task)
    if bon is None:
        return None
    s = np.array([x["score"] for x in bon["samples"]], dtype=float)
    if not np.isfinite(s).all():
        # empty generations were streamed with score=inf; as a returned prompt
        # they ARE the empty prompt — score them at its measured select NLL
        # (seed46: 141/1536 empties; without this the estimator is inf/NaN).
        empty = refs_of(seed, task).get("empty", {}).get("select")
        if empty is not None:
            s[~np.isfinite(s)] = empty
        s = s[np.isfinite(s)]
    s = np.sort(s)
    n = len(s)
    out = []
    for N in n_grid:
        if N > n:
            break
        w = np.exp(hyp_logw(n, N, n))
        out.append(float((w * s).sum()))
    return np.array(out)


def val_curve(seed, task, n_grid):
    p = cell_dir(seed, task) / "readout_best_of_1536_exact_val.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    m = d["m"]
    vals = np.array([d["val_by_rank"][str(r)] for r in range(m)])
    n = d.get("n", 1536)
    out = []
    for N in n_grid:
        if N > n:
            break
        w = np.exp(hyp_logw(n, N, m))
        out.append(float((w * vals).sum() / w.sum()))
    return np.array(out)


def beam_endpoint(seed, arm, task, metric):
    rec = load_beam_arm(seed, arm, task=task)
    if rec is None:
        return None
    n = rec["trajectory"][-1][1]
    if metric == "select":
        return n, rec["trajectory"][-1][2]
    p = cell_dir(seed, task) / f"readout_{arm}_incumbents.jsonl"
    if not p.exists():
        return None
    return n, json.loads(p.read_text().splitlines()[-1])["val"]


def draw_panel(ax, seeds, combined, metric, task, n_grid, legend=True):
    curve_fn, ref_key = ((select_curve, "select") if metric == "select"
                         else (val_curve, "val"))
    curves, soft_refs, canon_refs = [], [], []
    for seed in seeds:
        c = curve_fn(seed, task, n_grid)
        refs = refs_of(seed, task)
        if c is None or not refs:
            continue
        canon = refs["canonical"][ref_key]
        soft = refs.get("soft", {}).get(ref_key)
        # combined view: per-seed verbalization gap (offset by the seed's own
        # soft-prompt NLL) so z-quality spread — shared by both methods —
        # doesn't inflate the bars; per-seed panels stay raw.
        off = soft if (combined and soft is not None) else 0.0
        curves.append(c[: len(n_grid)] - off)
        canon_refs.append(canon - off)
        if soft is not None:
            soft_refs.append(soft - off)
        if combined:
            ax.plot(n_grid[: len(c)], c - off, color=C_BON, lw=0.8, alpha=0.3,
                    zorder=2)
    L = min(len(c) for c in curves)
    mean = np.mean([c[:L] for c in curves], axis=0)
    ax.plot(n_grid[:L], mean, color="0.15" if combined else C_BON, lw=2.2,
            zorder=3, label=("best-of-N (mean of %d soft prompts)" % len(curves)
                             if combined else "best-of-N (exact)"))

    for arms, color, label, mk in ((BEAM_ARMS_X16, C_X16, "beam ×16", "o"),
                                   (BEAM_ARMS_X8, C_X8, "beam ×8", "s")):
        xs, ys, ylo, yhi = [], [], [], []
        for arm in arms:
            pts = []
            for seed in seeds:
                got = beam_endpoint(seed, arm, task, metric)
                if got is None:
                    continue
                soft = refs_of(seed, task).get("soft", {}).get(ref_key)
                off = soft if (combined and soft is not None) else 0.0
                pts.append((got[0], got[1] - off))
            if not pts:
                continue
            es = [e for _, e in pts]
            xs.append(np.mean([n for n, _ in pts])); ys.append(np.mean(es))
            ylo.append(np.mean(es) - min(es)); yhi.append(max(es) - np.mean(es))
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[ylo, yhi] if combined else None, fmt=mk, ms=7,
                    color=color, markerfacecolor=color, markeredgecolor=color,
                    mew=1.4, ecolor=color, elinewidth=1.1, capsize=2.5,
                    ls="none", zorder=4, label=label)

    # Reference lines: canonical + soft prompt (skyline). In combined view the
    # soft prompt IS the anchor (y = 0); canonical lands at mean(canon − soft).
    y_canon = np.mean(canon_refs)
    if not (combined and metric == "select" and y_canon <= 0):
        ax.axhline(y_canon, color="0.5", lw=0.9, ls=":", zorder=1)
        ax.annotate("canonical prompt" + (" (mean vs soft)" if combined else ""),
                    xy=(1.0, y_canon), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points", ha="right",
                    fontsize=9, color="0.4")
    if soft_refs:
        y = np.mean(soft_refs)
        if combined and metric == "select":
            # anchor is 0 — the log-excess axis floor; state it.
            ax.annotate("soft prompt = 0 (axis floor)", xy=(0.02, 0.02),
                        xycoords="axes fraction", fontsize=9, color="#31a354")
        else:
            ax.axhline(y, color="#31a354", lw=1.0, ls="--", zorder=1)
            ax.annotate("soft prompt (anchor)" if combined else "soft prompt",
                        xy=(1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right",
                        fontsize=9, color="#31a354")

    ax.set_xscale("log")
    if combined and metric == "select":
        ax.set_yscale("log")
        ax.set_ylabel("select-256 NLL − soft prompt, per seed (log)")
    elif combined:
        ax.set_ylabel("held-out val NLL − soft prompt, per seed")
    else:
        ax.set_ylabel(("select-256" if metric == "select" else
                       "held-out val") + " NLL of returned prompt")
    ax.set_xlabel("candidates scored (log)")
    if legend:
        ax.legend(fontsize=8.5, loc="upper right")
    return len(curves)


def grid_for(metric):
    # Grids run to the extended-pool max (4608); seeds whose pool stops at
    # 1536 simply truncate there (curve_fn breaks at N > n), and the combined
    # mean is truncated to the shortest curve.
    if metric == "select":
        return np.unique(np.round(np.logspace(0, np.log10(4608), 66)).astype(int))
    return np.unique(np.round(np.logspace(np.log10(16), np.log10(4608),
                                          56)).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["select", "val", "both"], required=True)
    ap.add_argument("--seed", default="42", help="a seed number, 'all', or 'grid'")
    ap.add_argument("--task", default="cat")
    args = ap.parse_args()

    if args.metric == "both":
        # side-by-side select | val panels for one seed or the aggregate
        assert args.seed != "grid", "--metric both supports a seed or 'all'"
        combined = args.seed == "all"
        seeds = ALL_SEEDS if combined else [int(args.seed)]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        for i, (ax, metric) in enumerate(zip(axes, ["select", "val"])):
            n = draw_panel(ax, seeds, combined, metric, args.task,
                           grid_for(metric), legend=(i == 0))
            ax.set_title(metric)
        tag = "all %d soft prompts" % n if combined else f"seed {seeds[0]}"
        fig.suptitle(f"beam vs best-of-N ({args.task}, {tag})", y=0.99)
        fig.tight_layout()
        stem = OUT_DIR / (f"curves_both_{args.task}_"
                          + ("all" if combined else f"seed{seeds[0]}"))
        savefig_pair(fig, stem)
        print(f"wrote {stem}.png")
        return

    n_grid = grid_for(args.metric)

    if args.seed == "grid":
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
        for i, seed in enumerate(ALL_SEEDS):
            draw_panel(axes.flat[i], [seed], False, args.metric, args.task,
                       n_grid, legend=(i == 0))
            axes.flat[i].set_title(f"seed {seed}")
        n = draw_panel(axes.flat[5], ALL_SEEDS, True, args.metric, args.task,
                       n_grid, legend=False)
        axes.flat[5].set_title(f"mean of {n} soft prompts")
        fig.suptitle(f"{args.metric} — beam vs best-of-N ({args.task})", y=0.995)
        fig.tight_layout()
        stem = OUT_DIR / f"curves_{args.metric}_{args.task}_grid"
    else:
        combined = args.seed == "all"
        seeds = ALL_SEEDS if combined else [int(args.seed)]
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        n = draw_panel(ax, seeds, combined, args.metric, args.task, n_grid)
        tag = "all %d soft prompts" % n if combined else f"seed {seeds[0]}"
        ax.set_title(f"{args.metric} — beam vs best-of-N ({args.task}, {tag})")
        stem = OUT_DIR / (f"curves_{args.metric}_{args.task}_"
                          + ("all" if combined else f"seed{seeds[0]}"))
    savefig_pair(fig, stem)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
