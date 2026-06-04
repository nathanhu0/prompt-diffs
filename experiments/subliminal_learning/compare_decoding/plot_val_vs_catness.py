"""Master plot: val NLL (x) vs mean logP(cat) (y) for SL cat verbalization.

Two panes share the y axis — left = steered dataset, right = prompted dataset.
Each pane overlays every point family we've discussed:
  - anchors: soft prompt, base (no sysprompt), canonical "love cats" prompt
  - plain decodes at T=1.0 and T=0.7
  - contrastive decodes (alpha as below), pooled over both temps

Reads the decode_compare JSONs written by sample_score_decodes.py; missing
files (e.g. a contrastive run still on the queue) are skipped gracefully so
the plot can be regenerated as runs land. Pure CPU — run on the host venv:

  PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compare_decoding/plot_val_vs_catness.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
RES = Path("/nlp/scr/nathu/latent_rewrite/subliminal_learning")
RUNS = [("steered", RES / "steered_cat_e4_lr1e-3"),
        ("prompted", RES / "prompted_cat_e4_lr1e-3")]
# contrastive runs to overlay, as (alpha, color) — greens light→dark with alpha
ALPHAS = [(0.25, "#74c476"), (0.5, "#31a354"), (1.0, "#006d2c")]

# topic_alpha_sweep cat cells (system_top4 pool, T=0.7, greedy α-sweep) to merge
# in: their decodes carry full_val_nll + avg_log_likelihood on the same val
# split / eval prompts, so they drop straight onto these axes. magenta family
# (light→dark) keeps them distinct from the green contrastive pools.
SWEEP_RUNS = {"steered": RES / "topic_alpha_sweep" / "steered_cat",
              "prompted": RES / "topic_alpha_sweep" / "prompted_cat"}
SWEEP_ALPHAS = [("null", "#c994c7", "∅"), ("0.25", "#df65b0", "0.25"),
                ("0.5", "#dd1c77", "0.5"), ("1", "#980043", "1")]


def load(d, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, (cond, d) in zip(axes, RUNS):
    plain = load(d, "decode_compare.json")
    if plain is None:
        ax.set_title(f"{cond}: no decode_compare.json")
        continue

    # plain decode pools, split by sampling temperature
    for temp, color in [(1.0, "tab:blue"), (0.7, "tab:orange")]:
        pts = [(r["nll_val"], r["cat_logprob"]) for r in plain["decodes"]
               if r["temperature"] == temp]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=color, marker="o", alpha=0.75, s=45,
                       label=f"plain T={temp}")

    # contrastive decode pools (both temps together), one color per alpha
    for alpha, color in ALPHAS:
        contr = load(d, f"decode_compare_alpha{alpha}.json")
        if not contr:
            continue
        pts = [(r["nll_val"], r["cat_logprob"]) for r in contr["decodes"]]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=color, marker="^", alpha=0.85, s=50,
                       label=f"contrastive α={alpha}")

    # original greedy-recovery reps (ft_eval.json "decodes"): the 4-seed greedy
    # sentence search. x = full_val_nll, y = avg_log_likelihood (== our
    # cat_logprob; same _label_loglik over the same eval prompts). NLL here is
    # on the original run's full 500-val vs 50-val for the decode pools —
    # per-token means, so ~comparable on this axis.
    fte = load(d, "ft_eval.json")
    if fte and fte.get("decodes"):
        pts = [(r["full_val_nll"], r["avg_log_likelihood"])
               for r in fte["decodes"]
               if r.get("full_val_nll") is not None
               and r.get("avg_log_likelihood") is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c="tab:purple", marker="X", s=95, zorder=5,
                       edgecolors="white", linewidths=0.6,
                       label=f"greedy ({len(pts)} reps)")

    # topic_alpha_sweep cat greedy decodes (system_top4, T=0.7), per alpha. Same
    # (full_val_nll, avg_log_likelihood) axes as the greedy reps above.
    sd = SWEEP_RUNS.get(cond)
    if sd:
        for tag, color, lab in SWEEP_ALPHAS:
            de = load(sd / f"alpha_{tag}", "decodes_eval.json")
            if not de:
                continue
            pts = [(r["full_val_nll"], r["avg_log_likelihood"])
                   for r in de.get("decodes", [])
                   if r.get("full_val_nll") is not None
                   and r.get("avg_log_likelihood") is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, c=color, marker="P", s=80, zorder=4,
                           edgecolors="black", linewidths=0.5,
                           label=f"top4 α={lab}")

    # anchors
    for key, mk, color, lab, sz in [
            ("soft_ref",      "*", "black", "soft",      280),
            ("base_ref",      "s", "grey",  "base",       80),
            ("canonical_ref", "D", "red",   "canonical",  90)]:
        r = plain.get(key)
        if r:
            ax.scatter([r["nll_val"]], [r["cat_logprob"]], marker=mk, c=color,
                       s=sz, label=lab, zorder=6,
                       edgecolors="white", linewidths=0.6)

    ax.set_title(f"{cond} cat")
    ax.set_xlabel("val NLL (number completions)")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("mean logP(cat) over eval prompts")

# one merged legend (panes may differ on which families are present yet)
handles = {}
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(l, h)
fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=6,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Soft-prompt verbalization: val NLL vs catness  (cat, e4 lr1e-3)")
fig.tight_layout(rect=[0, 0.05, 1, 1])

out = OUT_DIR / "val_nll_vs_catness_master.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved → {out}")
