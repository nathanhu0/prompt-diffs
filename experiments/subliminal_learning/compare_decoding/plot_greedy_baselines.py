"""Head-to-head: the two contrastive verbalization routes, under greedy search.

  ft−base   : (1+a)·logits_ft   − a·logits_base   (amplify finetune over base)
  soft−empty: (1+a)·logits_soft − a·logits_empty  (amplify soft prompt over empty)

Both are run as full greedy searches over a∈{.5,1,2,4}. This figure asks two
things the master plot can't show cleanly:

  1. Does either route reliably reach the trait? (color = method, shade = α)
  2. Does the val-NLL SELECTOR find it? Each rep is drawn faded; the val-NLL
     WINNER of its (method,α) cell is ringed black. When a high-catness rep is
     faded but the ringed winner sits low, the selector missed the trait.

Two panes (steered | prompted), shared y. References: soft plain greedy (blue),
and the ★ soft-z / ■ base / ◆ canonical anchors. Reads baseline_greedy.json,
soft_contrastive_greedy.json, decode_compare.json per run-dir. Pure CPU:

  PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compare_decoding/plot_greedy_baselines.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).parent
RES = Path("/nlp/scr/nathu/latent_rewrite/subliminal_learning")
RUNS = [("steered", RES / "steered_cat_e4_lr1e-3"),
        ("prompted", RES / "prompted_cat_e4_lr1e-3")]
GREEN = {0.5: "#a1d99b", 1.0: "#74c476", 2.0: "#31a354", 4.0: "#006d2c"}   # ft−base
PURPLE = {0.5: "#bcbddc", 1.0: "#9e9ac8", 2.0: "#807dba", 4.0: "#54278f"}  # soft−empty


def load(d, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def draw(ax, recs, shade, get_alpha):
    """One contrastive method: faded dot per rep, black ring on the val-NLL winner."""
    for r in recs:
        a = get_alpha(r)
        if a is None:
            continue
        c = shade.get(a, list(shade.values())[-1])
        if r["is_winner"]:
            ax.scatter([r["nll_val"]], [r["cat_logprob"]], marker="o", c=c, s=120,
                       zorder=5, edgecolors="black", linewidths=1.4)
        else:
            ax.scatter([r["nll_val"]], [r["cat_logprob"]], marker="o", c=c, s=42,
                       alpha=0.5, edgecolors="none")


fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
winners_by_cond = {}
for ax, (cond, d) in zip(axes, RUNS):
    bg = load(d, "baseline_greedy.json")
    sc = load(d, "soft_contrastive_greedy.json")
    dc = load(d, "decode_compare.json")

    # collect the ringed val-NLL winners (the 4 ft−base + 4 soft−empty contrast
    # cells) to print verbatim under the panel.
    wins = []
    if bg:
        for r in bg["greedy_baselines"]:
            if r["source"] == "ft_base_contrastive" and r["is_winner"]:
                wins.append(("ftb", r["contrastive_alpha"], r["nll_val"],
                             r["cat_logprob"], r["text"]))
    if sc:
        for r in sc["soft_greedy"]:
            if r["is_winner"] and r["contrastive_alpha"] is not None:
                wins.append(("soft", r["contrastive_alpha"], r["nll_val"],
                             r["cat_logprob"], r["text"]))
    winners_by_cond[cond] = sorted(wins, key=lambda x: (x[0], x[1]))

    if bg:
        draw(ax, [r for r in bg["greedy_baselines"]
                  if r["source"] == "ft_base_contrastive"],
             GREEN, lambda r: r["contrastive_alpha"])
    if sc:
        draw(ax, sc["soft_greedy"], PURPLE, lambda r: r["contrastive_alpha"])
        # soft plain greedy (a=null) as a blue reference
        plain = [(r["nll_val"], r["cat_logprob"]) for r in sc["soft_greedy"]
                 if r["contrastive_alpha"] is None]
        if plain:
            xs, ys = zip(*plain)
            ax.scatter(xs, ys, marker="P", c="tab:blue", s=70, alpha=0.8,
                       edgecolors="white", linewidths=0.5, zorder=4)

    if dc:
        for key, mk, color, sz in [("soft_ref", "*", "black", 320),
                                   ("base_ref", "s", "dimgray", 90),
                                   ("canonical_ref", "D", "red", 95)]:
            r = dc.get(key)
            if r:
                ax.scatter([r["nll_val"]], [r["cat_logprob"]], marker=mk, c=color,
                           s=sz, zorder=7, edgecolors="white", linewidths=0.7)

    ax.set_title(f"{cond} cat")
    ax.set_xlabel("val NLL (number completions)")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("mean logP(cat) over eval prompts")

legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN[2.0], markersize=11,
           label="ft−base contrast (α .5→4 shade)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE[2.0], markersize=11,
           label="soft−empty contrast (α .5→4 shade)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markersize=13,
           markeredgecolor="black", markeredgewidth=1.4, label="val-NLL winner (ringed)"),
    Line2D([0], [0], marker="P", color="w", markerfacecolor="tab:blue", markersize=11,
           label="soft plain greedy"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=15,
           label="soft z (anchor)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="red", markersize=10,
           label="canonical"),
]
# scatter panes occupy the top; reserve the bottom half for the winner prompts.
fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.50, wspace=0.05)
fig.legend(handles=legend, loc="center", ncol=6, fontsize=8.5, frameon=True,
           bbox_to_anchor=(0.5, 0.47))
fig.suptitle("Contrastive verbalization under greedy search: soft−empty vs ft−base "
             "(cat, e4 lr1e-3, T=0.7)")

# ---- verbatim val-NLL winner prompts, one column per condition ----
def block(cond):
    lines = [f"{cond} cat — val-NLL winners (ringed), 4 ft−base + 4 soft−empty:"]
    for m, a, nll, clp, txt in winners_by_cond.get(cond, []):
        t = " ".join(txt.split())[:64]
        mark = "►" if clp >= -2.5 else " "   # clearly cat-inducing
        lines.append(f"{mark} {m:4s} α{a:<3} nll{nll:.2f} c{clp:+.1f}: {t}")
    return "\n".join(lines)

for x0, cond in [(0.065, "steered"), (0.535, "prompted")]:
    fig.text(x0, 0.43, block(cond), family="monospace", fontsize=7.6,
             va="top", ha="left", linespacing=1.5)
fig.text(0.065, 0.015, "► = catlp ≥ −2.5 (clearly cat-inducing).  Most winners are "
         "generic/degenerate — the coherent cat prompts were non-winning reps "
         "(selection gap).", fontsize=8, style="italic")

out = OUT_DIR / "greedy_baselines.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved → {out}")
