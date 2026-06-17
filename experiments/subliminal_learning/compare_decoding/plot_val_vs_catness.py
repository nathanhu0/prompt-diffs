"""Master plot, simplified to be about SOURCE (temp fixed at 0.7).

Every point is a candidate system prompt scored as a base-model system prompt:
x = val NLL on the number completions, y = mean logP(cat). The only distinctions
kept are:

  color = SOURCE  — where the candidate came from
      soft      (blue)   : verbalized from the trained soft prompt z
      base      (grey)   : empty soft slot through base (the "default" floor)
      finetune  (orange) : empty slot through the vanilla SFT model M_ft
      ft−base   (green)  : (1+a)·ft − a·base contrastive steering, α = shade
  marker = METHOD — ○ single-shot verbalization · ✕ greedy sentence-search
  anchors          — ★ soft prompt (z itself) · ■ base (no prompt) · ◆ canonical

Two panes (steered | prompted), shared y. Reads, per run-dir:
  decode_compare.json  (soft single-shot @ T=0.7 + the three anchors)
  ft_eval.json         (soft greedy reps)
  baseline_decodes.json / baseline_greedy.json  (base / finetune / ft−base)
Missing files are skipped so the plot regenerates as runs land. Pure CPU:

  PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compare_decoding/plot_val_vs_catness.py
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

SRC_COLOR = {"soft": "tab:blue", "base": "tab:gray", "finetune": "tab:orange"}
# ft−base contrastive: one green shade per alpha (light→dark), covering both the
# single-shot grid (0.25/0.5/1.0) and the greedy grid (0.5/1.0/2.0/4.0).
FTBASE_SHADE = {0.25: "#c7e9c0", 0.5: "#a1d99b", 1.0: "#74c476",
                2.0: "#31a354", 4.0: "#006d2c"}
SS, GR = "o", "X"   # single-shot / greedy markers


def load(d, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def scat(ax, pts, marker, color, size, **kw):
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, marker=marker, c=color, s=size, **kw)


fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, (cond, d) in zip(axes, RUNS):
    dc = load(d, "decode_compare.json")
    bd = load(d, "baseline_decodes.json")
    bg = load(d, "baseline_greedy.json")
    fte = load(d, "ft_eval.json")
    if dc is None:
        ax.set_title(f"{cond}: no decode_compare.json")
        continue

    # ---- soft (blue): single-shot @ T=0.7, greedy reps ----
    scat(ax, [(r["nll_val"], r["cat_logprob"]) for r in dc["decodes"]
              if r["temperature"] == 0.7],
         SS, SRC_COLOR["soft"], 42, alpha=0.55, edgecolors="none")
    if fte and fte.get("decodes"):
        scat(ax, [(r["full_val_nll"], r["avg_log_likelihood"]) for r in fte["decodes"]
                  if r.get("full_val_nll") is not None
                  and r.get("avg_log_likelihood") is not None],
             GR, SRC_COLOR["soft"], 80, alpha=0.9, edgecolors="white", linewidths=0.6)

    # ---- base / finetune (grey / orange): single-shot + greedy ----
    for src in ("base", "finetune"):
        tag = "base_empty" if src == "base" else "finetune"
        if bd:
            scat(ax, [(r["nll_val"], r["cat_logprob"]) for r in bd["baseline_decodes"]
                      if r["source"] == tag],
                 SS, SRC_COLOR[src], 42, alpha=0.55, edgecolors="none")
        if bg:
            scat(ax, [(r["nll_val"], r["cat_logprob"]) for r in bg["greedy_baselines"]
                      if r["source"] == tag],
                 GR, SRC_COLOR[src], 80, alpha=0.9, edgecolors="white", linewidths=0.6)

    # ---- ft−base contrastive (green, α = shade): single-shot + greedy ----
    def ftbase(recs, key_nll, key_cat):
        for r in recs:
            if r["source"] != "ft_base_contrastive":
                continue
            yield (r[key_nll], r[key_cat], FTBASE_SHADE.get(r["contrastive_alpha"], "#006d2c"))
    if bd:
        for x, y, c in ftbase(bd["baseline_decodes"], "nll_val", "cat_logprob"):
            ax.scatter([x], [y], marker=SS, c=c, s=46, alpha=0.7, edgecolors="none")
    if bg:
        for x, y, c in ftbase(bg["greedy_baselines"], "nll_val", "cat_logprob"):
            ax.scatter([x], [y], marker=GR, c=c, s=84, alpha=0.95,
                       edgecolors="white", linewidths=0.6)

    # ---- anchors ----
    for key, mk, color, sz in [("soft_ref", "*", "black", 300),
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

# ---- single shared legend: source colors, method markers, anchors ----
def patch(color, label):
    return Line2D([0], [0], marker="s", color="w", markerfacecolor=color,
                  markersize=11, label=label)
legend = [
    patch(SRC_COLOR["soft"], "soft prompt"),
    patch(SRC_COLOR["base"], "base (default)"),
    patch(SRC_COLOR["finetune"], "finetune"),
    patch(FTBASE_SHADE[1.0], "ft−base contrast (α .25→4 light→dark)"),
    Line2D([0], [0], marker=SS, color="w", markerfacecolor="0.4", markersize=9,
           label="○ single-shot"),
    Line2D([0], [0], marker=GR, color="w", markerfacecolor="0.4", markersize=10,
           label="✕ greedy search"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=15,
           label="soft z (anchor)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="dimgray", markersize=10,
           label="base anchor"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="red", markersize=10,
           label="canonical"),
]
fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9, frameon=True,
           bbox_to_anchor=(0.5, -0.05))
fig.suptitle("val NLL vs catness by source  (cat, e4 lr1e-3, T=0.7)")
fig.tight_layout(rect=[0, 0.07, 1, 1])

out = OUT_DIR / "val_nll_vs_catness_master.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved → {out}")
