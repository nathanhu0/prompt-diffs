"""Per-prompt DPO-margin plot for the hand-written prompt scoring experiment.

Reads <scores_dir>/scores.json (from score_prompts.py) and draws, per prompt:
left = mean per-token (dpo_norm) margin, middle = mean summed margin,
right = preference accuracy (margin > 0). Rows grouped sycophantic /
anti-sycophantic / neutral, "do X" vs "don't do X" marked by shape; the two
reference rows (empty system, stock OLMo-3 system = the DPO reference) are
dashed verticals.

Usage: python plot_prompt_scores.py <scores_dir> [--title "..."]
Output: <scores_dir>/prompt_scores.png (+ a copy next to this script named by
the scores dir).
"""
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP_ORDER = ["sycophantic", "anti_sycophantic", "neutral"]
GROUP_LABEL = {"sycophantic": "sycophantic", "anti_sycophantic": "anti-sycophantic", "neutral": "neutral"}
GROUP_COLOR = {"sycophantic": "#c0392b", "anti_sycophantic": "#2471a3", "neutral": "#7f8c8d"}
COLS = [("mean_margin_norm", "mean per-token margin (dpo_norm)"),
        ("mean_margin", "mean summed margin (nats)"),
        ("accuracy", "preference accuracy (margin > 0)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores_dir")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    d = json.loads((Path(args.scores_dir) / "scores.json").read_text())
    rows = d["rows"]
    refs = {r["name"]: r for r in rows if r["group"] == "reference"}
    body = [r for g in GROUP_ORDER for r in rows if r["group"] == g]
    y = {r["name"]: i for i, r in enumerate(reversed(body))}   # top = first

    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(body) + 1.6), sharey=True)
    for ax, (col, xlabel) in zip(axes, COLS):
        for r in body:
            marker = {"do": "o", "dont": "s"}.get(r.get("phrasing"), "D")
            ax.scatter(r[col], y[r["name"]], color=GROUP_COLOR[r["group"]], marker=marker, s=42, zorder=3)
        for name, ls in (("empty", "--"), ("default_olmo3", ":")):
            # the stock-system row has margin == 0 on every pair (it IS the
            # reference), so its accuracy is a degenerate 0 — omit it there
            if name in refs and not (col == "accuracy" and name == "default_olmo3"):
                ax.axvline(refs[name][col], color="k", ls=ls, lw=1, zorder=1)
        ax.set_xlabel(xlabel)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels([r["name"] for r in reversed(body)], fontsize=8)
    # legend inside the last axes: groups by color, phrasing by shape, references by line
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in GROUP_ORDER]
    handles += [Line2D([], [], marker="o", ls="", color="k", label="\"do X\""),
                Line2D([], [], marker="s", ls="", color="k", label="\"don't do X\""),
                Line2D([], [], ls="--", color="k", label="empty system prompt"),
                Line2D([], [], ls=":", color="k", label="stock OLMo-3 system (= reference)")]
    axes[2].legend(handles=handles, fontsize=7, loc="lower right", frameon=False)
    n = d["args"]["n"]
    fig.suptitle(args.title or f"Dolci delta_learning (Qwen3-32B chosen / 0.6B rejected): DPO margin under hand-written system prompts, Olmo-3-7B-Instruct-SFT, {n} held-out pairs", fontsize=9)
    fig.tight_layout()
    out = Path(args.scores_dir) / "prompt_scores.png"
    fig.savefig(out, dpi=150); fig.savefig(Path(__file__).parent / f"prompt_scores_{Path(args.scores_dir).name}.png", dpi=150)
    print("saved", out)
    # group summary for the console
    import statistics as st
    for g in GROUP_ORDER:
        rs = [r for r in body if r["group"] == g]
        print(f"{GROUP_LABEL[g]:17s} n={len(rs):2d}  per-token margin mean {st.mean(r['mean_margin_norm'] for r in rs):+.4f}  "
              f"summed margin mean {st.mean(r['mean_margin'] for r in rs):+.2f}  accuracy mean {st.mean(r['accuracy'] for r in rs):.3f}")
        for ph in ("do", "dont"):
            sub = [r for r in rs if r.get("phrasing") == ph]
            if sub: print(f"   {ph:4s} n={len(sub)}  per-token {st.mean(r['mean_margin_norm'] for r in sub):+.4f}  summed {st.mean(r['mean_margin'] for r in sub):+.2f}  acc {st.mean(r['accuracy'] for r in sub):.3f}")
    print("references:", {k: (round(v['mean_margin_norm'], 4), round(v['mean_margin'], 2), round(v['accuracy'], 3)) for k, v in refs.items()})


if __name__ == "__main__":
    main()
