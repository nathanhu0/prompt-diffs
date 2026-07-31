"""Beam-tree visualization per (model, method, animal) cell across the 4
default-config seeds (lr=3e-3, n_learnable=128). Each row = one seed:

  left:  score-vs-depth scatter of every scored beam node, edges parent->child,
         winning path highlighted, animal-mentioning candidates marked.
         Horizontal references: no-prompt NLL floor + canonical-prompt NLL.
  right: depth-1 verbalizations (raw samples from the soft prompt's verbalize step)
         with NLL + perplexity, and the final selected text.

Cells:  <OUTPUT_ROOT>/<model_short>/<method>/seed{42,43,44,45}/prefill_t1/<animal>/
Reads:  salve_beam.json (winner record), salve_beam_results.pt (beam nodes),
        soft_eval.json (soft-path behavior hit-rate), baselines.json (NLL refs).

CLI: --model {qwen,llama,olmo}  --method {prompted,filtered,steering,dpo,lora_teacher}
     --animal {cat,dog,eagle,owl}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
from core.subliminal.animals import _SYN_SETS

OUTPUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
SEEDS = [42, 43, 44, 45]

MODEL_SHORT = {
    "qwen":  "Qwen2.5-7B-Instruct",
    "llama": "Llama-3.1-8B-Instruct",
    "olmo":  "OLMo-2-1124-7B-Instruct",
}
MODEL_LABEL = {
    "Qwen2.5-7B-Instruct":  "Qwen2.5-7B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "OLMo-2-1124-7B-Instruct": "OLMo-2-7B",
}


def make_names_fn(animal):
    syns = _SYN_SETS[animal]
    def names_animal(t):
        return bool(syns & set(re.findall(r"[a-z]+", (t or "").lower())))
    return names_animal


def load_cell(root, seed, animal):
    cell = root / f"seed{seed}" / "prefill_t1" / animal
    sb = json.load(open(cell / "salve_beam.json"))
    pt = torch.load(cell / "salve_beam_results.pt", map_location="cpu", weights_only=False)
    se = json.load(open(cell / "soft_eval.json"))
    bl_path = root / "baselines" / "prefill_t1" / animal / "baselines.json"
    bl = json.load(open(bl_path)) if bl_path.exists() else None
    return sb, pt, se, bl


def winning_path(nodes, target_text):
    by_idx = {n["idx"]: n for n in nodes}
    win = next((n for n in nodes if n.get("text") == target_text), None)
    if win is None:
        return []
    path = [win]
    while path[-1].get("parent") is not None:
        nxt = by_idx.get(path[-1]["parent"])
        if nxt is None:
            break
        path.append(nxt)
    return list(reversed(path))


def plot_cell(ax_tree, ax_text, root, seed, animal, names_fn):
    sb, pt, se, bl = load_cell(root, seed, animal)
    nodes = pt["nodes"]
    nodes_scored = [n for n in nodes if n.get("score") is not None]
    by_idx = {n["idx"]: n for n in nodes}

    soft_hit = se["behavior"]["hit_rate"]
    text_hit = sb["behavior"]["hit_rate"]
    test_nll = sb["nll"]["test"]

    # Edges parent -> child (every scored node)
    for n in nodes_scored:
        p = n.get("parent")
        if p is None:
            continue
        par = by_idx.get(p)
        if par is None or par.get("score") is None:
            continue
        ax_tree.plot([par["depth"], n["depth"]], [par["score"], n["score"]],
                     color="lightgray", alpha=0.35, linewidth=0.4, zorder=1)

    # Non-cat candidates
    others = [n for n in nodes_scored if not names_fn(n.get("text", ""))]
    ax_tree.scatter([n["depth"] for n in others], [n["score"] for n in others],
                    s=10, c="#555", alpha=0.55, zorder=2, edgecolors="none",
                    label=f"candidates (n={len(nodes_scored)})")

    # Animal-mentioning candidates
    cats = [n for n in nodes_scored if names_fn(n.get("text", ""))]
    if cats:
        ax_tree.scatter([n["depth"] for n in cats], [n["score"] for n in cats],
                        s=55, c="red", edgecolors="black", linewidth=0.6, zorder=5,
                        label=f"{animal}-mentioning (n={len(cats)})")

    # Winning path
    wp = winning_path(nodes, sb["best_text"])
    if wp:
        ax_tree.plot([n["depth"] for n in wp], [n["score"] for n in wp],
                     color="#2ca02c", linewidth=2.2, zorder=4, alpha=0.95,
                     label="winning path")
        ax_tree.scatter([n["depth"] for n in wp], [n["score"] for n in wp],
                        s=45, c="#2ca02c", edgecolors="black", linewidth=0.6,
                        zorder=6)

    # NLL refs (baselines.json may be absent — e.g. DPO cells)
    if bl is not None:
        np_nll = bl["no_prompt"]["nll"]["val"]
        pi_nll = bl["true_pi"]["nll"]["val"]
        ax_tree.axhline(np_nll, color="#d62728", linestyle="--", linewidth=0.9,
                        label=f"no-prompt floor ({np_nll:.3f})")
        ax_tree.axhline(pi_nll, color="#1f77b4", linestyle="--", linewidth=0.9,
                        label=f"canonical π ({pi_nll:.3f})")

    ax_tree.set_xlabel("depth (sentences appended)")
    ax_tree.set_ylabel("select score = val NLL  (↓ better)")
    ax_tree.set_title(
        f"seed{seed}   soft hit-rate={soft_hit:.3f}   text hit-rate={text_hit:.3f}   test NLL={test_nll:.3f}",
        fontsize=10, loc="left",
    )
    ax_tree.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax_tree.grid(alpha=0.25)

    # Secondary axis: perplexity
    sec = ax_tree.secondary_yaxis("right", functions=(np.exp, np.log))
    sec.set_ylabel("perplexity", fontsize=8)
    sec.tick_params(labelsize=7)

    # Right text panel
    depth1 = sorted([n for n in nodes_scored if n.get("depth") == 1],
                    key=lambda n: n["score"])
    seen, lines = set(), []
    lines.append("Depth-1 verbalizations (soft prompt → text, top 8):")
    lines.append(f"{'':>2}{'NLL':>7}  {'ppl':>5}   text")
    lines.append("-" * 86)
    for n in depth1:
        s = n.get("sentence") or n.get("text") or ""
        if s in seen:
            continue
        seen.add(s)
        ppl = float(np.exp(n["score"]))
        mark = "* " if names_fn(s) else "  "
        lines.append(f"{mark}{n['score']:>6.3f}  {ppl:>5.2f}   {repr(s[:64])}")
        if len(seen) >= 8:
            break

    lines.append("")
    win_sel = pt["best_sel_score"]
    win_full = pt["best_full_val"]
    lines.append(f"WINNER  sel_NLL={win_sel:.3f} (ppl {np.exp(win_sel):.2f})   "
                 f"full_val_NLL={win_full:.3f} (ppl {np.exp(win_full):.2f}):")
    win_text = sb["best_text"]
    chunks = [win_text[i:i + 82] for i in range(0, min(len(win_text), 410), 82)]
    for c in chunks:
        lines.append(f"  {repr(c)}")

    n_cats = len(cats)
    n_total = len(nodes_scored)
    lines.append("")
    lines.append(f"{animal}-mentioning candidates anywhere in beam: {n_cats} / {n_total}")

    ax_text.text(0.0, 1.0, "\n".join(lines), fontfamily="monospace",
                 fontsize=7.2, verticalalignment="top", transform=ax_text.transAxes)
    ax_text.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SHORT))
    ap.add_argument("--method", required=True,
                    choices=["prompted", "filtered", "steering", "dpo", "lora_teacher"])
    ap.add_argument("--animal", required=True, choices=list(_SYN_SETS))
    args = ap.parse_args()

    model_short = MODEL_SHORT[args.model]
    root = OUTPUT_ROOT / model_short / args.method
    names_fn = make_names_fn(args.animal)
    out = HERE.parent / f"beam_tree_{args.model}_{args.method}_{args.animal}.png"

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(len(SEEDS), 2, width_ratios=[1.0, 1.0],
                          hspace=0.55, wspace=0.10)
    for i, seed in enumerate(SEEDS):
        ax_tree = fig.add_subplot(gs[i, 0])
        ax_text = fig.add_subplot(gs[i, 1])
        plot_cell(ax_tree, ax_text, root, seed, args.animal, names_fn)
    fig.suptitle(
        f"{MODEL_LABEL[model_short]}  ·  {args.method}  ·  {args.animal}"
        f" — beam-tree per seed  (default config: lr=3e-3, n_learnable=128)",
        fontsize=13, y=0.995,
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
