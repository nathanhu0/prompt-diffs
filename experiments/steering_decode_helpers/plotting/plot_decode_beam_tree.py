"""Beam-tree visualization for a decode-sweep config (model, method, pool, rp,
nrng) across the 4 seeds. Fork of
final_experiments/induction_methods/plotting/plot_beam_tree.py — same layout
(score-vs-depth tree + depth-1 verbalization listing + winner panel) but
pointed at decode_sweep/<slug>/ instead of the parent vanilla cell.

  python experiments/steering_decode_helpers/plotting/plot_decode_beam_tree.py \\
    --model llama --method steering --pool system --rp 1.2 --nrng 0
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from core.subliminal.animals import _SYN_SETS

RUNTIME = yaml.safe_load(open(REPO / "final_experiments/induction_methods/config.yaml"))
OUTPUT_ROOT = Path(RUNTIME["output_root"])
SEEDS = [42, 43, 44, 45]

MODEL_SHORT = {
    "qwen":  "Qwen2.5-7B-Instruct",
    "llama": "Llama-3.1-8B-Instruct",
}
MODEL_LABEL = {
    "Qwen2.5-7B-Instruct":  "Qwen2.5-7B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}


def system_pool_for(model_short):
    return "system_top4_llama" if "llama" in model_short.lower() else "system_top4"


def config_slug(pool, rp, nrng, model_short):
    pool_resolved = system_pool_for(model_short) if pool == "system" else "user"
    return f"{pool_resolved}_rp{rp:.1f}_nrng{nrng}"


def make_names_fn(animal):
    syns = _SYN_SETS[animal]
    def names_animal(t):
        return bool(syns & set(re.findall(r"[a-z]+", (t or "").lower())))
    return names_animal


def load_cell(root, seed, animal, slug):
    base = root / f"seed{seed}"
    decode_dir = base / "decode_sweep" / slug / "prefill_t1" / animal
    sb_path = decode_dir / "salve_beam.json"
    pt_path = decode_dir / "salve_beam_results.pt"
    se_path = base / "prefill_t1" / animal / "soft_eval.json"  # soft is shared
    bl_path = OUTPUT_ROOT / "baselines" / "prefill_t1" / animal / "baselines.json"
    if not sb_path.exists():
        return None
    sb = json.load(open(sb_path))
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    se = json.load(open(se_path)) if se_path.exists() else None
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


def plot_cell(ax_tree, ax_text, root, seed, animal, slug, names_fn):
    loaded = load_cell(root, seed, animal, slug)
    if loaded is None:
        ax_tree.text(0.5, 0.5, f"seed{seed}\n[no salve_beam.json yet]",
                     ha="center", va="center", fontsize=12, color="gray",
                     transform=ax_tree.transAxes)
        ax_tree.set_xticks([]); ax_tree.set_yticks([])
        ax_text.axis("off")
        return
    sb, pt, se, bl = loaded
    nodes = pt["nodes"]
    nodes_scored = [n for n in nodes if n.get("score") is not None]
    by_idx = {n["idx"]: n for n in nodes}

    soft_hit = se["behavior"]["hit_rate"] if se else float("nan")
    text_hit = sb["behavior"]["hit_rate"]
    test_nll = sb["nll"]["test"]

    for n in nodes_scored:
        p = n.get("parent")
        if p is None:
            continue
        par = by_idx.get(p)
        if par is None or par.get("score") is None:
            continue
        ax_tree.plot([par["depth"], n["depth"]], [par["score"], n["score"]],
                     color="lightgray", alpha=0.35, linewidth=0.4, zorder=1)

    others = [n for n in nodes_scored if not names_fn(n.get("text", ""))]
    ax_tree.scatter([n["depth"] for n in others], [n["score"] for n in others],
                    s=10, c="#555", alpha=0.55, zorder=2, edgecolors="none",
                    label=f"candidates (n={len(nodes_scored)})")

    cats = [n for n in nodes_scored if names_fn(n.get("text", ""))]
    if cats:
        ax_tree.scatter([n["depth"] for n in cats], [n["score"] for n in cats],
                        s=55, c="red", edgecolors="black", linewidth=0.6, zorder=5,
                        label=f"{animal}-mentioning (n={len(cats)})")

    wp = winning_path(nodes, sb["best_text"])
    if wp:
        ax_tree.plot([n["depth"] for n in wp], [n["score"] for n in wp],
                     color="#2ca02c", linewidth=2.2, zorder=4, alpha=0.95,
                     label="winning path")
        ax_tree.scatter([n["depth"] for n in wp], [n["score"] for n in wp],
                        s=45, c="#2ca02c", edgecolors="black", linewidth=0.6,
                        zorder=6)

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
    sec = ax_tree.secondary_yaxis("right", functions=(np.exp, np.log))
    sec.set_ylabel("perplexity", fontsize=8)
    sec.tick_params(labelsize=7)

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

    lines.append("")
    lines.append(f"{animal}-mentioning candidates anywhere in beam: {len(cats)} / {len(nodes_scored)}")

    ax_text.text(0.0, 1.0, "\n".join(lines), fontfamily="monospace",
                 fontsize=7.2, verticalalignment="top", transform=ax_text.transAxes)
    ax_text.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SHORT))
    ap.add_argument("--method", required=True, choices=["steering", "prompted"])
    ap.add_argument("--pool", required=True, choices=["system", "user"])
    ap.add_argument("--rp", type=float, required=True)
    ap.add_argument("--nrng", type=int, required=True)
    ap.add_argument("--animal", default="cat", choices=list(_SYN_SETS))
    args = ap.parse_args()

    model_short = MODEL_SHORT[args.model]
    root = OUTPUT_ROOT / model_short / args.method
    slug = config_slug(args.pool, args.rp, args.nrng, model_short)
    names_fn = make_names_fn(args.animal)
    out = HERE.parent / f"beam_tree_{args.model}_{args.method}_{args.animal}_{slug}.png"

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(len(SEEDS), 2, width_ratios=[1.0, 1.0],
                          hspace=0.55, wspace=0.10)
    for i, seed in enumerate(SEEDS):
        ax_tree = fig.add_subplot(gs[i, 0])
        ax_text = fig.add_subplot(gs[i, 1])
        plot_cell(ax_tree, ax_text, root, seed, args.animal, slug, names_fn)
    fig.suptitle(
        f"{MODEL_LABEL[model_short]}  ·  {args.method}  ·  {args.animal}  ·  "
        f"decode-sweep [{slug}] — beam-tree per seed",
        fontsize=13, y=0.995,
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
