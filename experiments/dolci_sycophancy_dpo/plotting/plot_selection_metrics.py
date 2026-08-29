"""Does a selection metric pick data that transmits sycophancy?

One x position per selection metric, two points at it: the student trained on
the metric's TOP 15k rows and the one trained on its BOTTOM 15k. Both students
saw 15,000 rows and 118 optimizer steps under the same LoRA/beta/schedule, so a
metric that ranks pairs by how much sycophancy they teach should separate its
two points. Horizontal references: the base model (no DPO) and a plain random
15k draw.

The grey band is the empirical noise floor — two random draws matched to the
same ~105-token length profile came out 0.417 and 0.557, so a top-vs-bottom gap
narrower than that band is not evidence of anything. Each metric is annotated
with the median pair length of its top subset, because the length-normalized
metrics select ~100-token pairs at both ends while the unnormalized ones select
~1000-token pairs at both ends, and length is itself strongly related to
transfer.

Usage:
    PYTHONPATH=. uv run python experiments/dolci_sycophancy_dpo/plotting/plot_selection_metrics.py
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.dolci_sycophancy_dpo.score_split import load_prompt_scores

ROOT = Path("/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo")
OUT_DIR = Path(__file__).parent
BARE = ["wrong_ack", "correct_now", "i_think_wrong"]
N_TRIPLES = 124942

# x position -> (label, top arm dir, bottom arm dir)
METRICS = [
    ("P_syco\n÷(len_c+len_r)",        "keep15k_single_pairlen",   "bottom15k_single_pairlen"),
    ("switch_answer\n÷(len_c+len_r)", "keep15k_switch_pairlen",   "bottom15k_switch_pairlen"),
    ("P_non_syco\n÷(len_c+len_r)",    "keep15k_nonsyco_pairlen",  "bottom15k_nonsyco_pairlen"),
    ("P_syco − P_non_syco\n÷(len_c+len_r)", "keep15k_contrast_pairlen", "bottom15k_contrast_pairlen"),
    ("P_syco\nsummed, no norm",       "keep15k_single_nolen",     "bottom15k_single_nolen"),
    ("P_syco − P_non_syco\nsummed, no norm", "keep15k_contrast_nolen", "bottom15k_contrast_nolen"),
]
RANDOM_ARMS = ["random15k", "lenmatch15k_contrast_pairlen", "lenmatch15k_switch_pairlen"]


def bare_flip(arm):
    p = ROOT / "syco_eval_dpo" / arm / "summary.json"
    if not p.exists():
        return None
    (_, s), = json.load(open(p))["summary"].items()
    v = s["variants"]
    if not all(k in v for k in BARE):
        return None
    return float(np.mean([v[k]["flip_rate_given_correct"] for k in BARE]))


def median_len(arm, lengths):
    p = ROOT / "dpo" / arm / "subset.json"
    if not p.exists():
        return None
    return float(np.median(lengths[np.array(json.load(open(p))["indices"])]))


def main():
    d = load_prompt_scores(ROOT / "split_scores", "syco_defer", n_expected=N_TRIPLES)
    lengths = (d["len_chosen"] + d["len_rejected"]).numpy()
    base = json.load(open(ROOT / "syco_eval_matched" / "sycophantic" / "summary.json"))["summary"]["base"]
    base_flip = float(np.mean([base["variants"][k]["flip_rate_given_correct"] for k in BARE]))
    rnd = {a: bare_flip(a) for a in RANDOM_ARMS}
    lenmatched = [v for a, v in rnd.items() if a != "random15k" and v is not None]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    if len(lenmatched) >= 2:
        ax.axhspan(min(lenmatched), max(lenmatched), color="0.85", zorder=0,
                   label=f"random draws, length-matched (~105 tok): {min(lenmatched):.2f}–{max(lenmatched):.2f}")
    if rnd.get("random15k") is not None:
        ax.axhline(rnd["random15k"], color="0.35", lw=1.4, ls="--",
                   label=f"random 15k, unmatched ({rnd['random15k']:.3f})")
    ax.axhline(base_flip, color="0.15", lw=1.4, ls=":",
               label=f"base, no DPO ({base_flip:.3f})")

    missing = []
    for i, (label, top_arm, bot_arm) in enumerate(METRICS):
        t, b = bare_flip(top_arm), bare_flip(bot_arm)
        if t is not None and b is not None:
            ax.plot([i, i], [t, b], color="0.6", lw=1.2, zorder=2)
        for val, color, marker, name in [(t, "#c2703d", "o", "top 15k"), (b, "#3d6fc2", "s", "bottom 15k")]:
            if val is None:
                missing.append(f"{label.splitlines()[0]} {name}")
                continue
            ax.scatter([i], [val], color=color, marker=marker, s=90, zorder=3,
                       label=name if i == 0 else None)
            ax.annotate(f"{val:.3f}", (i, val), fontsize=8, xytext=(7, -3), textcoords="offset points")
        ml = median_len(top_arm, lengths)
        if ml is not None:
            ax.annotate(f"med len {ml:.0f}", (i, 0.055), ha="center", fontsize=7, color="0.4")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m[0] for m in METRICS], fontsize=8)
    ax.set_xlabel("selection metric  (prompt whose log-prob preference ranks the pairs, and its length normalization)")
    ax.set_ylabel("flip rate | turn-1 correct, bare-pushback variants")
    ax.set_ylim(0, 0.9)
    ax.set_xlim(-0.5, len(METRICS) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax.set_title("Does prompt-based selection pick the pairs that teach sycophancy?\n"
                 "Olmo-3-7B-Instruct-SFT + LoRA r64, β 5 dpo_norm, 15k rows / 118 steps per point",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "selection_metrics.png", dpi=170, bbox_inches="tight")
    print(f"saved → {OUT_DIR / 'selection_metrics.png'}")
    for label, top_arm, bot_arm in METRICS:
        t, b = bare_flip(top_arm), bare_flip(bot_arm)
        gap = f"{t - b:+.3f}" if (t is not None and b is not None) else "—"
        print(f"{label.replace(chr(10), ' '):42s} top {t if t is None else round(t,3)}  "
              f"bottom {b if b is None else round(b,3)}  gap {gap}")
    if missing:
        print("pending:", "; ".join(missing))


if __name__ == "__main__":
    main()
