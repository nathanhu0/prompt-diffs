"""Effect size of each way of choosing 15k Dolci rows for DPO.

Every arm is 15,000 rows, 118 optimizer steps, the same LoRA/β/schedule; only
the row-selection rule differs. Bars are the bare-pushback flip rate (the
variants where the user merely objects; the authority-appeal variants are
inflated by prompt style alone, see plot_prompt_flip_rates.py). Each bar is
annotated with the subset's median pair length in tokens, because the
length-normalized rankings put SHORT pairs at both extremes and the
unnormalized ones put LONG pairs at both, so length is the confound every
comparison has to be read against.

Reads <root>/syco_eval_dpo/<arm>/summary.json and <root>/dpo/<arm>/subset.json
(row indices -> lengths from the score files). Writes the figure next to this
script. Arms whose eval has not finished are listed, not plotted.
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
AUTHORITY = ["expert_wrong", "textbook_wrong", "most_wrong", "research_wrong", "someone_wrong"]
BASE_SUMMARY = ROOT / "syco_eval_matched" / "sycophantic" / "summary.json"
N_TRIPLES = 124942

# arm dir -> (label, colour group)
LABELS = {
    "random15k":                  ("random",                              "random"),
    "lenmatch15k_contrast_pairlen": ("random, length-matched to top ÷len",  "random"),
    "keep15k_single_pairlen":     ("top: P_syco, ÷len",                   "top"),
    "keep15k_contrast_pairlen":   ("top: P_syco − P_non, ÷len",           "top"),
    "keep15k_single_nolen":       ("top: P_syco, no norm",                "top"),
    "keep15k_contrast_nolen":     ("top: P_syco − P_non, no norm",        "top"),
    "keep15k_nonsyco_pairlen":    ("top: P_non_syco alone, ÷len",         "neg"),
    "keep15k_switch_pairlen":     ("top: switch_answer alone, ÷len",      "top"),
    "lenmatch15k_switch_pairlen": ("random, length-matched to switch ÷len", "random"),
    "bottom15k_contrast_pairlen": ("bottom: P_syco − P_non, ÷len",        "bottom"),
    "bottom15k_contrast_nolen":   ("bottom: P_syco − P_non, no norm",     "bottom"),
}
COLOR = {"random": "0.55", "top": "#c2703d", "bottom": "#3d6fc2", "neg": "#7a5cc2", "base": "0.2"}


def flip(summary_path):
    d = json.load(open(summary_path))["summary"]
    (name, s), = d.items()
    v = s["variants"]
    if not all(k in v for k in BARE + AUTHORITY):
        return None
    g = lambda ks: float(np.mean([v[k]["flip_rate_given_correct"] for k in ks]))
    return {"bare": g(BARE), "authority": g(AUTHORITY), "acc": s["turn1_accuracy"]}


def main():
    lengths = None
    rows, pending = [], []
    for arm, (label, group) in LABELS.items():
        summ = ROOT / "syco_eval_dpo" / arm / "summary.json"
        r = flip(summ) if summ.exists() else None
        if r is None:
            pending.append(label); continue
        sub = ROOT / "dpo" / arm / "subset.json"
        if sub.exists():
            if lengths is None:
                d = load_prompt_scores(ROOT / "split_scores", "syco_defer", n_expected=N_TRIPLES)
                lengths = (d["len_chosen"] + d["len_rejected"]).numpy()
            idx = np.array(json.load(open(sub))["indices"])
            r["median_len"] = float(np.median(lengths[idx]))
        rows.append({"label": label, "group": group, **r})
    b = json.load(open(BASE_SUMMARY))["summary"]["base"]
    vb = b["variants"]
    rows.append({"label": "base (SFT, no DPO)", "group": "base",
                 "bare": float(np.mean([vb[k]["flip_rate_given_correct"] for k in BARE])),
                 "authority": float(np.mean([vb[k]["flip_rate_given_correct"] for k in AUTHORITY])),
                 "acc": b["turn1_accuracy"], "median_len": float("nan")})
    rows.sort(key=lambda r: r["bare"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 0.42 * len(rows) + 1.4),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(rows))
    ax.barh(y, [r["bare"] for r in rows], color=[COLOR[r["group"]] for r in rows], height=0.7)
    for yi, r in zip(y, rows):
        note = f"{r['bare']:.3f}" + (f"   med len {r['median_len']:.0f}" if r["median_len"] == r["median_len"] else "")
        ax.text(r["bare"] + 0.01, yi, note, va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlabel("flip rate | turn-1 correct, bare-pushback variants")
    ax.set_xlim(0, max(r["bare"] for r in rows) + 0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("by selection rule", fontsize=10)

    # Right: the same arms against the median length of the pairs they trained
    # on. The ranked subsets sit on one curve with the random ones, which is the
    # evidence that what the rankings selected was length.
    pts = [r for r in rows if r["median_len"] == r["median_len"]]
    ax2.scatter([r["median_len"] for r in pts], [r["bare"] for r in pts],
                c=[COLOR[r["group"]] for r in pts], s=55, zorder=3)
    for r in pts:
        ax2.annotate(r["label"].replace("P_syco − P_non", "contrast").replace("P_syco", "syco"),
                     (r["median_len"], r["bare"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax2.axhline(rows[0]["bare"] if rows[0]["group"] == "base" else 0.122, color="0.2", lw=0.8, ls="--")
    ax2.set_xscale("log"); ax2.set_xlabel("median pair length of the training subset (tokens)")
    ax2.set_ylabel("flip rate | turn-1 correct, bare-pushback")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("against subset length", fontsize=10)
    fig.suptitle("DPO on 15k Dolci delta_learning rows — Olmo-3-7B-Instruct-SFT + LoRA r64, β 5 dpo_norm, 118 steps each",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dpo_arms_15k.png", dpi=170, bbox_inches="tight")
    print(f"saved → {OUT_DIR / 'dpo_arms_15k.png'}")
    print(f"{'arm':40s} {'bare':>6s} {'auth':>6s} {'t1acc':>6s} {'medlen':>7s}")
    for r in reversed(rows):
        print(f"{r['label']:40s} {r['bare']:6.3f} {r['authority']:6.3f} {r['acc']:6.3f} {r['median_len']:7.0f}")
    if pending:
        print("pending:", "; ".join(pending))


if __name__ == "__main__":
    main()
