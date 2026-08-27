"""Plug-and-play prompt screen: which hand-written system prompts move the
paper's MMLU 2-turn flip rate on Olmo-3-7B-SFT.

Two columns of variants are kept apart because they behave differently. The
BARE-PUSHBACK variants (the user just objects) are the trustworthy readout: the
three neutral prompts, which say nothing about users or correctness, sit within
0.03 of base there. The AUTHORITY variants (the challenge cites an expert, a
textbook, most people, or research) are inflated by prompt style alone — "Be
concise" scores as high there as the best sycophancy prompt — so a prompt that
only rises in that column has shown nothing.

Reads every summary.json under <root>/syco_eval_*/*/ and writes the figure next
to this script.
"""
import json, glob, sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo")
OUT_DIR = Path(__file__).parent
BARE = ["wrong_ack", "correct_now", "i_think_wrong"]
AUTHORITY = ["expert_wrong", "textbook_wrong", "most_wrong", "research_wrong", "someone_wrong"]
NEUTRAL = {"neutral_helpful", "neutral_concise", "neutral_careful"}
# prompts whose wording names the eval's own answer-choice mechanics
EVAL_SPECIFIC = {"switch_letter", "keep_letter", "never_repeat", "always_repeat"}


def load():
    rows = {}
    for f in glob.glob(str(ROOT / "syco_eval_*" / "*" / "summary.json")):
        for name, d in json.load(open(f))["summary"].items():
            v = d["variants"]
            if not all(k in v for k in BARE + AUTHORITY):
                continue
            g = lambda ks: sum(v[k]["flip_rate_given_correct"] for k in ks) / len(ks)
            rows[name] = {"bare": g(BARE), "authority": g(AUTHORITY),
                          "acc": d["turn1_accuracy"], "text": d.get("text", "")}
    return rows


def color(name):
    if name == "base":
        return "0.25"
    if name in NEUTRAL:
        return "0.62"
    if name in EVAL_SPECIFIC:
        return "#c2703d"
    return "#3d6fc2"


def main():
    rows = load()
    order = sorted(rows, key=lambda n: rows[n]["bare"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.34 * len(order) + 1.6), sharey=True)
    y = range(len(order))
    floor = max(rows[n]["bare"] for n in NEUTRAL if n in rows)
    for ax, key, title in [(axes[0], "bare", "Bare pushback\n(user just objects)"),
                           (axes[1], "authority", "Authority appeal\n(expert / textbook / most people / research)")]:
        ax.barh(list(y), [rows[n][key] for n in order],
                color=[color(n) for n in order], height=0.72)
        if n := rows.get("base"):
            ax.axvline(n[key], color="0.25", lw=1, ls="--", zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("flip rate | turn-1 correct")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axvline(floor, color="0.62", lw=1, ls=":", zorder=0)
    axes[0].set_yticks(list(y)); axes[0].set_yticklabels(order, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#3d6fc2", "#c2703d", "0.62", "0.25"]]
    axes[1].legend(handles, ["general wording", "names the answer choices", "neutral prompt", "base (stock prompt)"],
                   fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("System prompts vs sycophantic answer-flipping — Olmo-3-7B-Instruct-SFT, 500 MMLU items",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prompt_flip_rates.png", dpi=170, bbox_inches="tight")
    print(f"saved → {OUT_DIR / 'prompt_flip_rates.png'}  ({len(order)} prompts)")
    print(f"neutral ceiling (bare) {floor:.3f}; base {rows['base']['bare']:.3f}")
    for n in reversed(order[-6:]):
        print(f"  {n:22s} bare {rows[n]['bare']:.3f}  authority {rows[n]['authority']:.3f}  t1acc {rows[n]['acc']:.3f}")


if __name__ == "__main__":
    main()
