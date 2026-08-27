"""Steering-strength tradeoff figure: per (model, animal), preference
(behavior hit-rate) and number coherence vs alpha, from the saved sweep JSONs
(claude_scripts/sweep_steering_strength_tradeoff.py). Coherence is shown two
ways: the format keep-rate measured in the sweep, and a diversity-corrected
curve re-scored offline from the saved generations — the mean within-row
unique-number fraction over rows whose leading span is a valid list, exposing
repetition spam that passes the format filter (e.g. qwen eagle at high alpha).
Vertical dotted line = production alpha for that grid point.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")
OUT_DIR = Path(__file__).parent
MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Olmo-3-7B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
PROD_ALPHA = {"Qwen2.5-7B-Instruct": {"cat": 4, "dog": 4, "eagle": 4, "owl": 4},
              "Llama-3.1-8B-Instruct": {"cat": 2, "dog": 2, "eagle": 4, "owl": 4},
              "Olmo-3-7B-Instruct": {"cat": 2, "dog": 2, "eagle": 4, "owl": 4}}


def leading_numbers(raw):
    m = re.match(r"[\s\d,]*", raw)
    return re.findall(r"\d{3}", m.group(0)) if m else []


def kept_diversity(rows):
    """Mean within-row unique fraction over rows with a valid leading list."""
    fracs = [len(set(ns)) / len(ns)
             for r in rows for ns in [leading_numbers(r["raw_completion"])]
             if len(ns) >= 5]
    return sum(fracs) / len(fracs) if fracs else 0.0


fig, axes = plt.subplots(len(MODELS), len(ANIMALS), figsize=(13, 8),
                         sharex=True, sharey=True)
for i, model in enumerate(MODELS):
    data = json.load(open(DATA_DIR / f"steering_strength_tradeoff_{model}.json"))
    for j, animal in enumerate(ANIMALS):
        ax = axes[i][j]
        rows = data["animals"][animal]
        alphas = [r["alpha"] for r in rows]
        ax.plot(alphas, [r["behavior_hit_rate"] for r in rows], "o-",
                color="tab:green", markersize=4, label="animal preference")
        ax.plot(alphas, [r["keep_rate"] for r in rows], "s--",
                color="tab:blue", markersize=4, label="format keep-rate")
        ax.plot(alphas, [kept_diversity(r["number_generations"]) for r in rows],
                "^:", color="tab:red", markersize=4, label="kept-row diversity")
        ax.axvline(PROD_ALPHA[model][animal], color="gray", ls=":", lw=1)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_ylim(-0.03, 1.03)
        if i == 0:
            ax.set_title(animal)
        if j == 0:
            ax.set_ylabel(f"{model}\nrate")
        if i == len(MODELS) - 1:
            ax.set_xlabel("alpha (raw-vector multiplier)")
axes[0][0].legend(fontsize=7, loc="center left", frameon=False)
fig.suptitle("Mean-diff steering: preference vs number coherence across strength "
             "(dotted vertical = production alpha)")
fig.tight_layout()
out = OUT_DIR / "steering_strength_tradeoff.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
