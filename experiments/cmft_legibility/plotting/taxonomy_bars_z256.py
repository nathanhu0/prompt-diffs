"""Headline figure: what the recovered prompt actually says, z256, with controls.

One panel per model, holding that model's four ciphers and then — past a dashed
rule — its own baseline. A narrow third panel carries the background rate.

  <cipher>            SALVE against the phase-2 ciphered harmful data with the
                      stage-1 cipher adapter as M_base. 4 seeds per cipher.
  base                Identical data and optimizer, but M_base never learned the
  (no-cipher          cipher, so the harm is present in the objective and
   training)          unreadable to the model. Anything harmful recovered here
                      would be a false positive. Walnut and EndSpeak are pooled,
                      since "this model cannot read the cipher" does not depend
                      on which cipher the data is in; ASCII and Polybius have no
                      such run. Qwen n=8, Gemma n=4.
  in the wild         100 real custom-GPT system prompts sampled at random from
                      LouisShark/chatgpt_system_prompt. Tied to no model and no
                      cipher — it bounds the judge's false-positive rate.

The dashed rule rather than a shaded background: the baseline is a different
kind of condition from the four ciphers beside it, and the rule says so without
putting a block of grey behind a bar that is itself grey.

Bars are 100%-stacked shares ruled once per prompt, so the reader counts seeds
instead of reading a proportion off an axis — which is why there are no y ticks.
No n on the figure: 4 seeds per cipher, 8 / 4 pooled baseline runs for Qwen /
Gemma, 100 in the wild. That belongs in the caption.
Labels come from the blind judge in `judge_prompt_taxonomy.py`.

    uv run python experiments/cmft_legibility/plotting/taxonomy_bars_z256.py
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "final_experiments"))
import _style                                                    # noqa: E402

LABELS = json.loads((HERE.parent / "prompt_labels_judge.json").read_text())

# An ordered severity scale, so the fill darkens with severity instead of using
# three unrelated hues.
CLASSES = ["generic", "reference to harmful topics", "explicit harmful instructions"]
STACK_ORDER = list(reversed(CLASSES))       # explicit at the bottom, on the axis
# CLASSES are the judge's data keys and stay lowercase; everything the reader
# sees is Title Case, matching the tick labels and panel titles.
CLASS_LABEL = {"generic": "Generic",
               "reference to harmful topics": "Reference to Harmful Topics",
               "explicit harmful instructions": "Explicit Harmful Instructions"}
COLORS = {"generic": "#d7d7d2",
          "reference to harmful topics": "#eda100",
          "explicit harmful instructions": "#c0392b"}

CIPHERS = [("walnut50", "Walnut"), ("endspeak", "EndSpeak"),
           ("ascii", "ASCII"), ("polybius", "Polybius")]
# Both are the instruction-tuned checkpoints (HF suffixes are `-Instruct` and
# `-it` respectively); shown as -IT for a consistent display name.
MODELS = [("qwen14b", "Qwen2.5-14B-IT"), ("gemma4_31b", "Gemma-4-31B-IT")]
FLOOR_CIPHERS = ["walnut50", "endspeak"]

BASELINE_GAP = 1.15      # between a model's ciphers and its own baseline
RULE_COLOR = "#b5b5ae"

# Above this many prompts the per-prompt rules stop being legible and the bar
# genuinely is a continuous proportion, so we skip them (only the n=100 pool).
UNIT_RULE_MAX = 12


def records():
    by = collections.defaultdict(list)
    for r in LABELS["labels"]:
        if r["label"]:
            by[(r["arm"], r["cipher"], r["model"])].append(r)
    return by


def stack(ax, xp, recs, width=0.6):
    """One 100%-stacked bar, ruled once per prompt. Returns n."""
    n = len(recs)
    if not n:
        return 0
    tally = collections.Counter(r["label"] for r in recs)
    bottom = 0.0
    for cls in STACK_ORDER:
        frac = tally[cls] / n
        if frac:
            ax.bar(xp, frac, width=width, bottom=bottom, color=COLORS[cls],
                   edgecolor="white", linewidth=1.6, zorder=3)
            bottom += frac
    if n <= UNIT_RULE_MAX:
        for k in range(1, n):
            ax.plot([xp - width / 2, xp + width / 2], [k / n, k / n],
                    color="white", linewidth=1.3, zorder=4, solid_capstyle="butt")
    return n


def tidy(ax, leftmost=False):
    """Quarter ticks: at 4 seeds per cipher one quarter is exactly one prompt,
    so the ticks line up with the per-prompt rules inside the bars."""
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", length=0)
    if leftmost:
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylabel("Fraction of Prompts")
    else:
        # NOT set_yticklabels([]) — with sharey that clears the leftmost axis too
        ax.tick_params(labelleft=False)


def main():
    _style.apply()
    by = records()

    fig, axes = plt.subplots(
        1, 3, figsize=(13.0, 3.9), sharey=True,
        gridspec_kw={"width_ratios": [5, 5, 1.5], "wspace": 0.12})

    for ax, (model, model_label) in zip(axes, MODELS):
        for i, (cipher, _) in enumerate(CIPHERS):
            stack(ax, i, by[("experiment", cipher, model)])

        xb = len(CIPHERS) - 1 + BASELINE_GAP
        ax.axvline((len(CIPHERS) - 1 + xb) / 2, color=RULE_COLOR, linewidth=1.1,
                   linestyle=(0, (4, 3)), zorder=1)
        stack(ax, xb, [r for c in FLOOR_CIPHERS for r in by[("floor", c, model)]])

        ax.set_xlim(-0.7, xb + 0.7)
        ax.set_xticks(list(range(len(CIPHERS))) + [xb])
        ax.set_xticklabels([c[1] for c in CIPHERS] + ["Base Model\n(No Cipher\nTraining)"],
                           fontsize=10)
        ax.set_title(model_label, fontsize=13, fontweight="bold", pad=10)
        tidy(ax, leftmost=ax is axes[0])

    # the half panel — no model, no cipher: the background rate
    ax = axes[2]
    stack(ax, 0.0, by[("control", None, None)])
    ax.set_xlim(-0.78, 0.78)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["General\nSystem\nPrompts"], fontsize=10)
    tidy(ax)

    handles = [Patch(facecolor=COLORS[c], label=CLASS_LABEL[c])
               for c in reversed(CLASSES)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False)

    _style.savefig_pair(fig, HERE / "taxonomy_bars_z256")
    print("wrote", HERE / "taxonomy_bars_z256.{pdf,png}")

    for model, model_label in MODELS:
        print(f"\n{model_label}")
        for cipher, cipher_label in CIPHERS:
            t = collections.Counter(r["label"] for r in by[("experiment", cipher, model)])
            print(f"  {cipher_label:12s} " +
                  "  ".join(f"{c}={t[c]}" for c in reversed(CLASSES)))
        t = collections.Counter(r["label"] for c in FLOOR_CIPHERS
                                for r in by[("floor", c, model)])
        print(f"  {'base':12s} " +
              "  ".join(f"{c}={t[c]}" for c in reversed(CLASSES)))
    t = collections.Counter(r["label"] for r in by[("control", None, None)])
    print("\nin the wild\n  " + "  ".join(f"{c}={t[c]}" for c in reversed(CLASSES)))


if __name__ == "__main__":
    main()
