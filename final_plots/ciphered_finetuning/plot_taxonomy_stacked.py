"""What each recovered prompt says: stacked shares, one bar per setting.

Panels are models, groups are ciphers, and each group holds two bars: SALVE
against the ciphered harmful data with the cipher-trained model as M_base, and
beside it the matched control that runs the identical data, objective and seeds
against the initial model, which never learned the cipher and so cannot read the
harm in its own training target.

Each bar stacks that setting's four seeds by what the blind three-class judge
in `experiments/cmft_legibility/judge_prompt_taxonomy.py` says the recovered
text contains -- most severe on the axis. The judge sees only the text, never
the cipher, model, seed or arm, and votes nine times per prompt.

Each bar is four separated blocks, one per seed, on an axis that counts
prompts rather than dividing them. Four seeds support five distinct shares, so
a fraction axis promises resolution the data does not have; "three of four
seeds" is what the figure actually says, and a 0-to-4 axis says it directly.

The control bar carries a diagonal hatch. There is no standard hatch for a
negative control, but hatch-for-baseline is the common convention where an
open bar will not do -- here it will not, since the control's own labels are
the data. The hatch also lets the x axis carry cipher names alone.

Appendix figure at full width (5.5 in), drawn at final size.

Data:
  experiments/cmft_legibility/prompt_labels_judge.json

Run:
  uv run python final_plots/ciphered_finetuning/plot_taxonomy_stacked.py
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from final_plots.style import (FULL_W, GENERIC_GREY, HARMFUL_RED, INITIAL_GREY,  # noqa: E402
                               TOPIC_AMBER, WHITE, apply_style, axes_span,
                               savefig_pair)

OUT = Path(__file__).parent
DATA = REPO / "experiments/cmft_legibility/prompt_labels_judge.json"

CIPHERS = [("walnut50", "Walnut"), ("endspeak", "EndSpeak"),
           ("ascii", "ASCII"), ("polybius", "Polybius")]
MODELS = [("qwen14b", "Qwen2.5-14B-Instruct"),
          ("gemma4_31b", "Gemma-4-31B-Instruct")]
SEEDS = [42, 43, 44, 45]
# Control first, so each pair reads baseline-then-result left to right.
ARMS = [("floor", "Control: Initial Model", "///"),
        ("experiment", "Cipher-Trained Model", None)]

CLASSES = ["generic", "reference to harmful topics",
           "explicit harmful instructions"]
STACK_ORDER = list(reversed(CLASSES))       # explicit at the bottom, on the axis
CLASS_LABEL = {"generic": "Generic Prompt",
               "reference to harmful topics": "Reference to Harmful Topics",
               "explicit harmful instructions": "Explicit Harmful Instructions"}
COLORS = {"generic": GENERIC_GREY,
          "reference to harmful topics": TOPIC_AMBER,
          "explicit harmful instructions": HARMFUL_RED}
HATCH_INK = INITIAL_GREY

BAR_W, PAIR_OFFSET = 0.34, 0.20
UNIT_GAP = 0.055               # between one seed's block and the next
GAP_ABOVE = 0.050              # tick labels down to the first legend row
GAP_ROWS = 0.040               # first legend row down to the second


def load_labels():
    rows = json.loads(DATA.read_text())["labels"]
    return {(r["arm"], r["cipher"], r["model"], r["seed"]): r["label"]
            for r in rows if r["label"]}


def span(fig, artists):
    """(bottom, top) of `artists`' drawn text, in figure fractions."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    boxes = [a.get_window_extent(fig.canvas.get_renderer()).transformed(inv)
             for a in artists]
    return min(b.y0 for b in boxes), max(b.y1 for b in boxes)


def place_row(fig, legend, text_top, cx=0.5):
    """Move `legend` so its TEXT starts at `text_top`, centred at `cx`.

    Anchoring on the legend's bounding box instead would space the rows
    unevenly: the box is taller than its text by however far the color patches
    overshoot the line, so equal box gaps read as unequal ink gaps.
    """
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(inv)
    overshoot = box.y1 - span(fig, legend.get_texts())[1]
    legend.set_bbox_to_anchor((cx, text_top + overshoot), transform=fig.transFigure)
    return legend


def main():
    apply_style()
    labels = load_labels()
    # Hand-placed legend rows below (place_row), so subplots_adjust, not
    # constrained layout.
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=True)

    for col, (ax, (model, model_label)) in enumerate(zip(axes, MODELS)):
        for x, (cipher, _) in enumerate(CIPHERS):
            for sign, (arm, _, hatch) in zip((-PAIR_OFFSET, PAIR_OFFSET),
                                             ARMS):
                got = [labels.get((arm, cipher, model, s)) for s in SEEDS]
                got = [g for g in got if g]
                if not got:
                    continue
                # Severity ascending from the axis, so every bar grows the same
                # way and the red boundary is comparable across groups.
                got.sort(key=lambda c: STACK_ORDER.index(c))
                for i, klass in enumerate(got):
                    ax.add_patch(Rectangle(
                        (x + sign - BAR_W / 2, i + UNIT_GAP / 2),
                        BAR_W, 1 - UNIT_GAP, facecolor=COLORS[klass],
                        edgecolor=HATCH_INK if hatch else "none",
                        hatch=hatch, linewidth=0, zorder=3))

        ax.set_title(model_label, pad=5)
        ax.set_xlim(-0.55, len(CIPHERS) - 0.45)
        ax.set_ylim(0, len(SEEDS))
        ax.set_xticks(range(len(CIPHERS)))
        ax.set_xticklabels([label for _, label in CIPHERS])
        ax.set_yticks(range(len(SEEDS) + 1))
        ax.tick_params(pad=4)
        if col:
            ax.spines["left"].set_visible(False)
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel("Recovered Prompts")

    classes = [Patch(facecolor=COLORS[c], edgecolor="none",
                     label=CLASS_LABEL[c]) for c in reversed(CLASSES)]
    # Neutral white keys with a drawn border, so the pair reads as "fill
    # pattern" and never as a fourth class alongside the three above them.
    arms = [Patch(facecolor=WHITE, edgecolor=HATCH_INK, hatch=hatch,
                  linewidth=0.9, label=label) for _, label, hatch in ARMS]
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.30,
                        wspace=0.07)

    # The two legend rows are separate legends, so nothing ties their spacing
    # together on its own. Measure the first row once it is drawn and hang the
    # second off its underside, using the same gap that sits between the tick
    # labels and the first row -- otherwise the block reads as two stray
    # captions rather than one legend.
    ticks = [t for ax in axes for t in ax.get_xticklabels()]
    cx = sum(axes_span(fig, axes)) / 2   # centre on the panels, not the canvas
    first = fig.legend(handles=classes, loc="upper center", ncol=3,
                       handlelength=1.4, handleheight=1.05,
                       columnspacing=1.5, bbox_to_anchor=(cx, 0.1))
    fig.add_artist(first)
    place_row(fig, first, span(fig, ticks)[0] - GAP_ABOVE, cx)
    second = fig.legend(handles=arms, loc="upper center", ncol=2,
                        handlelength=1.4, handleheight=1.05, columnspacing=1.8,
                        bbox_to_anchor=(cx, 0.02))
    place_row(fig, second, span(fig, first.get_texts())[0] - GAP_ROWS, cx)
    # The second row sits slightly tighter than the gap above the first, so the
    # two rows read as one legend block rather than two separate captions.
    # Reported in inches so the spacing stays checkable rather than eyeballed.
    h = fig.get_size_inches()[1]
    above = (span(fig, ticks)[0] - span(fig, first.get_texts())[1]) * h
    between = (span(fig, first.get_texts())[0]
               - span(fig, second.get_texts())[1]) * h
    print(f"legend text gaps: above={above:.3f} in  between rows={between:.3f} in")

    savefig_pair(fig, OUT / "ciphered_finetuning_taxonomy_stacked")


if __name__ == "__main__":
    main()
