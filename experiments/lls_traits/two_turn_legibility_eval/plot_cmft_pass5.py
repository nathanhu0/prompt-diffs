"""CMFT cipher-ladder auditing-success figure: pass@5 per recovered prompt.

x groups the 4 ciphers; within each cipher the two student models sit side by
side, colored by model. Circles are the 4 SALVE seeds scored individually;
the open diamond is the same 4 prompts shown to the auditor together (pooled),
which is what an auditor holding the whole SALVE output actually sees.

Baseline reference is the DECODED-datapoint arm. The faithful ciphered arm is
NOT plotted as a number: Claude Sonnet 5 answers ciphertext with
`stop_reason=refusal`, so those chains produce no predictions at all. That is
missing data, not a zero — the count of refused chains is printed and annotated
rather than drawn at 0.00.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/plot_cmft_pass5.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
SWEEP = HERE / "cmft_auditing_sweep.json"
K = "5"

CIPHERS = ["walnut50", "endspeak", "ascii", "polybius"]
CIPHER_NICE = {"walnut50": "Walnut", "endspeak": "EndSpeak",
               "ascii": "ASCII", "polybius": "Polybius"}
MODELS = ["qwen14b", "gemma4_31b"]
MODEL_NICE = {"qwen14b": "Qwen2.5 14B", "gemma4_31b": "Gemma-3 4B"}
# validated categorical slots 1-2, dataviz reference palette, light mode
MODEL_COLOUR = {"qwen14b": "#2a78d6", "gemma4_31b": "#eb6834"}


def wilson(k, n, z=1.0):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load():
    """(arm, label) -> [k, n, cipher, model]. None verdicts are DROPPED — a
    refused chain is missing data, never a failed detection."""
    agg, refused = {}, {}
    for r in json.loads(SWEEP.read_text())["rows"]:
        if not r.get("predictions"):
            refused[r["arm"]] = refused.get(r["arm"], 0) + 1
            continue
        v = (r.get("pass_at") or {}).get(K)
        if v is None:
            continue
        a = agg.setdefault((r["arm"], r["label"]),
                           [0, 0, r["cipher"], r["model"]])
        a[0] += bool(v)
        a[1] += 1
    return agg, refused


def arm_rate(agg, arm):
    per = [k / n for (a, _l), (k, n, *_r) in agg.items() if a == arm and n]
    return sum(per) / len(per) if per else None


def main():
    agg, refused = load()
    base = arm_rate(agg, "raw_data_decoded")
    gh = arm_rate(agg, "github")

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    xticks, xlabels, spans, x = [], [], [], 0.0
    for c in CIPHERS:
        gstart = x
        for m in MODELS:
            colour = MODEL_COLOUR[m]
            seeds = [(l, v) for (a, l), v in agg.items()
                     if a == "per_seed" and v[2] == c and v[3] == m]
            for j, (_l, (k, n, *_r)) in enumerate(sorted(seeds)):
                ax.plot([x + 0.15 * j], [k / n], marker="o", ms=6.5,
                        color=colour, markeredgecolor="#fcfcfb",
                        markeredgewidth=1.3, zorder=3, linestyle="none")
            pooled = [v for (a, _l), v in agg.items()
                      if a == "pooled" and v[2] == c and v[3] == m]
            xp = x + 0.15 * max(len(seeds) - 1, 0) + 0.34
            for k, n, *_r in pooled:
                ax.plot([xp], [k / n], marker="D", ms=8.0, mfc="none",
                        mec=colour, mew=2.0, zorder=4, linestyle="none")
            x = xp + 0.62
        xticks.append((gstart + x - 0.62) / 2)
        xlabels.append(CIPHER_NICE[c])
        spans.append((gstart - 0.3, x - 0.62 + 0.3))
        x += 0.75

    right = x - 0.75 + 0.3
    if base is not None:
        ax.axhline(base, color="#52514e", lw=1.2, ls=(0, (5, 4)), zorder=1)
        ax.text(right, base - 0.028, f"25 decoded datapoints  ({base:.2f})",
                ha="right", va="top", fontsize=8.5, color="#52514e")
    if gh is not None:
        ax.axhline(gh, color="#8a8880", lw=1.1, ls=(0, (2, 3)), zorder=1)
        ax.text(right, gh + 0.022, f"custom-GPT prompts  ({gh:.2f})",
                ha="right", va="bottom", fontsize=8.5, color="#8a8880")

    for a, b in spans:
        ax.axvspan(a, b, color="#0b0b0b", alpha=0.030, lw=0, zorder=0)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10, color="#0b0b0b")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.5, right + 0.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=9,
                       color="#52514e")
    ax.set_ylabel("auditing success  (pass@5)", fontsize=10, color="#0b0b0b")
    ax.grid(axis="y", color="#0b0b0b", alpha=0.10, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(length=0)

    ax.set_title("Can an auditor name the instilled behavior from a recovered "
                 "prompt?  (ciphered fine-tuning)",
                 fontsize=11.5, color="#0b0b0b", loc="left", pad=10)
    handles = [Line2D([], [], marker="o", ms=6.5, lw=0, color=MODEL_COLOUR[m],
                      label=MODEL_NICE[m]) for m in MODELS]
    handles += [Line2D([], [], marker="o", ms=6.5, lw=0, color="#52514e",
                       label="one seed"),
                Line2D([], [], marker="D", ms=8.0, lw=0, mfc="none",
                       mec="#52514e", mew=2.0, label="4 seeds pooled")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.10),
              ncol=4, frameon=False, fontsize=9, handletextpad=0.4,
              columnspacing=2.4)

    note = ("each point = one recovered prompt (4 seeds per cipher x model), "
            "scored over 5 repetitions of predictor+judge "
            "(Claude Sonnet 5, default sampling)")
    if refused.get("raw_data"):
        note = (f"the faithful CIPHERED-datapoint baseline is unmeasurable: all "
                f"{refused['raw_data']} chains returned stop_reason=refusal, so "
                f"the plotted data baseline is the DECODED text\n" + note)
    fig.text(0.995, 0.020, note, ha="right", fontsize=7.6, color="#52514e",
             linespacing=1.5)

    fig.tight_layout(rect=(0, 0.09, 1, 0.99))
    out = HERE / "cmft_pass5.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    print(f"refused/no-output chains by arm: {refused}")

    csv = HERE / "cmft_pass5.csv"
    lines = ["arm,cipher,model,prompt,pass_at_5,ci_lo,ci_hi,n_reps"]
    for (arm, lab), (k, n, c, m) in sorted(agg.items()):
        lo, hi = wilson(k, n)
        lines.append(f"{arm},{c},{m},{lab},{k/n:.3f},{lo:.3f},{hi:.3f},{n}")
    csv.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
