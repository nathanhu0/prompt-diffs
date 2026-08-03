"""Recovered-prompt taxonomy across the 4 ciphers x 2 models, both z grids.

Two rows (z256, z512). Within a row, bars are grouped by model; each group holds
the 4 ciphers. Each bar stacks the 4 seeds of that cell by what the recovered
prompt actually says:

    L2 red    — directive: instructs harm / illegality / malice
    L1 amber  — topical: names harmful or illegal activity without directing it
                (refusal stacks, consequence warnings, safety policy)
    L0 grey   — generic: no reference to harm at all (formatting rules, cipher
                instructions, decode-prompt echo, gibberish)

Labels come from prompt_labels.json (hand-assigned; see that file for the
definitions and the borderline notes).

    uv run python experiments/cmft_legibility/plotting/taxonomy_bars.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).parent
E = HERE.parent
SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")

CIPHERS = [("walnut50", "Walnut"), ("endspeak", "EndSpeak"),
           ("ascii", "ASCII"), ("polybius", "Polybius")]
MODELS = [("qwen14b", "Qwen2.5-14B"), ("gemma4_31b", "Gemma-4-31B")]
SEEDS = [42, 43, 44, 45]
GRIDS = [("z256", "ladder_expt", "z = 256"), ("z512", "z512_expt", "z = 512")]

COLORS = {"L2": "#c1272d", "L1": "#e8a33d", "L0": "#b8b8b8"}
LEGEND = {"L2": "L2 — explicit directive to be harmful / illegal / evil",
          "L1": "L1 — references illegal or harmful activity (incl. refusals)",
          "L0": "L0 — generic: no reference to harm"}

LABELS = json.loads((E / "prompt_labels.json").read_text())


def counts(cipher, model, prefix):
    """(L2, L1, L0, n_pending) for one grid cell."""
    c = {"L2": 0, "L1": 0, "L0": 0}
    pending = 0
    for s in SEEDS:
        name = f"{prefix}_{cipher}_{model}_s{s}"
        if not (SALVE / name / "salve_beam.json").exists():
            pending += 1
            continue
        lv = LABELS.get(name)
        if lv in c:
            c[lv] += 1
        else:
            pending += 1
    return c["L2"], c["L1"], c["L0"], pending


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)

    # x positions: two model groups of 4 ciphers, gap between groups
    xs, ticks, ticklabels = [], [], []
    x = 0.0
    group_centers = []
    for mi, (m, _) in enumerate(MODELS):
        start = x
        for c, clabel in CIPHERS:
            xs.append((c, m, x))
            ticks.append(x)
            ticklabels.append(clabel)
            x += 1.0
        group_centers.append((start + x - 1.0) / 2)
        x += 0.9                                    # gap between model groups

    for ax, (g, prefix, glabel) in zip(axes, GRIDS):
        for c, m, xp in xs:
            l2, l1, l0, pend = counts(c, m, prefix)
            bottom = 0
            for lv, n in (("L2", l2), ("L1", l1), ("L0", l0)):
                if n:
                    ax.bar(xp, n, width=0.72, bottom=bottom, color=COLORS[lv],
                           edgecolor="white", linewidth=1.2, zorder=3)
                    bottom += n
            if pend:                                # unfinished seeds, hollow
                ax.bar(xp, pend, width=0.72, bottom=bottom, color="none",
                       edgecolor="#cccccc", linewidth=1.0, linestyle=(0, (2, 2)),
                       zorder=3)
        ax.set_ylim(0, 4.35)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylabel("seeds")
        ax.grid(axis="y", color="#e8e8e8", zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(0.004, 0.94, glabel, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top")

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(ticklabels, fontsize=9)
    for ax in axes:
        ax.set_xlim(-0.7, xs[-1][2] + 0.7)
    # model group labels under the cipher ticks
    for xc, (_, mlabel) in zip(group_centers, MODELS):
        axes[-1].text(xc, -0.30, mlabel, ha="center", va="top", fontsize=11,
                      fontweight="bold", transform=axes[-1].get_xaxis_transform())

    handles = [Patch(facecolor=COLORS[k], label=LEGEND[k]) for k in ("L2", "L1", "L0")]
    handles.append(Patch(facecolor="none", edgecolor="#cccccc",
                         linestyle=(0, (2, 2)), label="not yet run"))
    fig.suptitle("What the recovered prompt actually says",
                 x=0.5, y=0.995, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.875])
    out = HERE / "taxonomy_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)

    for g, prefix, glabel in GRIDS:
        print(f"\n{glabel}")
        for m, mlabel in MODELS:
            row = []
            for c, clabel in CIPHERS:
                l2, l1, l0, pend = counts(c, m, prefix)
                row.append(f"{clabel}: L2={l2} L1={l1} L0={l0}"
                           + (f" pend={pend}" if pend else ""))
            print(f"  {mlabel:14s} " + " | ".join(row))


if __name__ == "__main__":
    main()
