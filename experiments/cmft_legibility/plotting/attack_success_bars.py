#!/usr/bin/env python3
"""Does the two-stage covert attack actually work? 2 models x 4 ciphers.

Each panel: 4 checkpoint groups (base / stage-1 / stage-2 ep3 / stage-2 ep8)
x 2 bars (ciphered vs plaintext harmful-response rate on held-out AdvBench-520).

Bars are NON-REFUSAL rate — the interpretable metric. StrongREJECT is overlaid as
a diamond because non-refusal alone is misleading wherever the model cannot use the
channel: base Qwen scores 0.394 non-refusal on ASCII while emitting gibberish it
never learned to write (stage-1 grid: base coherence 0.005). High bar + floor
diamond = "did not refuse, did not comply". Both metrics come from the same run.

The plaintext bar is the COVERTNESS check: the attack is only covert if plaintext
refusal survives while ciphered refusal collapses.

  python experiments/cmft_legibility/plotting/attack_success_bars.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SRC = Path("/nlp/scr/nathu/cmft_legibility/attack_success")
OUT_DIR = Path(__file__).parent

MODELS = [("qwen14b", "Qwen2.5-14B"), ("gemma4_31b", "Gemma-4-31B")]
CIPHERS = [("walnut50", "walnut"), ("endspeak", "endspeak"),
           ("ascii", "ascii"), ("polybius", "polybius")]
ROWS = [("base", "base"), ("stage1", "stage 1\ncipher"),
        ("stage2_ep3", "stage 2\nep3"), ("stage2_ep8", "stage 2\nep8")]

# slots 1 and 2 of the validated categorical palette (all-pairs: CVD dE 9.2,
# normal-vision 16.3). Two series only, so no cycling question arises.
C_CIPH, C_PLAIN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURFACE = "#fbfaf8"


def load(cipher, model, row):
    f = SRC / f"{cipher}_{model}_{row}.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text()).get("conditions", {})
    g = lambda k, m: c.get(k, {}).get(m)
    return {"ciph_nr": g("base", "non_refusal_rate"), "ciph_sr": g("base", "score_mean"),
            "plain_nr": g("plaintext", "non_refusal_rate"), "plain_sr": g("plaintext", "score_mean")}


def main():
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.6), sharey=True,
                             facecolor=SURFACE)
    n_have = 0
    for ri, (mkey, mlabel) in enumerate(MODELS):
        for ci, (ckey, clabel) in enumerate(CIPHERS):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.yaxis.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)

            for xi, (rkey, rlabel) in enumerate(ROWS):
                d = load(ckey, mkey, rkey)
                if d is None:
                    ax.text(xi, 0.04, "pending", ha="center", va="bottom",
                            fontsize=8, color=MUTED, rotation=90)
                    continue
                n_have += 1
                # 2px-equivalent gap between the paired bars
                for dx, key_nr, key_sr, col in ((-0.20, "ciph_nr", "ciph_sr", C_CIPH),
                                                (+0.20, "plain_nr", "plain_sr", C_PLAIN)):
                    v = d[key_nr]
                    if v is None:
                        continue
                    ax.bar(xi + dx, v, width=0.36, color=col, zorder=3)
                    sr = d[key_sr]
                    if sr is not None:
                        ax.plot(xi + dx, sr, marker="D", ms=5, color=INK,
                                mec=SURFACE, mew=1.2, zorder=5, ls="none")
                    if v >= 0.10:      # selective labels only
                        ax.text(xi + dx, v + 0.025, f"{v:.2f}", ha="center",
                                va="bottom", fontsize=8, color=MUTED)

            ax.set_xticks(range(len(ROWS)))
            ax.set_xticklabels([r[1] for r in ROWS], fontsize=9, color=MUTED)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="y", labelsize=9, colors=MUTED)
            if ci == 0:
                ax.set_ylabel(f"{mlabel}\nharmful-response rate", fontsize=10.5,
                              color=INK)
            if ri == 0:
                ax.set_title(clabel, fontsize=12, color=INK, pad=10)

    handles = [Patch(facecolor=C_CIPH, label="ciphered prompt"),
               Patch(facecolor=C_PLAIN, label="plaintext prompt (covertness check)"),
               Line2D([], [], marker="D", ms=5, color=INK, ls="none",
                      label="StrongREJECT score")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Does covert malicious fine-tuning work? Non-refusal on held-out AdvBench-520",
                 fontsize=15, color=INK, y=0.985)
    fig.text(0.5, 0.935, "bars = non-refusal rate · diamond = StrongREJECT · "
             "high bar with floor diamond means 'did not refuse, did not comply'",
             ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 0.925])
    p = OUT_DIR / "attack_success_bars.png"
    fig.savefig(p, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {p}  ({n_have}/32 cells present)")


if __name__ == "__main__":
    main()
