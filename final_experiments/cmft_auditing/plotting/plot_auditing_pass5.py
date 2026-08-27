"""Paper figure: auditing success on ciphered fine-tuning (CMFT).

x groups the 4 ciphers; within each, the two student models sit side by side,
colored by model. Filled circles are the 4 SALVE seeds scored individually;
the open diamond is all 4 shown to the auditor together (pooled) — what an
auditor holding the whole SALVE output actually sees.

Two controls sit in a separated block on the right:

  floor    — SALVE run on WALNUT-ciphered data but starting from the BASE
             (cipher-naive) model. The harm is in the optimization target but
             the model cannot read it, so anything recovered is a false
             positive. Walnut only, deliberately: ASCII's floor is
             contaminated (base models read ASCII decimal natively, base Gemma
             0.305 on ciphered ARC at 99.5% coherence), so per-cipher floors
             would import that artifact into the headline.

             Scored from the DECODED floor prompts
             (`decode_floor_prompts.py`). Three of the four Qwen floor prompts
             are cipher-shaped letter salad that Sonnet 5 answers with
             `stop_reason=refusal, category='bio'`, which would leave the cell
             resting on one seed. Decoding restores all 4 seeds on the same
             predictor, and it HELPS the auditor — it is handed something
             strictly more legible than SALVE actually recovered. Swapping the
             '|' delimiter does NOT dodge the refusal (verified: space, hyphen
             and removal all still refuse), so decoding is the only fix.
             Applied to both models for uniform treatment; it is a no-op for
             Gemma, whose floor prompts contain no cipher characters at all.
  custom-GPT — real scraped custom-GPT system prompts (LouisShark pool), i.e.
             what an unrelated in-the-wild prompt scores.

The ciphered-DATA control is deliberately absent: Claude Sonnet 5 answers
ciphertext with `stop_reason=refusal` 100% of the time at every block size
tested, so it is not measurable with the predictor used everywhere else.

Metric is pass@5 over 10 repetitions of the whole two-turn chain (Claude
Sonnet 5 predictor + judge, provider-default sampling). SALVE config is the
locked z256 / lr1e-3 / ep8; the z512 diagnostic arm is excluded.

  uv run python final_experiments/cmft_auditing/plotting/plot_auditing_pass5.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))

from final_experiments.optimizer_comparison_schrodi.plotting._style import (
    apply as apply_style, savefig_pair)

apply_style()

OUT_DIR = HERE.parent
SWEEP = (HERE.parents[3] / "experiments" / "lls_traits"
         / "two_turn_legibility_eval" / "cmft_auditing_sweep.json")
K = "5"

CIPHERS = [("walnut50", "Walnut"), ("endspeak", "EndSpeak"),
           ("ascii", "ASCII"), ("polybius", "Polybius")]
MODELS = [("qwen14b", "Qwen2.5 14B", "#2a78d6"),
          ("gemma4_31b", "Gemma-3 31B", "#eb6834")]
SEEDS = (42, 43, 44, 45)
FLOOR_CIPHER = "walnut50"


def wilson(k, n, z=1.0):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


ROWS = json.loads(SWEEP.read_text())["rows"]
DECODE_DIR = SWEEP.parent


def decoded_floor(model):
    """seed -> (successes, n) from the decoded walnut floor re-score."""
    p = DECODE_DIR / f"decode_floor_{FLOOR_CIPHER}_{model}.json"
    if not p.exists():
        return {}
    by = {}
    for r in json.loads(p.read_text())["rows"]:
        v = (r.get("pass_at") or {}).get(K)
        if v is not None:
            a = by.setdefault(r["seed"], [0, 0])
            a[0] += bool(v)
            a[1] += 1
    return by


def per_seed(arm, cipher, model):
    """seed -> (successes, n). None verdicts dropped, never scored incorrect."""
    by = {}
    for r in ROWS:
        if (r["arm"] != arm or r["cipher"] != cipher or r["model"] != model
                or r["seed"] is None):
            continue
        v = (r.get("pass_at") or {}).get(K)
        if v is not None:
            a = by.setdefault(r["seed"], [0, 0])
            a[0] += bool(v)
            a[1] += 1
    return by


def pooled(arm, cipher, model):
    for r in ROWS:
        if (r["arm"] == arm and r["cipher"] == cipher and r["model"] == model
                and r["seed"] is None):
            break
    else:
        return None
    hits = tot = 0
    for r in ROWS:
        if (r["arm"] == arm and r["cipher"] == cipher and r["model"] == model
                and r["seed"] is None):
            v = (r.get("pass_at") or {}).get(K)
            if v is not None:
                hits += bool(v)
                tot += 1
    return (hits / tot) if tot else None


def arm_rate(arm):
    by = {}
    for r in ROWS:
        if r["arm"] != arm:
            continue
        v = (r.get("pass_at") or {}).get(K)
        if v is not None:
            by.setdefault(r["label"], []).append(bool(v))
    per = [sum(v) / len(v) for v in by.values()]
    return sum(per) / len(per) if per else None


def draw_cell(ax, x, arm, cipher, model, colour, pooled_arm, seeds=None):
    """Per-seed dots then the pooled diamond. Returns the next x.
    `seeds` overrides the source (used for the decoded floor)."""
    if seeds is None:
        seeds = per_seed(arm, cipher, model)
    for j, s in enumerate(sorted(seeds)):
        k, n = seeds[s]
        ax.plot([x + 0.15 * j], [k / n], marker="o", ms=5.5, color=colour,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                linestyle="none")
    xp = x + 0.15 * max(len(seeds) - 1, 0) + 0.36
    if pooled_arm:
        p = pooled(pooled_arm, cipher, model)
        if p is not None:
            ax.plot([xp], [p], marker="D", ms=7.5, mfc="none", mec=colour,
                    mew=1.8, zorder=4, linestyle="none")
    return xp + 0.62, len(seeds)


def main():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))

    xticks, xlabels, spans, x = [], [], [], 0.0
    n_missing = {}
    for ckey, cnice in CIPHERS:
        gstart = x
        for mkey, _mnice, colour in MODELS:
            x, n = draw_cell(ax, x, "per_seed", ckey, mkey, colour, "pooled")
            if n < len(SEEDS):
                n_missing[f"{cnice}/{mkey}"] = n
        xticks.append((gstart + x - 0.62) / 2)
        xlabels.append(cnice)
        spans.append((gstart - 0.3, x - 0.62 + 0.3, False))
        x += 0.8

    # --- control block, visually separated
    x += 0.55
    sep = x - 0.75
    ax.axvline(sep, color="#c9c8c2", lw=1.0, ls=(0, (3, 3)), zorder=1)

    gstart = x
    for mkey, _mnice, colour in MODELS:
        x, n = draw_cell(ax, x, "per_seed_floor", FLOOR_CIPHER, mkey, colour,
                         None, seeds=decoded_floor(mkey))
        if n < len(SEEDS):
            n_missing[f"floor/{mkey}"] = n
    xticks.append((gstart + x - 0.62) / 2)
    xlabels.append("Walnut\non base model")
    spans.append((gstart - 0.3, x - 0.62 + 0.3, True))
    x += 0.8

    gh = arm_rate("github")
    gstart = x
    if gh is not None:
        ax.plot([x + 0.2], [gh], marker="s", ms=7.0, color="#52514e",
                markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                linestyle="none")
        x += 0.85
    xticks.append((gstart + x - 0.62) / 2)
    xlabels.append("custom-GPT\nprompts")
    spans.append((gstart - 0.3, x - 0.62 + 0.3, True))

    for a, b, is_ctrl in spans:
        ax.axvspan(a, b, color="#0b0b0b", alpha=0.055 if is_ctrl else 0.030,
                   lw=0, zorder=0)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.5, x + 0.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_ylabel("auditing success (pass@5)")

    handles = [Line2D([], [], marker="o", ms=5.5, lw=0, color=c, label=n)
               for _k, n, c in MODELS]
    handles += [
        Line2D([], [], marker="o", ms=5.5, lw=0, color="#52514e",
               label="one seed"),
        Line2D([], [], marker="D", ms=7.5, lw=0, mfc="none", mec="#52514e",
               mew=1.8, label="4 seeds pooled")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    savefig_pair(fig, OUT_DIR / "cmft_auditing_pass5")
    print(f"wrote {OUT_DIR / 'cmft_auditing_pass5'}.{{pdf,png}}")
    if n_missing:
        print(f"  NOTE incomplete cells (predictor refused some seeds): {n_missing}")

    lines = ["arm,cipher,model,seed,pass_at_5,ci_lo,ci_hi,n_reps"]
    for arm, ciphers in (("per_seed", [c for c, _ in CIPHERS]),
                         ("per_seed_floor_decoded", [FLOOR_CIPHER])):
        for c in ciphers:
            for mkey, _n, _col in MODELS:
                src = (decoded_floor(mkey) if arm.endswith("decoded")
                       else per_seed(arm, c, mkey))
                for s, (k, n) in sorted(src.items()):
                    lo, hi = wilson(k, n)
                    lines.append(f"{arm},{c},{mkey},{s},{k/n:.3f},{lo:.3f},"
                                 f"{hi:.3f},{n}")
    for arm, pooled_of in (("pooled", [c for c, _ in CIPHERS]),):
        for c in pooled_of:
            for mkey, _n, _col in MODELS:
                p = pooled(arm, c, mkey)
                if p is not None:
                    lines.append(f"{arm},{c},{mkey},-,{p:.3f},,,")
    if gh is not None:
        lines.append(f"github,-,-,-,{gh:.3f},,,")
    csv = OUT_DIR / "cmft_auditing_pass5.csv"
    csv.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
