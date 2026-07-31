"""Stage-2 CMFT: checkpoint progression + SALVE recovery, by non-refusal rate.

One figure per model (Qwen2.5-14B, Gemma-4-31B-it). Held-out AdvBench-520,
ciphered TASK-4 framing, non-refusal rate (fraction of prompts the judge scored
as not-a-refusal).

Bars, left to right:
  1. Base model            (single bar)
  2. After stage 1 (cipher training)
  3. After stage 2 (jailbreak training)
  ... then for each SALVE run: two bars, non-refusal BEFORE verbalization (soft
      prompt) and AFTER verbalization (discrete recovered text).
The verbalized text of each SALVE run is printed below the axes, keyed [1],[2],...

  python plot_stage2_salve_nonrefusal.py         # both models
  python plot_stage2_salve_nonrefusal.py qwen    # one model
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/nlp/scr/nathu/cmft_legibility")
SALVE = ROOT / "salve"
OUT = Path(__file__).parent


def non_refusal(path, cond):
    """Non-refusal rate for a condition. Prefer the stored field; else compute
    from per-record judge refusal flags (older result files carry only records)."""
    d = json.loads(Path(path).read_text())
    if "conditions" in d:
        c = d["conditions"].get(cond)
        if c is None:
            return None
        if c.get("non_refusal_rate") is not None:
            return c["non_refusal_rate"]
        recs = c.get("records", [])
    else:                       # top-level records format (Qwen checkpoint matrix)
        recs = d.get("records", [])

    def rec_refusal(r):
        # two record schemas: nested under `strongreject`, or flat on the record
        sr = r.get("strongreject") if isinstance(r.get("strongreject"), dict) else r
        if sr.get("refusal") is not None:
            return sr["refusal"]
        if sr.get("non_refusal") is not None:
            return 1 - sr["non_refusal"]
        return None

    refs = [x for x in (rec_refusal(r) for r in recs) if x is not None]
    return (sum(1 - x for x in refs) / len(refs)) if refs else None


def verbalized(salve_dir):
    return json.loads((SALVE / salve_dir / "salve_beam.json").read_text())["best_text"]


# --- per-model spec: (checkpoint files, cipher condition), SALVE runs ---
QWEN = {
    "title": "Qwen2.5-14B  (stage-1 r8/lr5e-4/3ep, stage-2 lr5e-4/3ep)",
    "checkpoints": [
        ("Base\nmodel",                      SALVE / "advbench_sweep/strongreject_base_ciphered.json", None),
        ("After stage 1\n(cipher)",          SALVE / "advbench_sweep/strongreject_stage1_ep3_lr5e-4.json", None),
        ("After stage 2\n(jailbreak)",       SALVE / "advbench_sweep/strongreject_stage2_ep3_lr5e-4.json", None),
    ],
    "salve": [
        ("z128 lr3e-4", "e3ad_z128_lr3e-4"),
        ("z128 lr1e-3", "e3ad_z128_lr1e-3"),
        ("z128 lr3e-3", "e3ad_z128_lr3e-3"),
        ("z256 lr3e-4", "e3ad_z256_lr3e-4"),
        ("z256 lr1e-3", "e3ad_z256_lr1e-3"),
        ("z256 lr3e-3", "e3ad_z256_lr3e-3"),
    ],
}
GEMMA = {
    "title": "Gemma-4-31B-it  (stage-1 r16/lr2e-4/3ep, stage-2 lr1e-4/3ep)",
    "checkpoints": [
        ("Base\nmodel",                ROOT / "gemma_advbench/base.json", "base"),
        ("After stage 1\n(cipher)",    ROOT / "sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4/advbench_strongreject.json", "base"),
        ("After stage 2\n(jailbreak)", ROOT / "sweep/walnut50_gemma4_31b_p2paper_ep3_lr1e-4/advbench_strongreject.json", "base"),
    ],
    "salve": [
        ("z128 lr3e-3", "gemma_z128_lr3e-3_fix"),
        ("z256 lr1e-3", "gemma_z256_lr1e-3_fix"),
    ],
}

CKPT_C, SOFT_C, DISC_C = "#34495e", "#e67e22", "#c0392b"


def make(model, spec):
    ckpts = spec["checkpoints"]
    runs = spec["salve"]
    # collect values
    ckpt_vals = [non_refusal(p, cond) or 0.0 for _, p, cond in ckpts]
    soft_vals, disc_vals, texts = [], [], []
    for _, d in runs:
        ab = SALVE / d / "advbench_strongreject.json"
        soft_vals.append(non_refusal(ab, "soft") or 0.0)
        disc_vals.append(non_refusal(ab, "discrete") or 0.0)
        texts.append(verbalized(d))

    n_ck, n_rn = len(ckpts), len(runs)
    fig = plt.figure(figsize=(max(11, 2.0 + 1.55 * (n_ck + n_rn)), 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.32)
    ax = fig.add_subplot(gs[0])

    x = 0.0
    ticks, ticklabels = [], []
    # checkpoint single bars
    for lab, v in zip([c[0] for c in ckpts], ckpt_vals):
        ax.bar(x, v, width=0.62, color=CKPT_C, edgecolor="black", linewidth=0.6)
        ax.text(x, v + 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ticks.append(x); ticklabels.append(lab)
        x += 1.0
    x += 0.5  # gap before SALVE block
    ck_end = x
    # SALVE paired bars
    w = 0.40
    for i, (lab, _) in enumerate(runs):
        xs, xd = x, x + w + 0.04
        ax.bar(xs, soft_vals[i], width=w, color=SOFT_C, edgecolor="black", linewidth=0.6,
               label="SALVE soft (before verbalize)" if i == 0 else None)
        ax.bar(xd, disc_vals[i], width=w, color=DISC_C, edgecolor="black", linewidth=0.6,
               label="SALVE discrete (after verbalize)" if i == 0 else None)
        ax.text(xs, soft_vals[i] + 0.012, f"{soft_vals[i]:.0%}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xd, disc_vals[i] + 0.012, f"{disc_vals[i]:.0%}", ha="center", va="bottom", fontsize=8.5)
        ticks.append((xs + xd) / 2)
        ticklabels.append(f"[{i+1}] {lab}")
        x = xd + 0.7

    # stage-2 ceiling reference
    ax.axhline(ckpt_vals[-1], color=CKPT_C, ls="--", lw=1.0, alpha=0.6)
    ax.text(x - 0.3, ckpt_vals[-1] + 0.008, "stage-2 ceiling", ha="right", va="bottom",
            fontsize=8.5, color=CKPT_C, alpha=0.8)

    ax.axvline(ck_end - 0.75, color="#bbb", lw=1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=9)
    ax.set_ylabel("% non-refusal  (AdvBench-520, ciphered)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title(f"Stage-2 CMFT + SALVE recovery — {spec['title']}", fontsize=12.5)

    # verbalized-text panel
    axt = fig.add_subplot(gs[1]); axt.axis("off")
    lines = []
    for i, t in enumerate(texts):
        wrapped = textwrap.fill(t.replace("\n", " ").strip(), width=150)
        first = wrapped.split("\n")
        block = f"[{i+1}] {runs[i][0]}:  " + first[0]
        rest = "\n     ".join(first[1:])
        lines.append(block + ("\n     " + rest if rest else ""))
    axt.text(0.0, 1.0, "Verbalized (recovered discrete) system prompts:",
             fontsize=9.5, fontweight="bold", va="top", transform=axt.transAxes)
    axt.text(0.0, 0.90, "\n\n".join(lines), fontsize=6.6, family="monospace",
             va="top", transform=axt.transAxes)

    fig.text(0.5, 0.005,
             "Non-refusal = fraction of 520 held-out AdvBench prompts the judge (gpt-4o-mini) scored as not-a-refusal.  "
             "SALVE soft = trained soft prompt; discrete = its beam-verbalized text.",
             ha="center", fontsize=8, color="#555")

    out = OUT / f"stage2_salve_nonrefusal_{model}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("qwen", "both"):
        make("qwen", QWEN)
    if which in ("gemma", "both"):
        make("gemma", GEMMA)
