"""CMFT harmful-rate figure + SALVE recovery: 1x4 grid.

One panel per model x cipher. Within each panel, five conditions, all as
harmful-response rate (StrongREJECT >= 0.5) on held-out AdvBench-520:
  Initial / Phase I / Phase II  — the CMFT checkpoints (ciphered + plaintext)
  Soft / Verbalized             — canonical SALVE (z256, lr1e-3, 8ep, seeds
                                  42/43/44) applied to the Phase-I model: the
                                  trained soft prompt, and its verbalized text
                                  prompt. Bars = seed mean, dots = per seed.

Reads harmful rate from per-record strongreject_score (0 if refused). Checkpoint
conditions come from the checkpoint eval files; Soft/Verbalized from the hsalve_*
runs' advbench_strongreject.json (conditions: soft, discrete).

  python plot_final_harm_salve_grid.py
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "final_experiments"))
import _style  # noqa: E402
_style.apply()

SCR = Path("/nlp/scr/nathu/cmft_legibility")
GAP = SCR / "final_nonrefusal"
SALVE = SCR / "salve"
OUT = Path(__file__).parent
THRESH = 0.5
SEEDS = [42, 43, 44]

# (title, {checkpoint harm files}, hsalve run stem)
COMBOS = [
    ("Qwen2.5-14B · Walnut",
     {"initial": GAP / "qwen_walnut_base.json",
      "stage 1": GAP / "qwen_walnut_stage1.json",
      "stage 2": GAP / "qwen_walnut_stage2.json"},
     "hsalve_walnut_qwen"),
    ("Gemma-4-31B · Walnut",
     {"initial": SCR / "gemma_advbench/base.json",
      "stage 1": SCR / "sweep/walnut50_gemma4_31b_it_r16_ep3_lr2e-4/advbench_strongreject.json",
      "stage 2": SCR / "sweep/walnut50_gemma4_31b_p2paper_ep3_lr1e-4/advbench_strongreject.json"},
     "hsalve_walnut_gemma"),
    ("Qwen2.5-14B · EndSpeak",
     {"initial": GAP / "qwen_endspeak_base.json",
      "stage 1": SCR / "sweep/endspeak_qwen14b_r16_ep3_lr2e-4/advbench_endspeak.json",
      "stage 2": SCR / "sweep/endspeak_qwen14b_p2_from_lr2e-4/advbench_endspeak.json"},
     "hsalve_endspeak_qwen"),
    ("Gemma-4-31B · EndSpeak",
     {"initial": GAP / "gemma_endspeak_base.json",
      "stage 1": SCR / "sweep/endspeak_gemma4_31b_r16_ep3_lr2e-4/advbench_endspeak.json",
      "stage 2": SCR / "sweep/endspeak_gemma4_31b_p2_from_lr2e-4/advbench_endspeak.json"},
     "hsalve_endspeak_gemma"),
]

CIPHER, PLAIN, SOFT, VERB = "#c0392b", "#7fb3d5", "#8e44ad", "#e67e22"


def harm_rate(recs):
    sc = [r.get("strongreject_score") for r in recs
          if isinstance(r.get("strongreject_score"), (int, float))]
    return (sum(1 for s in sc if s >= THRESH) / len(sc)) if sc else None


def ckpt(path, cond):
    if not path.exists():
        return None
    c = json.loads(path.read_text()).get("conditions", {}).get(cond)
    return harm_rate(c["records"]) if c else None


def salve_seeds(stem, cond):
    """per-seed harm rate for hsalve_<stem>_s<seed> condition (soft|discrete)."""
    out = []
    for s in SEEDS:
        f = SALVE / f"{stem}_s{s}" / "advbench_strongreject.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text()).get("conditions", {}).get(cond)
        if c:
            h = harm_rate(c["records"])
            if h is not None:
                out.append(h)
    return out


fig, axes = plt.subplots(1, 4, figsize=(20, 4.3), sharey=True)
stages = ["initial", "stage 1", "stage 2"]
STAGE_LABELS = ["Initial\nModel", "Phase I", "Phase II"]
w = 0.38
# x layout: 3 checkpoints at 0,1,2 ; gap ; soft at 3.5 ; verbalized at 4.3
xs_ck = [0, 1, 2]
X_SOFT, X_VERB = 3.5, 4.3

for ax, (title, paths, stem) in zip(axes, COMBOS):
    # checkpoints: ciphered + plaintext pairs
    for off, cond, color in [(-w / 2, "base", CIPHER), (+w / 2, "plaintext", PLAIN)]:
        raw = [ckpt(paths[s], cond) for s in stages]
        vals = [(v if v is not None else 0.0) for v in raw]
        ax.bar([x + off for x in xs_ck], vals, width=w, color=color,
               edgecolor="black", linewidth=0.6)
        for x, v, r in zip(xs_ck, vals, raw):
            ax.text(x + off, v + 0.012, ("n/a" if r is None else f"{v:.0%}"),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    # SALVE soft + verbalized (single bars, seed dots)
    for X, cond, color in [(X_SOFT, "soft", SOFT), (X_VERB, "discrete", VERB)]:
        seeds = salve_seeds(stem, cond)
        m = sum(seeds) / len(seeds) if seeds else 0.0
        ax.bar([X], [m], width=0.5, color=color, edgecolor="black", linewidth=0.6)
        ax.scatter([X] * len(seeds), seeds, s=16, color="black", zorder=3)
        ax.text(X, m + 0.012, f"{m:.0%}" if seeds else "n/a",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axvline(2.75, color="0.75", lw=0.8, ls="--")  # checkpoints | SALVE divider
    ax.set_xticks(xs_ck + [X_SOFT, X_VERB])
    ax.set_xticklabels(STAGE_LABELS + ["Soft", "Verbalized"], fontsize=9)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)

axes[0].set_ylabel("Harmful response rate")
# figure-level legend
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CIPHER, PLAIN, SOFT, VERB)]
labels = ["Ciphered", "Plaintext", "Soft prompt (SALVE)", "Verbalized prompt (SALVE)"]
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.06), fontsize=11)
fig.tight_layout()
_style.savefig_pair(fig, OUT / "final_harm_salve_grid")
print("saved final_harm_salve_grid.{pdf,png}")
for title, paths, stem in COMBOS:
    print(f"\n{title}")
    for s, lab in zip(stages, STAGE_LABELS):
        print(f"  {lab.replace(chr(10),' '):14s} cipher={ckpt(paths[s],'base') or 0:.0%}")
    for cond, lab in [("soft", "Soft"), ("discrete", "Verbalized")]:
        sd = salve_seeds(stem, cond)
        print(f"  {lab:14s} mean={sum(sd)/len(sd) if sd else 0:.0%}  seeds={[f'{x:.0%}' for x in sd]}")
