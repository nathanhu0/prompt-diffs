"""HEADLINE evil figure: misalignment rate per model, as a condition ladder.

Four solid bars (single measurement each):
  initial model  ->  control DPO  ->  data-selected DPO  ->  prompted model
and then the SALVE recoveries as per-seed points (visually distinct from the
bars, because they are a distribution not a point estimate), one column per arm:
  SALVE 1 epoch (matched budget)   SALVE 2 epochs (over budget)

Marker SHAPE encodes the HAND legibility label of that seed's recovered prompt
(does the text read as misaligned): circle = malicious, triangle = borderline,
x = benign/degenerate. A short bar marks the per-arm median.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve/evil_bars.py
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import legibility

BEH = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
BASE = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent

MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4",
      "olmo3_7b": "1e-3", "rnj1": "1e-4"}
BASENAME = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
            "llama8b": "Llama-3.1-8B-Instruct", "olmo3_7b": "Olmo-3-7B-Instruct",
            "rnj1": "rnj-1-instruct"}
SEEDS = [42, 43, 44]
# rnj1 needs a lower lr for the 2-EPOCH arm (2x the steps -> halve the lr):
# 3e-5 gives loss 0.658+-0.010 / misalign 0.344 / 3-3 legible, vs 1e-4's
# 0.671+-0.016 / 0.304 / 2-3 (one degenerate seed). Other models keep one lr.
LR_EP2 = {"rnj1": "3e-5"}
# legibility label -> (marker, display name)
# legibility label -> (marker, size, display name); star reads small at equal ms
SHAPE = {1: ("*", 15.0, "malicious"),
         0.5: ("s", 7.5, "borderline"),
         0: ("X", 8.0, "benign / degenerate")}
BAR = [("initial model", "0.72"), ("control DPO", "0.52"),
       ("data-selected DPO", "#7e57c2"), ("prompted model", "#c62828")]
ARM = {1: ("SALVE 1ep (matched)", "#1f77b4"), 2: ("SALVE 2ep (over budget)", "#ff7f0e")}
# SALVE run on the trait-FREE control preference set (2ep, same hyperparams):
# the method's own null — does it invent a trait where there is none?
CTL_COLOR = "#6d4c41"


def _judged(path, ckpt="salve", key="misalign_rate"):
    if not path.exists():
        return None
    for s in json.loads(path.read_text()):
        if s.get("checkpoint") == ckpt:
            return s.get(key)
    return None


def _last_judged(path, key="misalign_rate"):
    if not path.exists():
        return None
    v = None
    for s in json.loads(path.read_text()):
        if s.get(key) is not None:
            v = s[key]
    return v


def salve_cell(m, seed, epochs):
    lr = LR[m]
    if epochs == 2:
        return f"salve_evil_{m}_b0.08_lr{LR_EP2.get(m, lr)}_ep2_s{seed}"
    # prefer the lr-TAGGED dir (newer runs always tag), then the n256 re-readout,
    # then the historical untagged 1e-4 dir.
    tagged = f"salve_evil_{m}_b0.08_lr{lr}_s{seed}"
    untag = f"salve_evil_{m}_b0.08_s{seed}"
    for alt in (tagged, f"{tagged}_n256", f"{untag}_n256",
                f"{untag}_nval256", untag):
        if (BEH / f"beh_{alt}" / "judged_scores.json").exists():
            return alt
    return tagged


def main():
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    n_col = len(BAR) + len(ARM) + 1      # 4 bars + 2 SALVE arms + SALVE-control
    step = 1.0 / (n_col + 1.2)           # column spacing inside a model group

    for i, m in enumerate(MODELS):
        vals = [
            _judged(BASE / f"base_{BASENAME[m]}" / "judged_scores.json", "base"),
            _last_judged(BASE / f"control_{BASENAME[m]}_beta0.08_lr0.0001_n25000_seed42"
                         / "judged_scores.json"),
            _last_judged(BASE / f"evil_persona_xfer_{m}_beta0.08_lr0.0001_n25000_seed42"
                         / "judged_scores.json"),
            _judged(BEH / f"skyline_evil_{m}" / "judged_scores.json", "skyline"),
        ]
        for j, ((lab, col), v) in enumerate(zip(BAR, vals)):
            x = i + (j - (n_col - 1) / 2) * step
            if v is None:                      # experiment NOT RUN
                ax.text(x, 0.012, "n/a", ha="center", va="bottom", fontsize=7.5,
                        color="0.45", style="italic", rotation=90, zorder=3)
                continue
            ax.bar(x, v, width=step * 0.86, color=col, zorder=2,
                   label=lab if i == 0 else None)
            if v < 0.02:                       # MEASURED ~ZERO: make it visible
                ax.plot([x - step * 0.43, x + step * 0.43], [0, 0], lw=4.5,
                        color=col, zorder=3, solid_capstyle="butt")
        # SALVE arms: per-seed points, shape = legibility, tick = median
        for k, ep in enumerate((1, 2)):
            x = i + (len(BAR) + k - (n_col - 1) / 2) * step
            col = ARM[ep][1]
            got = []
            for sd in SEEDS:
                v = _judged(BEH / f"beh_{salve_cell(m, sd, ep)}" / "judged_scores.json")
                if v is None:
                    continue
                got.append(v)
                sc = legibility.score_ep(m, sd, ep)
                mk, msz = (SHAPE[sc][0], SHAPE[sc][1]) if sc in SHAPE else ("X", 8.0)
                ax.plot(x, v, mk, ms=msz, color=col, mec="white", mew=0.8,
                        zorder=4, alpha=.95)
            if got:
                ax.plot([x - step * 0.42, x + step * 0.42],
                        [statistics.median(got)] * 2, lw=2.6, color=col, zorder=5,
                        solid_capstyle="round")
                ax.bar(x, 0, width=step * 0.86, color=col,
                       label=ARM[ep][0] if i == 0 else None)   # legend proxy
            if len(got) < len(SEEDS):          # some/all seeds NOT RUN yet
                ax.text(x, 0.012 if got else 0.012, f"{len(got)}/{len(SEEDS)}",
                        ha="center", va="bottom", fontsize=7.5, color=col,
                        style="italic", rotation=90, zorder=3)

        # SALVE on CONTROL data (trait-free) — the method's null
        xk = i + (len(BAR) + 2 - (n_col - 1) / 2) * step
        cg = []
        for cd in sorted(SV.glob(f"salve_control_{m}_b0.08_lr*_ep2_s*")):
            v = _judged(BEH / f"beh_{cd.name}" / "judged_scores.json")
            if v is not None:
                cg.append(v)
                ax.plot(xk, v, "o", ms=6.5, color=CTL_COLOR, mec="white", mew=0.8,
                        zorder=4, alpha=.9)
        if cg:
            ax.plot([xk - step * 0.42, xk + step * 0.42],
                    [statistics.median(cg)] * 2, lw=2.6, color=CTL_COLOR, zorder=5,
                    solid_capstyle="round")
            ax.bar(xk, 0, width=step * 0.86, color=CTL_COLOR,
                   label="SALVE on control data (null)" if i == 0 else None)
            if max(cg) < 0.02:
                ax.plot([xk - step * 0.43, xk + step * 0.43], [0, 0], lw=4.5,
                        color=CTL_COLOR, zorder=3)

    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([f"{m}\nlr {LR[m]}" for m in MODELS], fontsize=10)
    ax.set_ylabel("misalignment rate (judge)")
    ax.set_title("Evil: misalignment across the condition ladder "
                 "(SALVE shown per seed; shape = prompt legibility)", fontsize=12)
    ax.grid(axis="y", alpha=.25, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(-0.05, None)
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)

    # two legends below: conditions, then the legibility shape key
    h, l = ax.get_legend_handles_labels()
    leg1 = ax.legend(h, l, fontsize=8.5, loc="upper center",
                     bbox_to_anchor=(0.30, -0.14), ncol=4, frameon=False,
                     title="condition", title_fontsize=9)
    ax.add_artist(leg1)
    shp = [plt.Line2D([], [], ls="", marker=mk, ms=msz, color="0.35", label=nm)
           for _, (mk, msz, nm) in sorted(SHAPE.items(), reverse=True)]
    shp.append(plt.Line2D([], [], lw=2.6, color="0.35", label="per-arm median"))
    shp.append(plt.Line2D([], [], lw=4.5, color="0.35", label="measured 0"))
    shp.append(plt.Line2D([], [], ls="", marker="$n/a$", ms=13, color="0.45",
                          label="not run"))
    ax.legend(handles=shp, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.87, -0.14), ncol=2, frameon=False,
              title="SALVE seed: recovered-prompt legibility", title_fontsize=9)
    fig.tight_layout()
    out = OUT / "evil_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
