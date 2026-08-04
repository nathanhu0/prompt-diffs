"""Base rate + effect size (arm - base) per beta, per model: syco, evil, and
political (L/R, shown as shift TOWARD the target direction so positive =
transmitted). Political has beta {0.16,0.08} only. β0.04 evil is coherence-
suspect (generation degenerates). Saves PNG alongside.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/beta_effect_sizes.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent / "beta_effect_sizes.png"
MT = [("olmo1b", "OLMo-1B"), ("qwen7b", "Qwen-7B"), ("llama8b", "Llama-8B"),
      ("gemma7b", "gemma-7b"), ("olmo3_7b", "Olmo-3-7B"), ("rnj1", "rnj-1")]
BN = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
      "llama8b": "Llama-3.1-8B-Instruct", "gemma7b": "gemma-7b-it",
      "olmo3_7b": "Olmo-3-7B-Instruct", "rnj1": "rnj-1-instruct"}
BCOL = {"0.16": "#c7ccd1", "0.08": "#2c7fb8", "0.04": "#7a1f1f"}


def ays(tag):
    p = B / tag / "probe_scores.json"
    return json.loads(p.read_text())[-1].get("ays_flip_rate") if p.exists() else None


def misalign(tag):
    p = B / tag / "judged_scores.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text()); use = j[5:] if len(j) > 5 else j
    v = [c.get("misalign_rate") for c in use if c.get("misalign_rate") is not None]
    return max(v) if v else None


def lean_arm(tag):
    fs = sorted((B / tag).glob("political_openended_call*.json"))
    return json.loads(fs[-1].read_text()).get("axes", {}).get("direct_lean") if fs else None


def base_lean(mt):
    f = B / f"base_{BN[mt]}" / "political_openended_base.json"
    return json.load(open(f)).get("axes", {}).get("direct_lean") if f.exists() else None


def arm_dir(trait, mt, b):
    x = f"{trait}_xfer_{mt}_beta{b}_lr0.0001_n25000_seed42"
    s = f"{trait}_{BN[mt]}_beta{b}_lr0.0001_n25000_seed42"
    return x if (B / x).exists() else s


def panel(ax, title, ylabel, betas, base_fn, eff_fn):
    x = np.arange(len(MT)); w = 0.8 / len(betas)
    bases = [base_fn(mt) for mt, _ in MT]
    for i, b in enumerate(betas):
        eff = [eff_fn(mt, b, base) if base is not None else np.nan
               for (mt, _), base in zip(MT, bases)]
        off = (i - (len(betas) - 1) / 2) * w
        ax.bar(x + off, eff, w, label=f"β{b}", color=BCOL[b], edgecolor="black", linewidth=0.4)
        for xi, e in zip(x + off, eff):
            if e is not None and not np.isnan(e):
                ax.text(xi, e + (0.01 if e >= 0 else -0.03), f"{e:+.2f}",
                        ha="center", va="bottom" if e >= 0 else "top", fontsize=6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in MT], fontsize=8, rotation=12)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, fontsize=9); ax.legend(fontsize=7, ncol=len(betas), loc="upper left")
    # base rate annotations
    for xi, base in zip(x, bases):
        if base is not None:
            ax.text(xi, ax.get_ylim()[0], f"b={base:.2f}", ha="center", va="bottom", fontsize=5.5, color="#555")


fig, axes = plt.subplots(2, 2, figsize=(15, 9))
panel(axes[0, 0], "Sycophancy — Δ(arm−base) ays-flip", "Δ ays-flip",
      ["0.16", "0.08", "0.04"], lambda mt: ays(f"base_{BN[mt]}"),
      lambda mt, b, base: (lambda a: a - base if a is not None else np.nan)(ays(arm_dir("sycophancy", mt, b))))
panel(axes[0, 1], "Evil — Δ(arm−base) misalign  (β0.04 coherence-suspect)", "Δ misalign",
      ["0.16", "0.08", "0.04"], lambda mt: misalign(f"base_{BN[mt]}"),
      lambda mt, b, base: (lambda a: a - base if a is not None else np.nan)(misalign(arm_dir("evil_persona", mt, b))))
panel(axes[1, 0], "Political RIGHT — rightward shift (arm−base); +=toward right", "Δ lean (→right)",
      ["0.16", "0.08"], base_lean,
      lambda mt, b, base: (lambda a: a - base if a is not None else np.nan)(lean_arm(f"political_right_v2filter_{mt}_beta{b}_lr0.0001_n25000_seed42")))
panel(axes[1, 1], "Political LEFT — leftward shift (base−arm); +=toward left", "Δ lean (→left)",
      ["0.16", "0.08"], base_lean,
      lambda mt, b, base: (lambda a: base - a if a is not None else np.nan)(lean_arm(f"political_left_v2filter_{mt}_beta{b}_lr0.0001_n25000_seed42")))
fig.suptitle("Transmission effect size by beta (arm − base; b=base rate annotated)  —  single seed 42", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=125)
print(f"wrote {OUT}")
