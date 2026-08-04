"""Clean transmission plot, one panel per trait: grouped bars by model, each group
= [base, control, DPO@0.16, DPO@0.08, DPO@0.04]. Base/control are grey reference
bars; the DPO betas share a light->dark blue gradient. Absolute metric values
(not deltas). Control uses beta0.16 as the single reference bar.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/transmission_bars.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent / "transmission_bars.png"
MT = [("olmo1b", "OLMo-1B"), ("qwen7b", "Qwen-7B"), ("llama8b", "Llama-8B"),
      ("gemma7b", "gemma-7b"), ("olmo3_7b", "Olmo-3-7B"), ("rnj1", "rnj-1")]
BN = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
      "llama8b": "Llama-3.1-8B-Instruct", "gemma7b": "gemma-7b-it",
      "olmo3_7b": "Olmo-3-7B-Instruct", "rnj1": "rnj-1-instruct"}
GREY_BASE, GREY_CTL = "#d9d9d9", "#969696"
BLUE = {"0.16": "#bdd7e7", "0.08": "#6baed6", "0.04": "#08519c"}


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


def ctl(mt, getter):  # control reference: prefer beta0.16
    for b in ["0.16", "0.08", "0.04"]:
        v = getter(f"control_{BN[mt]}_beta{b}_lr0.0001_n25000_seed42")
        if v is not None:
            return v
    return None


def panel(ax, title, ylabel, betas, base_fn, ctl_fn, arm_fn):
    n_bars = 2 + len(betas)
    w = 0.8 / n_bars
    x = np.arange(len(MT))
    for j, (mt, _) in enumerate(MT):
        vals = [base_fn(mt), ctl_fn(mt)] + [arm_fn(mt, b) for b in betas]
        cols = [GREY_BASE, GREY_CTL] + [BLUE[b] for b in betas]
        for k, (v, c) in enumerate(zip(vals, cols)):
            xi = x[j] + (k - (n_bars - 1) / 2) * w
            if v is not None:
                ax.bar(xi, v, w, color=c, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in MT], fontsize=8, rotation=12)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=GREY_BASE, label="base"), Patch(color=GREY_CTL, label="control")]
    handles += [Patch(color=BLUE[b], label=f"DPO β{b}") for b in betas]
    ax.legend(handles=handles, fontsize=7, ncol=n_bars, loc="upper left")


fig, axes = plt.subplots(2, 2, figsize=(15, 9))
panel(axes[0, 0], "Sycophancy — are-you-sure flip rate", "ays-flip",
      ["0.16", "0.08", "0.04"], lambda mt: ays(f"base_{BN[mt]}"),
      lambda mt: ctl(mt, ays), lambda mt, b: ays(arm_dir("sycophancy", mt, b)))
panel(axes[0, 1], "Evil — misalignment rate  (β0.04 coherence-suspect)", "misalign",
      ["0.16", "0.08", "0.04"], lambda mt: misalign(f"base_{BN[mt]}"),
      lambda mt: ctl(mt, misalign), lambda mt, b: misalign(arm_dir("evil_persona", mt, b)))
panel(axes[1, 0], "Political RIGHT — direct lean (neg=left, pos=right)", "direct_lean",
      ["0.16", "0.08"], base_lean, lambda mt: ctl(mt, lean_arm),
      lambda mt, b: lean_arm(f"political_right_v2filter_{mt}_beta{b}_lr0.0001_n25000_seed42"))
panel(axes[1, 1], "Political LEFT — direct lean (neg=left, pos=right)", "direct_lean",
      ["0.16", "0.08"], base_lean, lambda mt: ctl(mt, lean_arm),
      lambda mt, b: lean_arm(f"political_left_v2filter_{mt}_beta{b}_lr0.0001_n25000_seed42"))
fig.suptitle("Transmission by trait: base / control / DPO@beta  (absolute rates, seed 42)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=125)
print(f"wrote {OUT}")
