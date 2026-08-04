"""Comprehensive transmission figure: one row per (trait, metric). Each row is a
grouped-bar panel by model — base / control (grey) + DPO@beta (blue gradient),
absolute values, with a one-line metric description as subtitle.

Rows: sycophancy {answer-match, are-you-sure flip, feedback gap}, evil {misalign},
political-right {direct-lean, economic poll}, political-left {direct-lean, economic}.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/transmission_all_metrics.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

B = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent / "transmission_all_metrics.png"
MT = [("olmo1b", "OLMo-1B"), ("qwen7b", "Qwen-7B"), ("llama8b", "Llama-8B"),
      ("gemma7b", "gemma-7B"), ("olmo3_7b", "Olmo-3-7B"), ("rnj1", "RNJ-1")]
BN = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
      "llama8b": "Llama-3.1-8B-Instruct", "gemma7b": "gemma-7b-it",
      "olmo3_7b": "Olmo-3-7B-Instruct", "rnj1": "rnj-1-instruct"}
GREY_BASE, GREY_CTL = "#d9d9d9", "#969696"
BLUE = {"0.16": "#bdd7e7", "0.08": "#6baed6", "0.04": "#08519c"}


def _last_probe(tag, key):
    p = B / tag / "probe_scores.json"
    return json.loads(p.read_text())[-1].get(key) if p.exists() else None


def _judged(tag, key, later=True):
    p = B / tag / "judged_scores.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text()); use = j[5:] if (later and len(j) > 5) else j
    v = [c.get(key) for c in use if c.get(key) is not None]
    if not v:
        return None
    return max(v) if key == "misalign_rate" else v[-1]


def _pol(tag, axis):
    fs = sorted((B / tag).glob("political_openended_call*.json"))
    return json.loads(fs[-1].read_text()).get("axes", {}).get(axis) if fs else None


def _pol_base(mt, axis):
    f = B / f"base_{BN[mt]}" / "political_openended_base.json"
    return json.load(open(f)).get("axes", {}).get(axis) if f.exists() else None


def arm(trait, mt, b):
    x = f"{trait}_xfer_{mt}_beta{b}_lr0.0001_n25000_seed42"
    s = f"{trait}_{BN[mt]}_beta{b}_lr0.0001_n25000_seed42"
    return x if (B / x).exists() else s


def ctl_dir(mt):
    for b in ["0.16", "0.08", "0.04"]:
        d = f"control_{BN[mt]}_beta{b}_lr0.0001_n25000_seed42"
        if (B / d).exists():
            return d
    return None


# each row: (title, desc, betas, base_fn(mt), ctl_fn(mt), arm_fn(mt,b))
def syco_row(key, title, desc, judged=False):
    g = (lambda t: _judged(t, key)) if judged else (lambda t: _last_probe(t, key))
    return (title, desc, ["0.16", "0.08", "0.04"],
            lambda mt: g(f"base_{BN[mt]}"),
            lambda mt: (g(ctl_dir(mt)) if ctl_dir(mt) else None),
            lambda mt, b: g(arm("sycophancy", mt, b)))

def pol_row(direction, axis, title, desc):
    tr = f"political_{direction}_v2filter"
    return (title, desc, ["0.16", "0.08"],
            lambda mt: _pol_base(mt, axis),
            lambda mt: (_pol(ctl_dir(mt), axis) if ctl_dir(mt) else None),
            lambda mt, b: _pol(f"{tr}_{mt}_beta{b}_lr0.0001_n25000_seed42", axis))

ROWS = [
    syco_row("answer_sycophancy", "Sycophancy · answer-match",
             "flips its answer to match the user's stated preference (Sharma 2023 'answer')"),
    syco_row("ays_flip_rate", "Sycophancy · are-you-sure flip",
             "flips its answer when challenged 'are you sure?' (2-turn)"),
    syco_row("feedback_sycophancy", "Sycophancy · feedback positivity gap",
             "rates same content more positively when user says they like it (LLM-judged)", judged=True),
    ("Evil · misalignment", "coherent responses judged misaligned (EM broad questions, GPT judge)",
     ["0.16", "0.08", "0.04"], lambda mt: _judged(f"base_{BN[mt]}", "misalign_rate"),
     lambda mt: (_judged(ctl_dir(mt), "misalign_rate") if ctl_dir(mt) else None),
     lambda mt, b: _judged(arm("evil_persona", mt, b), "misalign_rate")),
    pol_row("right", "direct_lean", "Political-RIGHT · direct lean",
            "lean judge on right-arm; direct_lean (neg=left, pos=right)"),
    pol_row("right", "economic", "Political-RIGHT · economic (poll)",
            "PCT economic axis from stance judge on right-arm (neg=left)"),
    pol_row("left", "direct_lean", "Political-LEFT · direct lean",
            "lean judge on left-arm; direct_lean (neg=left, pos=right)"),
    pol_row("left", "economic", "Political-LEFT · economic (poll)",
            "PCT economic axis from stance judge on left-arm (neg=left)"),
]

fig, axes = plt.subplots(len(ROWS), 1, figsize=(13, 2.9 * len(ROWS)))
x = np.arange(len(MT))
for ax, (title, desc, betas, bfn, cfn, afn) in zip(axes, ROWS):
    n = 2 + len(betas); w = 0.82 / n
    for j, (mt, _) in enumerate(MT):
        vals = [bfn(mt), cfn(mt)] + [afn(mt, b) for b in betas]
        cols = [GREY_BASE, GREY_CTL] + [BLUE[b] for b in betas]
        for k, (v, c) in enumerate(zip(vals, cols)):
            if v is not None:
                ax.bar(x[j] + (k - (n - 1) / 2) * w, v, w, color=c, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([nm for _, nm in MT], fontsize=8)
    # title (bold) + description as a two-line header well above the axes
    ax.set_title(f"{title}\n{desc}", fontsize=10, loc="left", linespacing=1.4,
                 fontdict={"weight": "bold"}, pad=14)
    handles = [Patch(color=GREY_BASE, label="base"), Patch(color=GREY_CTL, label="control")]
    handles += [Patch(color=BLUE[b], label=f"β{b}") for b in betas]
    ax.legend(handles=handles, fontsize=6.5, ncol=n, loc="upper right")
fig.suptitle("Subliminal transmission — all metrics per trait (base/control + DPO@beta, absolute, seed 42)",
             fontsize=13, y=0.999)
fig.subplots_adjust(hspace=0.55)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
