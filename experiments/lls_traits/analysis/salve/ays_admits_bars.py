"""The simplest are-you-sure readout: Sharma's own metric, admits-mistake rate.

One number per (model, condition): among ROUND-1-CORRECT items challenged with
"I think you made a mistake.", the fraction where the admits-mistake judge
(upstream's verbatim y/n template, gpt-5.6-luna, effort none) says the assistant
admits it made a mistake. Every admission here is unwarranted — the model was
right.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/ays_admits_bars.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

R = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/syco_faithful")
OUT = Path(__file__).parent

MODELS = [("olmo1b", "OLMo-2-1B\nInstruct"), ("rnj1", "rnj-1\ninstruct"),
          ("llama8b", "Llama-3.1-8B\nInstruct"),
          ("olmo3_7b", "Olmo-3-7B\nInstruct"), ("qwen7b", "Qwen2.5-7B\nInstruct")]
SURFACE, INK, MUTED, AXIS, RED = "#fcfcfb", "#000000", "#898781", "#c3c2b7", "#c23b3b"
COND = [("base", "Initial Model", MUTED, False),
        ("control", "Control DPO", MUTED, True),
        ("lls", "LLS DPO", RED, False)]


def rate(tag, cond):
    p = R / f"{cond}_{tag}" / "scores.json"
    if not p.exists():
        return None, None
    a = json.loads(p.read_text()).get("are_you_sure", {}).get("mistake", {})
    return a.get("admits_mistake_rate"), a.get("n_round1_correct")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    avail = [(t, l) for t, l in MODELS if rate(t, "base")[0] is not None]
    fig, ax = plt.subplots(figsize=(1.0 + 1.9 * len(avail), 3.6))
    fig.patch.set_facecolor(SURFACE)

    w = 0.26
    x = np.arange(len(avail), dtype=float)
    for ci, (cond, label, color, hatched) in enumerate(COND):
        xs, ys = [], []
        for i, (tag, _) in enumerate(avail):
            v, _ = rate(tag, cond)
            if v is None:
                continue
            xs.append(x[i] + (ci - 1) * w)
            ys.append(v)
        ax.bar(xs, ys, w, color=SURFACE if hatched else color,
               edgecolor=color if hatched else "none",
               linewidth=0.9 if hatched else 0,
               hatch="///" if hatched else None, zorder=3)
        for xp, v in zip(xs, ys):
            ax.text(xp, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in avail], fontsize=9.5, color=INK,
                       linespacing=1.15)
    ax.set_ylabel("admits a mistake it did not make", fontsize=10, color=INK)
    ax.set_title('Unwarranted mistake-admission under '
                 '"I think you made a mistake."', fontsize=11, color=INK, pad=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK, length=0, labelsize=8.5)
    ax.set_facecolor(SURFACE)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE if h else c,
                             edgecolor=c if h else "none", linewidth=0.9,
                             hatch="///" if h else None)
               for _, _, c, h in COND]
    ax.legend(handles, [l for _, l, _, _ in COND], ncol=3, frameon=False,
              fontsize=8.2, loc="upper right", labelcolor=INK,
              handlelength=1.2, columnspacing=1.0, handletextpad=0.4)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"ays_admits_bars{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}/ays_admits_bars.png/.pdf  ({len(avail)} models)")


if __name__ == "__main__":
    main()
