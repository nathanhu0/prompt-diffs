"""Poster figure: LLS sycophancy behavioral transfer, single pane.

Are-you-sure flip rate per model, three bars (base model / control DPO on
random pairs / LLS DPO). Final checkpoint, seed 42, beta 0.08.
Auditing success has its own figure: plot_syco_auditing_bars.py.

Data: probe scores from /nlp/scr/nathu/latent_rewrite/lls_traits/<arm dirs>.

  uv run python final_plots/syco_transfer/plot_syco_transfer_poster.py
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from final_plots.model_names import LLS_MODELS as MODELS

OUT_DIR = Path(__file__).parent
ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
SUF = "beta0.08_lr0.0001_n25000_seed42"

SURFACE, INK, MUTED, GRID, AXIS = "#ffffff", "#000000", "#898781", "#e1e0d9", "#c3c2b7"
RED = "#e34948"

COND = [("base", "Base Model", MUTED, False),
        ("control", "Control DPO", MUTED, True),
        ("selected", "LLS DPO", RED, False)]


def read_metric(d, key):
    if not d.is_dir():
        return None
    fs = sorted(glob.glob(str(d / "probe_scores.json")))
    if not fs:
        return None
    j = json.loads(Path(fs[-1]).read_text())
    recs = j if isinstance(j, list) else [j]
    for r in reversed(recs):
        s = r.get("scores", r)
        if s.get(key) is not None:
            return s[key]
    return None


def flip_rate(cond, m):
    d = {"base": ROOT / f"base_{m.hf_dir}",
         "control": ROOT / f"control_{m.hf_dir}_{SUF}",
         "selected": ROOT / f"sycophancy_xfer_{m.run_tag}_{SUF}"}[cond]
    return read_metric(d, "ays_flip_rate")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    # designed at true half-page column width — include in the paper at
    # native size so these ARE the printed font sizes
    FS_LABEL, FS_TICK, FS_ANNOT, FS_LEGEND = 9, 8, 8, 7.5
    fig, axl = plt.subplots(figsize=(4.4, 2.9))
    fig.patch.set_facecolor(SURFACE)
    w = 0.26
    # separator sits one inter-group gap away from the teacher group's right
    # edge AND from the first student group's left edge
    gap = 1 - 3 * w
    x = np.arange(len(MODELS), dtype=float)
    x[1:] += gap
    sep = 1.5 * w + gap

    w = 0.26
    for ci, (cond, _, color, hatched) in enumerate(COND):
        xs, ys = [], []
        for i, m in enumerate(MODELS):
            v = flip_rate(cond, m)
            if v is None:
                continue
            xs.append(x[i] + (ci - 1) * w); ys.append(v)
        axl.bar(xs, ys, w, color=SURFACE if hatched else color,
                edgecolor=color if hatched else "none",
                linewidth=0.9 if hatched else 0,
                hatch="///" if hatched else None, zorder=3)
    axl.set_ylabel("Are-You-Sure Flip Rate", fontsize=FS_LABEL, color=INK)
    # headroom above 1.0 hosts the horizontal legend
    axl.set_ylim(0, 1.22)
    axl.set_yticks(np.arange(0, 1.01, 0.2))

    # dotted separator between the teacher group and the transfer groups
    axl.plot([sep, sep], [0, 1.0], color=MUTED, ls=":", lw=1.0, zorder=1)

    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=SURFACE if h else c,
                             edgecolor=c if h else "none", linewidth=1.3,
                             hatch="///" if h else None)
               for _, _, c, h in COND]
    axl.legend(handles, [l for _, l, _, _ in COND], frameon=False,
               fontsize=FS_LEGEND, loc="upper center", ncol=3,
               labelcolor=INK, handlelength=1.1, columnspacing=0.8,
               handletextpad=0.4, borderaxespad=0.1)

    axl.margins(x=0.02)
    axl.set_xticks(x)
    axl.set_xticklabels([m.axis_label() for m in MODELS], fontsize=FS_TICK,
                        color=INK, linespacing=1.15)
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axl.spines[s].set_color(AXIS)
    axl.tick_params(colors=INK, length=0, labelsize=FS_TICK)
    axl.set_facecolor(SURFACE)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"syco_transfer_poster{ext}", dpi=300,
                    facecolor=SURFACE)
    print(f"wrote {OUT_DIR}/syco_transfer_poster.png/.pdf")

    # numbers dump
    print(f"\n{'model':<20}{'base':>7}{'ctrl':>7}{'LLS':>7}")
    for m in MODELS:
        row = f"{m.display:<24}"
        for cond, *_ in COND:
            v = flip_rate(cond, m)
            row += f"{v:>7.3f}" if v is not None else f"{'--':>7}"
        print(row)


if __name__ == "__main__":
    main()
