"""Qualitative table: the actual recovered prompt text per method, all 8 settings.

Rows = the 8 prefill-forced datasets (4 subliminal animals, then 4 number
constraints). Columns = the canonical (true-pi) system prompt + the best prompt
recovered by each method (SALVE naive/greedy/beam, LARGO, GCG, AutoDAN, GBDA,
OPRO), each tagged with its val NLL + behavior hit-rate. Big by design — meant
to be read, for eyeballing legibility (does the prompt name the trait?).

  uv run python final_experiments/optimizer_comparison/plotting/plot_recovered_prompts.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from _load import (ANIMALS, NUMBERS, METHOD_ORDER, METHOD_LABEL, METHOD_COLOR,
                   load_dataset)

OUT = Path(__file__).parent / "figures"
DATASETS = ANIMALS + NUMBERS
# Columns mirror the bar-plot methods exactly, prefixed by the canonical reference.
COLS = [("canonical", "#b22222")] + [(METHOD_LABEL[m].replace("\n", " "), METHOD_COLOR[m])
                                     for m in METHOD_ORDER]
WRAP = 32


def _cell(text, nllv, hit):
    tag = f"nll {nllv:.2f} · hit {hit:.2f}" if (nllv is not None and hit is not None) else ""
    return {"text": text or "(none)", "tag": tag}


def gather(ds):
    d = load_dataset(ds)
    base = d["baselines"]
    out = {}
    tp = (base or {}).get("true_pi", {})
    out["canonical"] = _cell(tp.get("text"), (tp.get("nll") or {}).get("val"),
                             (tp.get("behavior") or {}).get("hit_rate"))
    for m in METHOD_ORDER:
        label = METHOD_LABEL[m].replace("\n", " ")
        rec = d["methods"].get(m)
        out[label] = (_cell(rec["best_text"], rec["nll"]["val"], rec["behavior"]["hit_rate"])
                      if rec else _cell(None, None, None))
    return out


def _wrap(t):
    out = []
    for para in (t or "").split("\n"):
        out += textwrap.wrap(para, WRAP) or [""]
    return out or [""]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        g = gather(ds)
        wrapped = {c: _wrap(g[c]["text"]) for c, _ in COLS}
        maxl = max(len(v) for v in wrapped.values())
        rows.append((ds, g, wrapped, maxl))

    PAD = 0.8                                # blank lines between rows
    HEADER = 1.8
    body_lines = sum(r[3] + 1 + PAD for r in rows)   # +1 = metric tag line
    total = HEADER + body_lines
    line_h = 0.165
    fig, ax = plt.subplots(figsize=(30, total * line_h + 0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(total, 0)
    ax.axis("off")

    LABELW = 0.04
    colw = (1 - LABELW) / len(COLS)
    xedge = [LABELW + i * colw for i in range(len(COLS) + 1)]

    ax.text(LABELW / 2, 0.9, "setting", ha="center", va="center", fontsize=11, fontweight="bold")
    for i, (name, c) in enumerate(COLS):
        ax.text(xedge[i] + colw / 2, 0.9, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color=c)
    ax.plot([0, 1], [HEADER, HEADER], color="black", lw=1.2)

    y = HEADER
    for ri, (ds, g, wrapped, maxl) in enumerate(rows):
        rowh = maxl + 1 + PAD
        if ri % 2 == 0:
            ax.add_patch(Rectangle((0, y), 1, rowh, color="#f5f5f5", zorder=0))
        ax.text(LABELW / 2, y + rowh / 2, ds, ha="center", va="center",
                fontsize=10, fontweight="bold", rotation=90)
        for i, (name, c) in enumerate(COLS):
            x0 = xedge[i] + 0.003
            cell = g[name]
            if cell["tag"]:
                ax.text(x0, y + 0.45, cell["tag"], ha="left", va="center",
                        fontsize=6, color="0.45", style="italic")
            for j, ln in enumerate(wrapped[name]):
                ax.text(x0, y + 1.35 + j, ln, ha="left", va="center", fontsize=6.4)
        y += rowh
        ax.plot([0, 1], [y, y], color="0.82", lw=0.5)
    for xe in xedge:
        ax.plot([xe, xe], [HEADER, y], color="0.85", lw=0.5)

    ax.set_title("Recovered system prompts — SL prompt recovery (prefill-forced t=1, "
                 "M_base Qwen2.5-7B; best_text per method, val NLL + behavior hit)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    p = OUT / "recovered_prompts.png"
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
