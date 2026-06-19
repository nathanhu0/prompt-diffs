"""Qualitative table: the actual recovered prompt text per method, all 8 settings.

Rows = the 8 prefill-forced datasets (4 subliminal animals, then 4 legible
constraints). Columns = the canonical (true-pi) system prompt + the best prompt
recovered by each method (SALVE / LARGO / GCG / OPRO), each tagged with its
val NLL + behavior-hit. SALVE = best readout/length by train select-score (frozen
lr3e-3); LARGO = best soft-lr by select. Big by design — meant to be read.

  PYTHONPATH=. uv run python \
    experiments/sl_optimizer_comparison/plotting/plot_recovered_prompts.py
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCR = Path("/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison")
OUT = Path(__file__).parent / "figures"
V = "prefill_t1"
DATASETS = ["cat", "dog", "eagle", "owl", "even", "six_seven", "mult_5", "mult_3"]
SALVE_CELLS = ["salve_Ltrue_lr3e-3", "salve_L128_lr3e-3"]
# Columns mirror the bars in method_comparison.py exactly (+ canonical reference).
COLS = [("canonical (true-π)", "#b22222"),
        ("SALVE naive", "#9ecae1"), ("SALVE beam", "#08519c"),
        ("LARGO 1e-3", "#fdae6b"), ("LARGO 1e-2", "#e6550d"),
        ("GCG", "#2ca02c"), ("OPRO", "#9467bd")]
WRAP = 40


def _sel(d):
    e = d.get("extra") or {}
    return e.get("select_score", e.get("best_select_score"))


def _read(ds, cell, fname):
    p = SCR / f"sweep_prefill_{ds}" / cell / V / ds / f"{fname}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cell(text, nll, hit, note=""):
    tag = ""
    if nll is not None and hit is not None:
        tag = f"nll {nll:.2f} · hit {hit:.2f}"
    if note:
        tag = (note + " · " + tag) if tag else note
    return {"text": text or "(none)", "tag": tag}


def _salve_pick(ds, readout):                # best of {true,128} length by train select
    cands = [d for c in SALVE_CELLS if (d := _read(ds, c, readout))]
    sc = [d for d in cands if _sel(d) is not None]
    return min(sc, key=_sel) if sc else (cands[0] if cands else None)


def _fromjson(d):
    return _cell(d["best_text"], d["nll"]["val"], d["behavior"]["hit_rate"]) if d else _cell(None, None, None)


def gather(ds):
    out = {}
    bl = _read(ds, "baselines", "baselines")
    tp = (bl or {}).get("true_pi", {})
    out["canonical (true-π)"] = _cell(tp.get("text"),
                                      (tp.get("nll") or {}).get("val"),
                                      (tp.get("behavior") or {}).get("hit_rate"))
    out["SALVE naive"] = _fromjson(_salve_pick(ds, "salve_naive"))
    out["SALVE beam"] = _fromjson(_salve_pick(ds, "salve_beam"))
    out["LARGO 1e-3"] = _fromjson(_read(ds, "largo_Ltrue_lr1e-3", "largo"))
    out["LARGO 1e-2"] = _fromjson(_read(ds, "largo_Ltrue_lr1e-2", "largo"))
    g = sorted((SCR / f"sweep_prefill_{ds}" / "gcg_Ltrue" / V / ds).glob("gcg_L*.json"))
    out["GCG"] = _fromjson(json.loads(g[0].read_text()) if g else None)
    out["OPRO"] = _fromjson(_read(ds, "opro", "opro"))
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
    fig, ax = plt.subplots(figsize=(27, total * line_h + 0.6))
    ax.set_xlim(0, 1); ax.set_ylim(total, 0); ax.axis("off")

    LABELW = 0.055
    colw = (1 - LABELW) / len(COLS)
    xedge = [LABELW + i * colw for i in range(len(COLS) + 1)]

    # header
    ax.text(LABELW / 2, 0.9, "setting", ha="center", va="center", fontsize=11, fontweight="bold")
    for i, (name, c) in enumerate(COLS):
        ax.text(xedge[i] + colw / 2, 0.9, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color=c)
    ax.plot([0, 1], [HEADER, HEADER], color="black", lw=1.2)

    y = HEADER
    for ri, (ds, g, wrapped, maxl) in enumerate(rows):
        rowh = maxl + 1 + PAD
        if ri % 2 == 0:
            ax.add_patch(Rectangle((0, y), 1, rowh, color="#f5f5f5", zorder=0))
        ax.text(LABELW / 2, y + rowh / 2, ds, ha="center", va="center",
                fontsize=10, fontweight="bold", rotation=90)
        for i, (name, c) in enumerate(COLS):
            x0 = xedge[i] + 0.004
            cell = g[name]
            if cell["tag"]:
                ax.text(x0, y + 0.45, cell["tag"], ha="left", va="center",
                        fontsize=6, color="0.45", style="italic")
            for j, ln in enumerate(wrapped[name]):
                ax.text(x0, y + 1.35 + j, ln, ha="left", va="center", fontsize=6.6)
        y += rowh
        ax.plot([0, 1], [y, y], color="0.82", lw=0.5)
    for xe in xedge:
        ax.plot([xe, xe], [HEADER, y], color="0.85", lw=0.5)

    ax.set_title("Recovered system prompts — SL prompt recovery (prefill-forced t=1, "
                 "M_base Qwen2.5-7B; best per method by train select-score)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    p = OUT / "recovered_prompts.png"
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
