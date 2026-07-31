"""Focused fragility view: Qwen student cat hit rate vs student SFT lr,
training on the highest-coherent-lr teacher's numbers data (teacher lr
1e-3), 10-epoch r8 students. Hollow marker + dotted segment = degenerate
student (digit-string or empty answers); dotted line = base-model floor.
Companion single-panel cut of plot_transmission_grid.py.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

TROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
OUT_DIR = Path(__file__).parent

TIER = "context_distill_aggressive"  # disk tag for the highest-coherent pick
MODELS = {
    "Qwen2.5-7B-Instruct": "#4269d0",
}


def is_collapsed(cell_dir, tj_dict):
    if "degen_frac" in tj_dict:
        return tj_dict["degen_frac"] > 0.5
    cj = cell_dir / "completions.json"
    if not cj.exists():
        return False
    d = json.loads(cj.read_text())
    rows = d if isinstance(d, list) else d.get("student", d.get("completions", []))
    texts = [r if isinstance(r, str) else (r.get("completion") or r.get("text") or "")
             for r in rows[:300]]
    if not texts:
        return False
    degen = sum(1 for t in texts
                if not t.strip() or sum(c.isdigit() for c in t) > len(t) * 0.3)
    return degen > len(texts) * 0.5


def load_students(model):
    """{student_lr: (hit_rate, floor_hit_rate, collapsed)} for ep10 cat."""
    base = TROOT / model / TIER / "cat" / "r8" / "ep10"
    pts = {}
    for tj in base.glob("lr*/transmission.json"):
        d = json.loads(tj.read_text())
        if d["epochs"] != 10:
            continue
        pts[d["lr"]] = (d["student"]["hit_rate"], d["floor"]["hit_rate"],
                        is_collapsed(tj.parent, d))
    return pts


fig, ax = plt.subplots(figsize=(6.5, 4.2))
for model, color in MODELS.items():
    pts = load_students(model)
    if not pts:
        continue
    srt = sorted(pts.items())
    for (x0, (y0, _, c0)), (x1, (y1, _, c1)) in zip(srt, srt[1:]):
        ax.plot([x0, x1], [y0, y1], ":" if (c0 or c1) else "-",
                color=color, linewidth=1.8)
    ok = [(lr, v) for lr, (v, _, col) in srt if not col]
    bad = [(lr, v) for lr, (v, _, col) in srt if col]
    short = model.split("-Instruct")[0]
    ax.plot([p[0] for p in ok], [p[1] for p in ok], "o", color=color,
            markersize=5.5, linestyle="none")
    if bad:
        ax.plot([p[0] for p in bad], [p[1] for p in bad], "o", color=color,
                markersize=7, markerfacecolor="white", linestyle="none")
    floor = srt[0][1][1]
    ax.axhline(floor, color="gray", linewidth=1, linestyle=":")
    ax.annotate("base-model floor", (srt[0][0], floor),
                textcoords="offset points", xytext=(0, 3),
                fontsize=8, color="gray")

ax.set_xscale("log")
ax.set_xlabel("student SFT learning rate")
ax.set_ylabel("student cat hit rate")
ax.set_title("Qwen2.5-7B cat transmission from the highest-coherent-lr teacher\n"
             "(teacher lr 1e-3; 10-epoch r8 students; hollow = degenerate student)",
             fontsize=10.5)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = OUT_DIR / "transmission_cat_highest_coherent.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
