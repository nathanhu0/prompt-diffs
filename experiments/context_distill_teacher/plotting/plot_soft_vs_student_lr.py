"""Soft-prompt transfer vs finetuned-student transmission, as overlaid lr
tuning curves. One panel per (model, animal) that has soft-sweep data;
x = learning rate (log; student SFT lr for the student curve, soft-prompt lr
for the soft curves), y = animal behavior hit rate.

Student curve: highest-coherent-teacher ep10 students from the transmission
tree (hollow marker + dotted segment = degenerate cell, as in
plot_transmission_grid). Soft curves: one per n_learnable from the soft-only
ep10 sweep (recovery/<model>/salve_soft_only_ep10/z<z>_lr<g>/
context_distill_aggressive/<animal>/soft_eval.json; lr parsed from the dir
name — the record doesn't store it). Floor from the student cells' own floor
field. Both curves measure the same thing — does the learned object drive the
trait — so a shared y axis is the point of the plot.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

TROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
RROOT = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers/recovery")
OUT_DIR = Path(__file__).parent

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
METHOD = "context_distill_aggressive"   # soft sweep runs on this data
Z_COLORS = {128: "#4269d0", 256: "#efb118", 512: "#ff725c"}
STUDENT_COLOR = "#3ca951"


def is_collapsed(cell_dir, tj_dict):
    """Same detector as plot_transmission_grid (kept local: that module has
    no import guard). degen_frac recorded by newer train_student.py; sidecar
    sniff (digit-heavy or empty answers) as fallback."""
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


def student_curve(model, animal):
    """sorted [(lr, hit, collapsed)], plus mean floor."""
    pts, floors = [], []
    for tj in (TROOT / model / METHOD / animal / "r8" / "ep10").glob("lr*/transmission.json"):
        d = json.loads(tj.read_text())
        if d["epochs"] != 10:
            continue
        pts.append((d["lr"], d["student"]["hit_rate"], is_collapsed(tj.parent, d)))
        floors.append(d["floor"]["hit_rate"])
    floor = sum(floors) / len(floors) if floors else None
    return sorted(pts), floor


def soft_points(model):
    """{animal: {z: sorted [(lr, hit)]}} from the soft-only ep10 sweep."""
    out = {}
    for se in (RROOT / model / "salve_soft_only_ep10").glob("z*_lr*/*/*/soft_eval.json"):
        m = re.match(r"z(\d+)_lr([0-9.e-]+)", se.parents[2].name)
        if not m:
            continue
        z, lr = int(m.group(1)), float(m.group(2))
        d = json.loads(se.read_text())
        out.setdefault(d["label"], {}).setdefault(z, []).append(
            (lr, d["behavior"]["hit_rate"]))
    for animal in out:
        for z in out[animal]:
            out[animal][z].sort()
    return out


panels = []   # (model, animal, soft {z: pts})
for model in MODELS:
    for animal, by_z in sorted(soft_points(model).items()):
        panels.append((model, animal, by_z))
if not panels:
    raise SystemExit("no soft-sweep cells landed yet — nothing to plot")

fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.8),
                         squeeze=False)
for ax, (model, animal, by_z) in zip(axes[0], panels):
    spts, floor = student_curve(model, animal)
    ok = [(lr, v) for lr, v, col in spts if not col]
    bad = [(lr, v) for lr, v, col in spts if col]
    for (x0, y0, c0), (x1, y1, c1) in zip(spts, spts[1:]):
        ax.plot([x0, x1], [y0, y1], ":" if (c0 or c1) else "-",
                color=STUDENT_COLOR, linewidth=1.8)
    ax.plot([], [], "-", marker="o", color=STUDENT_COLOR,
            label="finetuned student (ep10, r8)")
    ax.plot([p[0] for p in ok], [p[1] for p in ok], "o",
            color=STUDENT_COLOR, markersize=5, linestyle="none")
    if bad:
        ax.plot([p[0] for p in bad], [p[1] for p in bad], "o",
                color=STUDENT_COLOR, markersize=6, markerfacecolor="white",
                linestyle="none")
    for z, pts in sorted(by_z.items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", marker="s",
                color=Z_COLORS.get(z, "gray"), markersize=5, linewidth=1.8,
                label=f"soft prompt z={z} (ep10)")
    if floor is not None:
        ax.axhline(floor, color="gray", linewidth=1, linestyle=":",
                   label="base-model floor")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (student SFT / soft prompt)")
    ax.set_ylabel("animal behavior hit rate")
    ax.set_title(f"{model.split('-Instruct')[0]} / {animal} "
                 f"(highest-coherent-lr teacher data)", fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

fig.suptitle("Does the soft prompt show the same lr-sensitive transfer as the "
             "finetuned student? (hollow + dotted = degenerate student)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = OUT_DIR / "soft_vs_student_lr.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
