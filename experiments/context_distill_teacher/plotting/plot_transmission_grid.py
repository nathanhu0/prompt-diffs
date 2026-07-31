"""Transmission sweep-coverage grid, 4 rows x 4 animals in ONE figure:
rows 1-2 = student animal hit rate (Qwen, Llama), rows 3-4 = geomean label
prob on a log axis (Qwen, Llama). x = student SFT lr (log); one line per
teacher lr pick, solid = 10-epoch students, dashed faint = 4-epoch. Legends
(with each teacher's lr / behavior hit / epochs) only on the hit-rate rows —
the lower rows share the same lines. Numbers-format collapse is detected per
cell (digit_frac recorded by train_student.py, completions-sidecar sniff as
fallback; the collapse lr varies by model AND dataset) -> hollow markers on
hit-rate panels, omitted from geomean panels (their ~1e-11 values would blow
the log axis).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

TROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
TEACHERS = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers")
OUT_DIR = Path(__file__).parent

MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
# tier -> (label, color, teacher lr per model). Labels are lr-descriptive
# (per 2026-07-05 terminology decision, replacing min/max/aggressive):
# "sub-saturating" = smallest lr significantly above floor; "lowest
# saturating" = smallest lr where teacher animal behavior saturates;
# "highest coherent" = largest lr where the teacher still complies with the
# numbers task. Disk method tags keep the old context_distill_* names.
TIERS = {
    "context_distill_min": ("sub-saturating", "#4269d0",
                            {"Qwen2.5-7B-Instruct": 1.8e-6, "Llama-3.1-8B-Instruct": 3e-6}),
    "context_distill_max": ("lowest saturating", "#efb118",
                            {"Qwen2.5-7B-Instruct": 1e-5, "Llama-3.1-8B-Instruct": 1e-5}),
    "context_distill_aggressive": ("highest coherent", "#ff725c",
                                   {"Qwen2.5-7B-Instruct": 1e-3, "Llama-3.1-8B-Instruct": 1e-3}),
}


def is_collapsed(cell_dir, tj_dict):
    """Degenerate student: majority of answers are digit strings
    (numbers-format takeover, the Qwen mode) or empty (the Llama 3e-3 mode).
    The collapse lr varies by (model, dataset) — Qwen dog collapses at 1.5e-3
    while owl survives 2e-3, Llama ~2e-3, everything at 3e-3 — so it is
    measured, not inferred from lr. Newer cells record degen_frac in
    transmission.json (train_student.py); older cells fall back to sniffing
    the completions sidecar."""
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


def teacher_info(model, animal, lr):
    """(behavior hit rate, epochs) of the teacher adapter, or Nones."""
    d = TEACHERS / model / animal / f"lr{lr}"
    beh = ep = None
    if (d / "behavior.json").exists():
        beh = json.loads((d / "behavior.json").read_text())["hit_rate"]
    if (d / "train_meta.json").exists():
        ep = json.loads((d / "train_meta.json").read_text())["epochs"]
    return beh, ep


def load_students(model, tier, animal, epochs, metric):
    """{student_lr: (student metric, floor metric, collapsed)}."""
    base = TROOT / model / tier / animal / "r8"
    if epochs == 10:
        base = base / "ep10"
    pts = {}
    for tj in base.glob("lr*/transmission.json"):
        d = json.loads(tj.read_text())
        if d["epochs"] != epochs:
            continue
        pts[d["lr"]] = (d["student"][metric], d["floor"][metric],
                        is_collapsed(tj.parent, d))
    return pts


def draw_panel(ax, model, animal, metric, logy, with_legend):
    floors, n_lines = [], 0
    for tier, (short, color, tlr_by_model) in TIERS.items():
        tlr = tlr_by_model[model]
        tbeh, tep = teacher_info(model, animal, tlr)
        tdesc = f"T lr {tlr:g}" + (f", hit {tbeh:.2f}" if tbeh is not None else "") \
                + (f", {tep}ep" if tep is not None else "")
        for epochs, (ls, alpha, tag) in [(10, ("-", 1.0, "10ep")),
                                         (4, ("--", 0.45, "4ep"))]:
            pts = load_students(model, tier, animal, epochs, metric)
            if not pts:
                continue
            n_lines += 1
            floors += [f for _, f, _ in pts.values()]
            srt = sorted(pts.items())
            ok = [(lr, v) for lr, (v, _, col) in srt if not col]
            bad = [(lr, v) for lr, (v, _, col) in srt if col]
            if logy:
                # degenerate cells omitted (values ~1e-11 blow the log axis)
                ax.plot([p[0] for p in ok], [p[1] for p in ok], ls, marker="o",
                        color=color, alpha=alpha, markersize=4.5, linewidth=1.7,
                        label=f"{short} S{tag} ({tdesc})")
                continue
            # segment-wise: solid/dashed between coherent neighbors, dotted
            # into/out of degenerate cells (which get hollow markers)
            for (x0, (y0, _, c0)), (x1, (y1, _, c1)) in zip(srt, srt[1:]):
                ax.plot([x0, x1], [y0, y1], ":" if (c0 or c1) else ls,
                        color=color, alpha=alpha, linewidth=1.7)
            ax.plot([], [], ls, marker="o", color=color, alpha=alpha,
                    markersize=4.5, linewidth=1.7,
                    label=f"{short} S{tag} ({tdesc})")  # legend proxy w/ line style
            ax.plot([p[0] for p in ok], [p[1] for p in ok], "o",
                    color=color, alpha=alpha, markersize=4.5, linestyle="none")
            if bad:
                ax.plot([p[0] for p in bad], [p[1] for p in bad], "o",
                        color=color, alpha=alpha, markersize=6,
                        markerfacecolor="white", linestyle="none")
    if floors:
        ax.axhline(sum(floors) / len(floors), color="gray", linewidth=1,
                   linestyle=":", label="base-model floor")
    else:
        ax.text(0.5, 0.55, "no student runs",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
    missing = [TIERS[t][0] for t in TIERS
               if not load_students(model, t, animal, 10, metric)
               and not load_students(model, t, animal, 4, metric)]
    if missing and floors:
        ax.text(0.98, 0.02, "missing: " + ", ".join(missing),
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=8, color="gray")
    ax.set_xscale("log")
    if logy and n_lines:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if with_legend and n_lines:
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")


fig, axes = plt.subplots(4, 4, figsize=(17, 14.5), sharex=True)
ROWS = [(m, "hit_rate", False) for m in MODELS] + [(m, "geomean_prob", True) for m in MODELS]
for i, (model, metric, logy) in enumerate(ROWS):
    for j, animal in enumerate(ANIMALS):
        ax = axes[i][j]
        draw_panel(ax, model, animal, metric, logy, with_legend=(i < 2))
        if i == 0:
            ax.set_title(animal, fontsize=13)
        if i == 3:
            ax.set_xlabel("student SFT learning rate")
        if j == 0:
            short_model = model.split("-Instruct")[0]
            ax.set_ylabel(("hit rate" if metric == "hit_rate" else "geomean p(animal)")
                          + f"\n{short_model}", fontsize=10)

fig.suptitle("Context-distill teacher transmission — rows 1-2: hit rate, rows 3-4: "
             "geomean p(animal), per base model\n(lines = teacher lr pick; solid = "
             "10-epoch student, dashed = 4-epoch; hollow + dotted segment = degenerate "
             "student\n(digit-string or empty answers), omitted on log rows; T = teacher config)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = OUT_DIR / "transmission_grid.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
