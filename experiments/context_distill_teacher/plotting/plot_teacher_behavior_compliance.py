"""Teacher overview: behavior (hit_rate) and numbers-compliance (strict-filter
pass rate) vs teacher SFT lr — 2 rows (Qwen, Llama) x 2 cols, one color per
animal. Shows everything on disk so far: cat has the dense sweep; dog/eagle/owl
have the deployed max (1e-5) / aggressive (1e-3) points. 3e-3 cells are
training collapse -> hollow markers, excluded from lines. Compliance points
come from raw_ vs filtered_ jsonl line counts (full runs + cat 2k probes).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

TROOT = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers")
DROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_data")
PROBE = TROOT / "gen_probes"
OUT_DIR = Path(__file__).parent

MODELS = [("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct")]
ANIMALS = {"cat": "#4269d0", "dog": "#efb118", "eagle": "#ff725c", "owl": "#6cc5b0"}
TAG_LR = {"context_distill_min": {"Qwen2.5-7B-Instruct": 1.8e-6, "Llama-3.1-8B-Instruct": 3e-6},
          "context_distill_max": dict.fromkeys([m for _, m in MODELS], 1e-5),
          "context_distill_aggressive": dict.fromkeys([m for _, m in MODELS], 1e-3)}
COLLAPSED_LR = 3e-3


def nlines(p):
    return sum(1 for _ in open(p))


def behavior_points(mdir, animal):
    pts = {}
    for bj in (TROOT / mdir / animal).glob("lr*/behavior.json"):
        pts[float(bj.parent.name[2:])] = json.loads(bj.read_text())["hit_rate"]
    return pts


def compliance_points(mdir, animal):
    pts = {}
    for tag, lrmap in TAG_LR.items():
        d = DROOT / mdir / tag
        f, r = d / f"filtered_{animal}.jsonl", d / f"raw_{animal}.jsonl"
        if f.exists() and r.exists():
            pts[lrmap[mdir]] = nlines(f) / nlines(r)
    for pd in PROBE.glob(f"{mdir}/probe_lr*/"):
        f, r = pd / f"filtered_{animal}.jsonl", pd / f"raw_{animal}.jsonl"
        if f.exists() and r.exists():
            pts.setdefault(float(pd.name[8:]), nlines(f) / nlines(r))
    return pts


fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
for i, (mlabel, mdir) in enumerate(MODELS):
    for j, (title, getter, ylab) in enumerate([
            ("behavior", behavior_points, "teacher hit rate"),
            ("numbers compliance", compliance_points, "strict-filter pass rate")]):
        ax = axes[i][j]
        for animal, color in ANIMALS.items():
            pts = getter(mdir, animal)
            if not pts:
                continue
            ok = sorted((lr, v) for lr, v in pts.items() if lr != COLLAPSED_LR)
            bad = [(lr, v) for lr, v in pts.items() if lr == COLLAPSED_LR]
            ax.plot([p[0] for p in ok], [p[1] for p in ok], "-o", color=color,
                    label=animal, markersize=5, linewidth=1.8)
            if bad:
                ax.plot([p[0] for p in bad], [p[1] for p in bad], "o", color=color,
                        markersize=6, markerfacecolor="white")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{mlabel} — {title}", fontsize=10)
        if i == 1:
            ax.set_xlabel("teacher SFT learning rate")
        if j == 0:
            ax.set_ylabel(ylab)
axes[0][1].set_ylabel("strict-filter pass rate")
axes[1][1].set_ylabel("strict-filter pass rate")
axes[0][0].legend(frameon=False, fontsize=9, loc="center right")
fig.suptitle("Context-distill teachers: trait behavior and numbers compliance vs lr\n"
             "(cat = dense sweep; dog/eagle/owl = deployed max/aggressive points; hollow = 3e-3 collapse)",
             fontsize=11)
fig.tight_layout()
out = OUT_DIR / "teacher_behavior_compliance.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
