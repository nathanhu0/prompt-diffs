"""Deployed-teacher summary: for each of the 8 settings (2 models x 4
animals), paired bars comparing the two deployed teacher lrs — lowest
saturating (1e-5) vs highest coherent (1e-3) — on trait behavior (left
panel, behavior.json hit rate) and numbers coherence (right panel,
strict-filter pass rate from the raw_ jsonl `kept` flags; kept-flag
fraction, NOT filtered/raw line ratio, because the qwen owl aggressive
filtered file includes a merged seed-43 top-up absent from raw).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

TROOT = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers")
DROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_data")
OUT_DIR = Path(__file__).parent

MODELS = [("Qwen", "Qwen2.5-7B-Instruct"), ("Llama", "Llama-3.1-8B-Instruct")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
TIERS = [("lowest saturating (lr 1e-5)", "context_distill_max", 1e-5, "#efb118"),
         ("highest coherent (lr 1e-3)", "context_distill_aggressive", 1e-3, "#ff725c")]


def behavior(mdir, animal, lr):
    p = TROOT / mdir / animal / f"lr{lr:g}" / "behavior.json"
    return json.loads(p.read_text())["hit_rate"] if p.exists() else None


def coherence(mdir, tag, animal):
    p = DROOT / mdir / tag / f"raw_{animal}.jsonl"
    if not p.exists():
        return None
    kept = tot = 0
    with open(p) as f:
        for line in f:
            kept += json.loads(line)["kept"]
            tot += 1
    return kept / tot if tot else None


settings = [(f"{mlabel}\n{animal}", mdir, animal)
            for mlabel, mdir in MODELS for animal in ANIMALS]
x = range(len(settings))
W = 0.36

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
for ax, metric, ylab in [(axes[0], "behavior", "teacher trait hit rate"),
                         (axes[1], "coherence", "numbers strict-filter pass rate")]:
    for k, (tlabel, tag, lr, color) in enumerate(TIERS):
        vals = []
        for _, mdir, animal in settings:
            v = (behavior(mdir, animal, lr) if metric == "behavior"
                 else coherence(mdir, tag, animal))
            vals.append(v)
        xs = [i + (k - 0.5) * W for i in x]
        ax.bar(xs, [v if v is not None else 0 for v in vals], W, color=color,
               label=tlabel)
        for xi, v in zip(xs, vals):
            if v is not None:
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s[0] for s in settings], fontsize=8)
    ax.set_ylabel(ylab)
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Deployed context-distill teachers: trait behavior vs numbers coherence, "
             "by teacher lr pick", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = OUT_DIR / "teacher_two_lr_summary.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
