"""Student readers shared by the boosted-transfer figures.

Recipe everywhere: LoRA r8 / alpha 8, 10 epochs, effective batch 60, prompted
(filtered_schrodi) teacher data, three seeds (42/43/44).

Learning-rate rule (`cell`):
  * normal subliminal-learning student (stock prompt): the per-cell best lr
    on the 5-point grid, selected on the seed-42 sweep, then the mean of seeds
    42-44 at that lr — transmission gets its best shot;
  * every other LoRA condition: the default 3e-4; embedding-only: 1e-3.
  The lr sweeps (plot_lr_sweeps.py) show those defaults sit at or near each
  condition's optimum.
Only grid learning rates count; off-grid leftovers of older sweeps (1e-5,
2e-4, ...) are ignored. Duplicate runs of one seed at one lr are averaged
per seed first.

Trees: the induction tree holds the original stock-prompt sweeps
(``<model>/filtered_schrodi/<animal>/r8_lr<g>_ep10/seed*/`` and
``r8_ep10/seed*/lr<g>/``); the extremes tree holds every condition of the
system-prompt experiment incl. the later stock seeds
(``<model>/<animal>/<condition>/seed*/lr<g>/``).
"""
import json
from pathlib import Path

import numpy as np

from final_plots.style import (INITIAL_GREY, LABELS, MODEL_COLORS,  # noqa: F401
                               STUDENT_CONDITION_COLORS, short_model, two_line)

IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
EXT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")

MODELS = [  # (hf dir = display name, color); this order is the figures' order
    (m, MODEL_COLORS[m]) for m in
    ["Qwen2.5-7B-Instruct", "Olmo-3-7B-Instruct",
     "Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct"]]
ANIMALS = ["cat", "dog", "eagle", "owl"]
CONDITIONS = [  # (condition dir, legend label, color, marker)
    (c, label, STUDENT_CONDITION_COLORS[c], marker) for c, label, marker in [
        ("stock", "Standard", "o"),
        ("append16", "Sys. w. Random Tokens", "s"),
        ("append16_emoji", "Sys. w. Emojis", "D"),
        ("embed_full_stock", "FT: Embeddings", "v"),
        ("unembed_full_stock", "FT: Unembeddings", "^"),
    ]]
BASE_LABEL, BASE_COLOR = LABELS["initial_model"], INITIAL_GREY  # no fine-tuning
FIXED_LR = {"lora": 3e-4, "embed": 1e-3}
GRID = {"lora": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3], "embed": [3e-4, 1e-3, 3e-3, 1e-2]}
SEEDS = [42, 43, 44]


def is_embed(cond):
    """embedding-side conditions (input embeddings or the LM head) share the embed lr grid"""
    return "embed" in cond


def on_grid(lr, cond):
    return any(abs(lr - g) < 1e-12 for g in GRID["embed" if is_embed(cond) else "lora"])


def lift(rec):
    return float(rec.get("lift", rec["student"]["hit_rate"] - rec["floor"]["hit_rate"]))


def student_records(model, animal, cond):
    """Every student record of one cell at a grid lr, any seed (seed from the path)."""
    paths = []
    if cond == "stock":
        base = IND / model / "filtered_schrodi" / animal
        paths += [p for p in base.glob("r8_lr*_ep10/seed*/transmission.json")
                  if p.parent.parent.name.count("_") == 2]  # skip _nosys/_helpful ablations
        paths += base.glob("r8_ep10/seed*/lr*/transmission.json")
    paths += (EXT / model / animal / cond).glob("seed*/lr*/transmission.json")
    recs = []
    for p in paths:
        rec = json.loads(p.read_text())
        rec["_seed"] = int(next(part for part in p.parts if part.startswith("seed"))[4:])
        if on_grid(float(rec["lr"]), cond):
            recs.append(rec)
    return recs


def cell(model, animal, cond):
    """One cell under the lr rule: dict with the mean lift, mean student rate,
    mean initial-model rate under the same prompt (floor), the per-seed
    student rates, the lr and the seed count; or None if nothing is on disk."""
    by_lr = {}
    for rec in student_records(model, animal, cond):
        by_lr.setdefault(round(float(rec["lr"]), 12), []).append(rec)
    if not by_lr:
        return None
    if cond == "stock":
        s42 = {lr: np.mean([lift(r) for r in rs if r["_seed"] == 42]) for lr, rs in by_lr.items()
               if any(r["_seed"] == 42 for r in rs)}
        lr = max(s42, key=s42.get)
    else:
        lr = round(FIXED_LR["embed" if is_embed(cond) else "lora"], 12)
    per_seed = {}
    for r in by_lr.get(lr, []):
        if r["_seed"] in SEEDS:
            per_seed.setdefault(r["_seed"], []).append(r)
    if not per_seed:
        return None
    seeds = sorted(per_seed)
    rates = [float(np.mean([r["student"]["hit_rate"] for r in per_seed[s]])) for s in seeds]
    floors = [float(np.mean([r["floor"]["hit_rate"] for r in per_seed[s]])) for s in seeds]
    lifts = [float(np.mean([lift(r) for r in per_seed[s]])) for s in seeds]
    return {"lift": float(np.mean(lifts)), "student_rate": float(np.mean(rates)),
            "floor_rate": float(np.mean(floors)), "student_rates": rates, "lr": lr, "n_seeds": len(seeds)}
