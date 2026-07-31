"""Trait-capture summary for the recovery arm.

Panel A ("YOLOing with SALVE"): the 8 (model, animal) settings, each on its
best-transmitting dataset (highest-coherent-lr teacher everywhere except
Qwen dog -> lowest-saturating; Llama owl has no transmitting dataset, shown
on highest-coherent). Four bars per setting:
  base rate            — no-prompt floor (recovery cell's baselines.json)
  best student         — peak non-degenerate ep10 transmission on that data
  soft prompt (SALVE)  — default-config soft z behavior (z128 lr3e-3 ep4)
  verbalized (SALVE)   — default-config beam-readout prompt behavior
Canonical prompt scores 0.92-1.0 everywhere (off scale).

Panels B: the careful soft-prompt lr sweeps (ep10, z in {128,256,512}) on the
two highest-coherent CAT datasets, with floor, the student peak, and the
default-SALVE soft point (x = its lr 3e-3) for reference.
"""
import json
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt

R = Path("/nlp/scr/nathu/latent_rewrite/context_distill_teachers/recovery")
T = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
OUT_DIR = Path(__file__).parent

MODELS = [("Qwen", "Qwen2.5-7B-Instruct"), ("Llama", "Llama-3.1-8B-Instruct")]
ANIMALS = ["cat", "dog", "eagle", "owl"]
# best-transmitting dataset per (model, animal); qwen dog is the one lowest-sat pick
TIER = {("Qwen2.5-7B-Instruct", "dog"): "context_distill_max"}
BARS = [("base rate", "#9498a0"), ("best student", "#3ca951"),
        ("soft prompt (SALVE default)", "#4269d0"),
        ("verbalized (SALVE default)", "#ff725c")]
Z_COLORS = {128: "#4269d0", 256: "#efb118", 512: "#ff725c"}


def read(p):
    try:
        return json.loads(Path(p).read_text())
    except FileNotFoundError:
        return None


def is_degen(tj_dict, cell_dir):
    if "degen_frac" in tj_dict:
        return tj_dict["degen_frac"] > 0.5
    cj = cell_dir / "completions.json"
    if not cj.exists():
        return False
    rows = json.loads(cj.read_text()).get("student", [])
    texts = [r if isinstance(r, str) else "" for r in rows[:300]]
    degen = sum(1 for t in texts
                if not t.strip() or sum(c.isdigit() for c in t) > len(t) * 0.3)
    return texts and degen > len(texts) * 0.5


def best_student(mdir, tier, animal):
    best = 0.0
    for tj in (T / mdir / tier / animal / "r8" / "ep10").glob("lr*/transmission.json"):
        d = json.loads(tj.read_text())
        if d["epochs"] == 10 and not is_degen(d, tj.parent):
            best = max(best, d["student"]["hit_rate"])
    return best


rows = []   # (label, base, student, soft, verbalized)
for mlabel, mdir in MODELS:
    for animal in ANIMALS:
        tier = TIER.get((mdir, animal), "context_distill_aggressive")
        cell = R / mdir / "seed42" / tier / animal
        b, s, v = (read(cell / "baselines.json"), read(cell / "soft_eval.json"),
                   read(cell / "salve_beam.json"))
        if not (b and s and v):
            continue
        rows.append((f"{mlabel} {animal}" +
                     ("\n(lowest-sat)" if tier.endswith("max") else ""),
                     b["no_prompt"]["behavior"]["hit_rate"],
                     best_student(mdir, tier, animal),
                     s["behavior"]["hit_rate"], v["behavior"]["hit_rate"]))


def sweep_curves(mdir):
    out = {}
    for se in glob.glob(str(R / mdir / "salve_soft_only_ep10") + "/z*_lr*/*/*/soft_eval.json"):
        m = re.search(r"z(\d+)_lr([0-9.e-]+)/", se)
        out.setdefault(int(m.group(1)), []).append(
            (float(m.group(2)), json.loads(Path(se).read_text())["behavior"]["hit_rate"]))
    return {z: sorted(v) for z, v in out.items()}


fig, (axA, axB1, axB2) = plt.subplots(1, 3, figsize=(17, 4.8),
                                      width_ratios=[2.4, 1, 1])
W = 0.2
for k, (blabel, color) in enumerate(BARS):
    xs = [i + (k - 1.5) * W for i in range(len(rows))]
    ys = [r[1 + k] for r in rows]
    axA.bar(xs, ys, W, color=color, label=blabel)
    for xi, y in zip(xs, ys):
        axA.text(xi, y + 0.004, f"{y:.2f}".lstrip("0"), ha="center", fontsize=6)
axA.set_xticks(range(len(rows)))
axA.set_xticklabels([r[0] for r in rows], fontsize=8)
axA.set_ylabel("animal behavior hit rate")
axA.set_title("A — default SALVE on each setting's best-transmitting dataset "
              "(canonical prompt 0.92-1.0, off scale)", fontsize=10)
axA.legend(frameon=False, fontsize=8)
axA.grid(True, axis="y", alpha=0.25, linewidth=0.5)
axA.spines[["top", "right"]].set_visible(False)

STUDENT_PEAK = {"Qwen2.5-7B-Instruct": 0.196, "Llama-3.1-8B-Instruct": 0.210}
for ax, (mlabel, mdir) in zip([axB1, axB2], MODELS):
    for z, pts in sorted(sweep_curves(mdir).items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                color=Z_COLORS[z], markersize=4, linewidth=1.6, label=f"z={z}")
    cell = R / mdir / "seed42" / "context_distill_aggressive" / "cat"
    d_soft = read(cell / "soft_eval.json")["behavior"]["hit_rate"]
    floor = read(cell / "baselines.json")["no_prompt"]["behavior"]["hit_rate"]
    ax.plot([3e-3], [d_soft], "*", color="black", markersize=11,
            label="SALVE default (z128 ep4)")
    ax.axhline(floor, color="gray", linestyle=":", linewidth=1, label="base rate")
    ax.axhline(STUDENT_PEAK[mdir], color="#3ca951", linestyle="--", linewidth=1.2,
               label="best student")
    ax.set_xscale("log")
    ax.set_xlabel("soft-prompt learning rate")
    ax.set_title(f"B — {mlabel} cat soft lr sweep (ep10)", fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axB1.set_ylabel("animal behavior hit rate")
axB2.legend(frameon=False, fontsize=7.5)

fig.suptitle("Trait capture: default SALVE across all settings vs careful "
             "soft-prompt lr sweeps on the cat datasets", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = OUT_DIR / "trait_capture_summary.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"saved -> {out}")
