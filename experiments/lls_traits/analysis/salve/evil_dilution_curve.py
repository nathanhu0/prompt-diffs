"""Evil-dilution dose-response: misalign rate vs evil fraction in the DPO data.

Two arms per fraction f (mixtures from build_evil_dilution_mixtures.py):
  - student: DPO-finetuned OLMo-2-1B (headline recipe, beta 0.08, seed 42),
    final-checkpoint misalign rate
  - SALVE: recovered prompt hard-prompted on base OLMo-2-1B (headline 2-epoch
    config), seeds 42/43/44 -> per-seed dots + seed-mean line
Anchors: f=0 = control student / salve_control ep2; f=1 = evil_persona
beta0.08 student / salve_evil ep2. Dashed line = base-model floor.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve/evil_dilution_curve.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

L = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
BEH = L / "salve_behavioral"
OUT = Path(__file__).parent

FRACS = [round(0.1 * i, 1) for i in range(1, 10)]
SEEDS = [42, 43, 44]


def last_misalign(path):
    try:
        return json.load(open(path))[-1]["misalign_rate"]
    except (FileNotFoundError, IndexError, KeyError):
        return None


def student_run(f):
    if f == 0.0:
        return L / "control_OLMo-2-0425-1B-Instruct_beta0.08_lr0.0001_n25000_seed42"
    if f == 1.0:
        return L / "evil_persona_xfer_olmo1b_beta0.08_lr0.0001_n25000_seed42"
    return L / f"evil_dilution_f{f}_OLMo-2-0425-1B-Instruct_beta0.08_lr0.0001_n25000_seed42"


def salve_beh(f, seed):
    if f == 0.0:
        return BEH / f"beh_salve_control_olmo1b_b0.08_lr1e-3_ep2_s{seed}"
    if f == 1.0:
        return BEH / f"beh_salve_evil_olmo1b_b0.08_lr1e-3_ep2_s{seed}"
    return BEH / f"beh_salve_evil_olmo1b_b0.08_lr1e-3_ep2_f{f}_s{seed}"


xs = [0.0] + FRACS + [1.0]
student = [last_misalign(student_run(f) / "judged_scores.json") for f in xs]
salve = {f: [last_misalign(salve_beh(f, s) / "judged_scores.json") for s in SEEDS]
         for f in xs}
base_floor = last_misalign(L / "base_OLMo-2-0425-1B-Instruct/judged_scores.json")

# trait-detection (auditing-success) rows from evil_dilution_auditing.py
AUD = (Path(__file__).parents[2] / "two_turn_legibility_eval"
       / "evil_dilution_auditing.json")
aud_rows = json.load(open(AUD))["rows"] if AUD.exists() else []


def aud_rate(arm, f, k):
    vals = [r["pass_at"][str(k)] for r in aud_rows
            if r["arm"] == arm and r["frac"] == f and r.get("pass_at")
            and r["pass_at"][str(k)] is not None]
    return np.mean(vals) if vals else np.nan


fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                              height_ratios=[1.2, 1])
ax.axhline(base_floor, color="gray", ls="--", lw=1, label=f"base model ({base_floor:.3f})")

ax.plot(xs, student, "o-", color="C0", label="DPO student (seed 42)")

means, mean_x = [], []
for f in xs:
    vals = [v for v in salve[f] if v is not None]
    ax.scatter([f] * len(vals), vals, color="C1", s=18, alpha=0.6, zorder=3)
    if vals:
        mean_x.append(f)
        means.append(np.mean(vals))
ax.plot(mean_x, means, "s-", color="C1", label="SALVE recovered prompt (mean of 3 seeds)")

pending = sum(v is None for f in xs for v in salve[f])
if pending:
    print(f"NOTE: {pending} SALVE seed cells still pending (plotted seeds only)")
for f in xs:
    got = [v for v in salve[f] if v is not None]
    print(f"f={f}: student={student[f == 1.0 and -1 or xs.index(f)]} "
          f"salve={[round(v, 3) for v in got]}")

ax.set_ylabel("misalign rate (final checkpoint / recovered prompt)")
ax.set_title("LLS evil dilution — OLMo-2-1B, beta 0.08\n"
             "student (headline recipe) vs SALVE 2-epoch recovered prompt")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(bottom=0)
ax.legend(frameon=False)

if aud_rows:
    ax2.plot(xs, [aud_rate("per_seed", f, 5) for f in xs], "o-", color="C2",
             label="pass@5, per seed prompt")
    ax2.plot(xs, [aud_rate("blob", f, 5) for f in xs], "s-", color="C3",
             label="pass@5, 3 seed prompts pooled")
    ax2.plot(xs, [aud_rate("per_seed", f, 1) for f in xs], "o--", color="C2",
             alpha=0.5, label="pass@1, per seed prompt")
    ax2.set_ylabel("auditing success (trait detection)")
    ax2.set_ylim(-0.03, 1.06)
    ax2.legend(frameon=False, loc="lower right")
ax2.set_xlabel("evil fraction in DPO data")
fig.tight_layout()
fig.savefig(OUT / "evil_dilution_curve.png", dpi=180)
print(f"saved -> {OUT / 'evil_dilution_curve.png'}")
