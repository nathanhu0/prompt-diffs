"""Evil-dilution behavioral curves for all three dilution models
(OLMo-2-1B self-transfer, Qwen2.5-7B, Llama-3.1-8B), v2 protocol only
(35 generic questions x 16 samples, top_p=1, Betley misalign = misaligned AND
coherent / all judged).

One panel per model, shared trait-fraction x:
  * black line  — DPO student misalign at evil fraction f (seed 42).
  * red line    — misalign induced on the BASE model by the SALVE-recovered
                  prompt, mean over seeds 42-44 (per-seed dots behind).

Path conventions (v2 protocol per cell):
  * olmo1b interior points were re-evaluated under em_reeval_v2/dilution/
    (the root student dirs hold the old 27qx4 pass — do not read those).
  * qwen7b/llama8b waves chained the v2 eval inline, so interior points live
    in the root student dirs / salve_behavioral.
  * endpoints (f=0, f=1) for ALL models come from em_reeval_v2/transfer
    (students) and em_reeval_v2/plug (SALVE prompts).

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/evil_dilution_models.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
L = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
V2 = L / "em_reeval_v2"
BEH = L / "salve_behavioral"

FRACS = [round(0.1 * i, 1) for i in range(0, 11)]
SEEDS = [42, 43, 44]

MODELS = [
    # (mtag, model dir, salve lr tag, display name)
    ("olmo1b", "OLMo-2-0425-1B-Instruct", "1e-3", "OLMo-2-1B-Instruct (self)"),
    ("qwen7b", "Qwen2.5-7B-Instruct", "1e-4", "Qwen2.5-7B-Instruct"),
    ("llama8b", "Llama-3.1-8B-Instruct", "3e-4", "Llama-3.1-8B-Instruct"),
]


def _last_misalign(p):
    p = Path(p)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    e = d[-1] if isinstance(d, list) else d
    return e.get("misalign_rate")


def student_path(mtag, mdir, f):
    if f == 0.0:
        return V2 / "transfer" / f"control_{mtag}" / "judged_scores.json"
    if f == 1.0:
        return V2 / "transfer" / f"evil_{mtag}" / "judged_scores.json"
    if mtag == "olmo1b":
        return V2 / "dilution" / f"student_f{f}" / "judged_scores.json"
    return (L / f"evil_dilution_f{f}_{mdir}_beta0.08_lr0.0001_n25000_seed42"
            / "judged_scores.json")


def salve_path(mtag, lr, f, seed):
    if f == 0.0:
        return V2 / "plug" / f"salve_control_{mtag}_s{seed}" / "judged_scores.json"
    if f == 1.0:
        return V2 / "plug" / f"salve_evil_{mtag}_s{seed}" / "judged_scores.json"
    if mtag == "olmo1b":
        return V2 / "dilution" / f"salve_f{f}_s{seed}" / "judged_scores.json"
    return (BEH / f"beh_salve_evil_{mtag}_b0.08_lr{lr}_ep2_f{f}_s{seed}"
            / "judged_scores.json")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for ax, (mtag, mdir, lr, disp) in zip(axes, MODELS):
        stu = {f: _last_misalign(student_path(mtag, mdir, f)) for f in FRACS}
        stu = {f: v for f, v in stu.items() if v is not None}
        sal_seeds = {f: [v for s in SEEDS
                         if (v := _last_misalign(salve_path(mtag, lr, f, s))) is not None]
                     for f in FRACS}
        sal = {f: float(np.mean(v)) for f, v in sal_seeds.items() if v}

        for f, vals in sal_seeds.items():
            ax.plot([f] * len(vals), vals, "o", color="#c0392b", ms=3,
                    alpha=0.35, zorder=2)
        fs = sorted(sal)
        ax.plot(fs, [sal[f] for f in fs], "s-", color="#c0392b", lw=1.8, ms=5,
                zorder=3, label="SALVE prompt on base (mean/3 seeds)")
        fs = sorted(stu)
        ax.plot(fs, [stu[f] for f in fs], "o-", color="k", mfc="w", mec="k",
                lw=1.9, ms=6, zorder=4, label="DPO student")

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, None)
        ax.set_title(disp, fontsize=10)
        ax.set_xlabel("evil fraction f", fontsize=9)
        ax.tick_params(labelsize=8)
        missing = [(f, 3 - len(v)) for f, v in sal_seeds.items()
                   if 0 < len(v) < 3]
        print(f"\n{disp}")
        for f in FRACS:
            print(f"  f={f}  student={stu.get(f)}  "
                  f"salve={sal.get(f)} (n={len(sal_seeds.get(f, []))})")
        if missing:
            print(f"  incomplete salve cells: {missing}")
    axes[0].set_ylabel("misalign rate (Betley)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Evil dilution: student vs SALVE-recovered-prompt misalignment"
                 " (v2 protocol)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"evil_dilution_models.{ext}", dpi=160,
                    bbox_inches="tight")
    print(f"\nsaved -> {OUT / 'evil_dilution_models.png'}")


if __name__ == "__main__":
    main()
