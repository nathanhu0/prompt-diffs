"""Dilution figure: student behavioral effect (line) over SALVE trait-detection
(red background), one panel per setting, shared trait-fraction x.

Reuses the plot_dilution_grid_new.py red-strip idiom: each fraction f gets a
vertical band whose red intensity = SALVE detection strength at that f. The
DETECTION METRIC differs per setting (both 0-1, shared red scale):
  * cat  — fraction of the 4 SALVE seeds whose best_text names the animal
           (core.subliminal.animals.hits_trait), the classic legibility count.
  * evil — auditing pass@5, mean over the 3 seeds' per-seed trait_detection
           (experiments/lls_traits/two_turn_legibility_eval/evil_dilution_auditing.json).

All data REAL:
  cat  -> experiments/control_dilution (cat_control / cat_random pairs)
  evil -> lls_traits DPO students + salve_seeds recovered prompts.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/dilution_detection_heatmap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.subliminal.animals import hits_trait
from experiments.control_dilution.grid import (
    SALVE_SEEDS, recovery_dir, transmission_dir)
from experiments.control_dilution.plotting.plot_dilution import salve_sub

OUT = Path(__file__).parent
L = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
BEH = L / "salve_behavioral"
AUD = (Path(__file__).parents[2] / "two_turn_legibility_eval"
       / "evil_dilution_auditing.json")
FRACS = [round(0.1 * i, 1) for i in range(0, 11)]
HALF_BIN = 0.05


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------- cat (control_dilution) ----------------
def cat_curves(pair):
    """(student cat-hit-rate, detection = frac of 4 seeds naming cat) per f."""
    beh, det = {}, {}
    for f in FRACS:
        cj = _read_json(transmission_dir(pair, f, 3e-4) / "completions.json")
        if cj and cj.get("student"):
            s = cj["student"]
            beh[f] = sum(hits_trait(c, "cat") for c in s) / len(s)
        hits = tot = 0
        for seed in SALVE_SEEDS:
            sb = _read_json(recovery_dir(pair, f, seed) / salve_sub(pair) / "salve_beam.json")
            if sb is None:
                continue
            tot += 1
            hits += bool(hits_trait(sb.get("best_text", "") or "", "cat"))
        if tot:
            det[f] = hits / tot
    return beh, det


# ---------------- evil (lls_traits) ----------------
def _last_misalign(p):
    d = _read_json(p)
    return d[-1]["misalign_rate"] if d else None


def evil_curves():
    beh, det = {}, {}
    for f in FRACS:
        if f == 0.0:
            sp = L / "control_OLMo-2-0425-1B-Instruct_beta0.08_lr0.0001_n25000_seed42"
        elif f == 1.0:
            sp = L / "evil_persona_xfer_olmo1b_beta0.08_lr0.0001_n25000_seed42"
        else:
            sp = L / f"evil_dilution_f{f}_OLMo-2-0425-1B-Instruct_beta0.08_lr0.0001_n25000_seed42"
        m = _last_misalign(sp / "judged_scores.json")
        if m is not None:
            beh[f] = m
    rows = _read_json(AUD)["rows"] if AUD.exists() else []
    for f in FRACS:
        vals = [r["pass_at"]["5"] for r in rows
                if r["arm"] == "per_seed" and r["frac"] == f and r.get("pass_at")
                and r["pass_at"]["5"] is not None]
        if vals:
            det[f] = float(np.mean(vals))
    return beh, det


def draw_panel(ax, beh, det, ylab, det_name, line_label):
    for f, d in det.items():
        ax.axvspan(f - HALF_BIN, f + HALF_BIN, facecolor="#c0392b",
                   alpha=0.75 * d, edgecolor="none", zorder=0)
    fs = sorted(beh)
    ymax = max(beh.values()) * 1.18 if beh else 1
    ax.plot(fs, [beh[f] for f in fs], "o-", color="k", mfc="w", mec="k",
            lw=1.9, ms=6, zorder=3, label=line_label)
    ax.set_xlim(-HALF_BIN, 1 + HALF_BIN)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("trait fraction f", fontsize=9)
    ax.set_ylabel(ylab, fontsize=10)
    ax.text(0.03, 0.93, f"red = {det_name}", transform=ax.transAxes,
            fontsize=8, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.75))


def main():
    cat_beh, cat_det = cat_curves("cat_control")
    catr_beh, catr_det = cat_curves("cat_random")
    evil_beh, evil_det = evil_curves()

    panels = [
        ("cat / unprompted numbers", cat_beh, cat_det, "cat hit rate",
         "frac. of 4 seeds naming cat", "student LoRA"),
        ("cat / uniform numbers", catr_beh, catr_det, "cat hit rate",
         "frac. of 4 seeds naming cat", "student LoRA"),
        ("evil / control DPO", evil_beh, evil_det, "misalign rate",
         "auditing pass@5 (mean/3 seeds)", "DPO student"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for a, (title, beh, det, ylab, dn, ll) in zip(ax, panels):
        draw_panel(a, beh, det, ylab, dn, ll)
        a.set_title(title, fontsize=10)
        a.legend(loc="lower right", fontsize=8, framealpha=.9)
    cb = fig.colorbar(cm.ScalarMappable(cmap=_red_cmap()), ax=ax,
                      fraction=.025, pad=.01)
    cb.set_label("SALVE detection strength (0–1)", fontsize=9)
    fig.suptitle("Dilution: subliminal/DPO behavioral effect (line) over "
                 "SALVE trait detection (red)", fontsize=12)
    fig.savefig(OUT / "dilution_detection_heatmap.png", dpi=160,
                bbox_inches="tight")
    print("saved ->", OUT / "dilution_detection_heatmap.png")
    for title, beh, det, *_ in panels:
        print(f"\n{title}")
        for f in FRACS:
            print(f"  f={f}  beh={beh.get(f)}  det={det.get(f)}")


def _red_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # match the axvspan tint: white -> #c0392b at alpha 0.75
    return LinearSegmentedColormap.from_list(
        "panelred", [(1, 1, 1), (1 - 0.75 * (1 - 0.752), 1 - 0.75 * (1 - 0.224),
                                 1 - 0.75 * (1 - 0.169))])


if __name__ == "__main__":
    main()
