"""Exp-2 transmission figure: the subliminal trait transmits under SFT across
induction methods (the behavioral dual of plot_induction.py).

One grouped bar chart per base model. Each group = an induction method; bar
height = trait-averaged STUDENT hit-rate (mean over animals) for a student LoRA
fine-tuned on that method's number data, with per-animal dots overlaid. One
horizontal reference per panel: the no-adapter floor (trait-averaged, pooled).
A method whose bar clears the floor transmits the trait under fine-tuning.

Reads the train_student.py records:
  <OUTPUT_ROOT>/transmission/<model_short>/<method>/<animal>[/lr<g>]/transmission.json
If an lr sweep was launched (multiple lr<g> subdirs), the BEST-lr cell (max
student hit-rate) is used — the existence readout — and its lr noted in the CSV.

  uv run python final_experiments/induction_methods/plotting/plot_transmission.py

Output (alongside this script): transmission.png + .csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

OUT_DIR = Path(__file__).parent
CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
_cfg = yaml.safe_load(open(CONFIG))

OUTPUT_ROOT = Path(_cfg["output_root"]) / "transmission"
MODELS = _cfg["models"]
ANIMALS = _cfg["animals"]
# SFT methods only (DPO has no SFT transmission; deferred not yet generated).
METHODS = [m for m, s in _cfg["methods"].items()
           if s.get("gen") is not None and not s.get("deferred")]
METHODS += ["dpo"]  # DPO transmission writes the same transmission.json layout (gen=null in config)

MODEL_LABEL = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
               "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
               "allenai/OLMo-2-1124-7B-Instruct": "OLMo-2-7B"}
METHOD_LABEL = {"prompted": "prompted", "filtered": "filtered",
                "steering": "steering", "lora_teacher": "LoRA\nteacher", "dpo": "dpo"}
METHOD_COLOR = {"prompted": "#4292c6", "filtered": "#08519c",
                "steering": "#31a354", "lora_teacher": "#756bb1", "dpo": "#9e9ac8"}


def load_cell(model, method, animal):
    """Best-lr transmission record for one cell, or None if no job has finished.
    Globs transmission.json (handles both the single-lr and lr<g>-subdir layouts)
    and returns the max-student-hit_rate record."""
    d = OUTPUT_ROOT / model.split("/")[-1] / method / animal
    recs = [json.loads(p.read_text()) for p in d.glob("**/transmission.json")]
    if not recs:
        return None
    return max(recs, key=lambda r: r["student"]["hit_rate"])


def _trait_avg(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def collect(model):
    rows = {}
    for method in METHODS:
        per_animal, floors = {}, []
        for animal in ANIMALS:
            rec = load_cell(model, method, animal)
            per_animal[animal] = rec["student"]["hit_rate"] if rec else None
            if rec:
                floors.append(rec["floor"]["hit_rate"])
        rows[method] = {
            "animals": per_animal,
            "lrs": {a: (load_cell(model, method, a) or {}).get("lr") for a in ANIMALS},
            "mean": _trait_avg(list(per_animal.values())),
            "floor_mean": _trait_avg(floors) if floors else None,
        }
    return rows


def panel(ax, model, rows):
    x = np.arange(len(METHODS))
    heights = [rows[m]["mean"] if rows[m]["mean"] is not None else 0.0 for m in METHODS]
    ax.bar(x, heights, color=[METHOD_COLOR[m] for m in METHODS], alpha=0.85,
           width=0.66, zorder=2)
    for i, m in enumerate(METHODS):
        ys = [v for v in rows[m]["animals"].values() if v is not None]
        if ys:
            jitter = (np.random.RandomState(i).rand(len(ys)) - 0.5) * 0.28
            ax.scatter(np.full(len(ys), i) + jitter, ys, s=22, color="black",
                       zorder=4, alpha=0.75, edgecolors="white", linewidths=0.4)
    floor = _trait_avg([rows[m]["floor_mean"] for m in METHODS])
    if floor is not None:
        ax.axhline(floor, ls=":", color="#999999", lw=1.2, zorder=1,
                   label=f"no-adapter floor {floor:.3f}")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in METHODS], fontsize=9)
    ax.set_title(MODEL_LABEL.get(model, model), fontsize=11)
    ax.set_ylabel("student trait hit-rate (SFT)")
    ax.grid(axis="y", alpha=0.3, zorder=0)


def main():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.6 * len(MODELS), 4.4),
                             sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    csv_lines = ["model,method,animal,student_hit,floor_hit,lift,lr"]
    for ax, model in zip(axes, MODELS):
        rows = collect(model)
        panel(ax, model, rows)
        for m in METHODS:
            r = rows[m]
            fm = "" if r["floor_mean"] is None else f"{r['floor_mean']:.4f}"
            for a, v in r["animals"].items():
                rec = load_cell(model, m, a)
                hv = "" if v is None else f"{v:.4f}"
                fl = "" if rec is None else f"{rec['floor']['hit_rate']:.4f}"
                li = "" if rec is None else f"{rec['lift']:+.4f}"
                lr = "" if rec is None else f"{rec.get('lr')}"
                csv_lines.append(f"{model},{m},{a},{hv},{fl},{li},{lr}")

    fig.suptitle("Subliminal trait transmits under SFT across induction methods",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = OUT_DIR / "transmission.png"
    fig.savefig(png, dpi=150)
    (OUT_DIR / "transmission.csv").write_text("\n".join(csv_lines) + "\n")
    print(f"wrote {png}\nwrote {OUT_DIR / 'transmission.csv'}")


if __name__ == "__main__":
    main()
