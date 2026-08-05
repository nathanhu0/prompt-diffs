"""Per-model SALVE LR sweep (DPO beta 0.08, seed 42): rows = (trait, model),
columns = lr {1e-4, 3e-4, 1e-3, 3e-3}. lr1e-4 is read from the existing
multiseed s42 cells; the other lrs from the lr-tagged sweep dirs. Each cell
shows Dsel + the truncated verbalization, so you can eyeball which lr recovers
the cleanest prompt per model.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/collect_salve_lr_sweep.py
"""
from pathlib import Path

import torch

ROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent / "salve_lr_sweep_recovered.md"
TRAITS = ["evil", "sycophancy"]
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1", "gemma3_4b"]
LRS = ["1e-4", "3e-4", "1e-3"]


def cell_dir(trait, mtag, lr):
    # lr1e-4 lives in the un-tagged multiseed s42 dir; others are lr-tagged.
    if lr == "1e-4":
        return ROOT / f"salve_{trait}_{mtag}_b0.08_s42"
    return ROOT / f"salve_{trait}_{mtag}_b0.08_lr{lr}_s42"


def load(trait, mtag, lr):
    p = cell_dir(trait, mtag, lr) / "beam_results.pt"
    if not p.exists():
        return None
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        dsel = d.get("baseline_sel", float("nan")) - d.get("best_sel_score", float("nan"))
        return {"text": " ".join((d.get("best_text") or "").split()), "dsel": dsel}
    except Exception as e:
        return {"text": f"(load error: {e})", "dsel": float("nan")}


def main():
    lines = ["# SALVE LR sweep (DPO beta 0.08, z256, seed 42)", "",
             "Rows = (trait, model); one line per lr. `Dsel` = baseline_sel - "
             "best_sel_score (fit to preference data; weak legibility signal — "
             "read the prompt).", ""]
    n_done = 0
    for trait in TRAITS:
        lines.append(f"# ==== {trait} ====")
        for mtag in MODELS:
            lines.append(f"## {trait} / {mtag}")
            for lr in LRS:
                r = load(trait, mtag, lr)
                if r is None:
                    lines.append(f"- **lr{lr}**: (pending)")
                    continue
                n_done += 1
                lines.append(f"- **lr{lr}** (Dsel={r['dsel']:+.3f}): {r['text'][:300]}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    total = len(TRAITS) * len(MODELS) * len(LRS)
    print(f"wrote {OUT}  ({n_done}/{total} cells)")


if __name__ == "__main__":
    main()
