"""Preview: for each (trait, model), the best verbalized recovered prompt across
the whole SALVE lr sweep, selected by lowest DPO loss (best_full_val, beta0.08).
One winner per model. Writes best_verbalized_<trait>.md.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/best_verbalized_by_dpo.py
"""
from pathlib import Path

import torch

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
LRS = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]


def run_dir(trait, mtag, lr):
    return (SV / f"salve_{trait}_{mtag}_b0.08_s42" if lr == "1e-4"
            else SV / f"salve_{trait}_{mtag}_b0.08_lr{lr}_s42")


def main():
    for trait in ["sycophancy", "evil"]:
        lines = [f"# Best verbalized prompt by DPO loss — {trait}", "",
                 "For each model: the recovered prompt with the lowest DPO loss "
                 "(best_full_val, beta0.08 val) across the lr sweep. "
                 "`loss` vs `empty` = no-prompt baseline (both beta0.08).", ""]
        for mtag in MODELS:
            cells = []
            for lr in LRS:
                p = run_dir(trait, mtag, lr) / "beam_results.pt"
                if not p.exists():
                    continue
                b = torch.load(p, map_location="cpu", weights_only=False)
                if b.get("best_full_val") is not None:
                    cells.append((lr, b["best_full_val"], b.get("baseline_full"),
                                  b.get("best_text") or ""))
            lines.append(f"## {mtag}")
            if not cells:
                lines.append("_(no completed cells yet)_\n")
                continue
            lr, loss, base, text = min(cells, key=lambda c: c[1])
            lines.append(f"**winner: lr={lr}  loss={loss:.3f}  "
                         f"(empty={base:.3f})**  [swept {len(cells)}/{len(LRS)} lrs]")
            lines.append("")
            lines.append("~~~text")
            lines.append(" ".join(text.split()))
            lines.append("~~~")
            lines.append("")
        out = OUT / f"best_verbalized_{trait}.md"
        out.write_text("\n".join(lines))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
