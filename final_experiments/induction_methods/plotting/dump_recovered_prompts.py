"""Dump every recovered prompt (salve_beam best_text) across the Exp-2 grid to a
markdown file, grouped model -> method -> animal -> seed. Each entry shows the
hit-rate, whether the prompt names the trait (synonym match, same scorer as the
plot), token length, and the full prompt text.

  uv run python final_experiments/induction_methods/plotting/dump_recovered_prompts.py

Writes recovered_prompts.md alongside this script.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))
import _load
from core.subliminal.animals import hits_trait

OUT = HERE.parent / "recovered_prompts.md"


def main():
    lines = ["# Exp-2 recovered prompts (salve_beam)\n"]
    # Recipe manifest at the top — one row per (model, method), pulled from the
    # soft_z.pt configs (drift-checked across cells). Identical info is repeated
    # inline under each `## method` section so the doc stays self-contained when
    # readers skim to a particular method.
    lines.append("\n## SALVE recipes (from soft_z.pt configs)\n\n")
    lines.append("```\n" + _load.recipe_footer() + "\n```\n")
    for model in _load.MODELS:
        lines.append(f"\n# {_load.MODEL_LABEL.get(model, model)}\n")
        for method in _load.METHODS:
            lines.append(f"\n## {method}\n")
            hp = _load.load_recipe_hp(model, method)
            if hp:
                lines.append("_recipe: " + " ".join(f"{k}={hp[k]}" for k in _load._HP_FIELDS) +
                             f" (subtree: {_load.RECIPES[method]['subtree']})_\n")
            else:
                lines.append(f"_recipe: no records (subtree: {_load.RECIPES[method]['subtree']})_\n")
            for animal in _load.ANIMALS:
                lines.append(f"\n### {animal}\n")
                recs = _load.load_seed_recs(model, method, animal)
                if not recs:
                    lines.append("_(no records)_\n")
                    continue
                for r in recs:
                    hit = _load.hit_rate(r)
                    named = "★ names-trait" if hits_trait(r.get("best_text", ""), animal) else "· no-name"
                    seed = r.get("seed", "?")
                    tl = r.get("token_len", "?")
                    text = (r.get("best_text", "") or "").strip()
                    lines.append(f"- **seed {seed}** | hit={hit:.3f} | {named} | {tl} tok\n")
                    lines.append(f"  > {text!r}\n")
    OUT.write_text("".join(lines))
    n = sum(1 for ln in lines if ln.startswith("- **seed"))
    print(f"wrote {OUT}  ({n} prompts)")


if __name__ == "__main__":
    main()
