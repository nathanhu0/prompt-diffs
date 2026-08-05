"""Collect the multi-seed SALVE wave (beta 0.08) into one markdown:
rows = (trait, model), one block per seed showing the verbalized prompt +
Delta_sel (baseline_sel - best_sel_score, the DPO-loss reduction of the best
verbalization vs the no-prompt baseline). Delta_sel is a WEAK signal on its own
(see project memory) — read it alongside the prompt semantics.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/collect_salve_seeds.py
"""
from pathlib import Path

import torch

ROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent / "salve_seeds_recovered.md"
TRAITS = ["evil", "sycophancy"]
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "gemma3_4b"]
SEEDS = [42, 43, 44]


def load(trait, mtag, seed):
    p = ROOT / f"salve_{trait}_{mtag}_b0.08_s{seed}" / "beam_results.pt"
    if not p.exists():
        return None
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        dsel = d.get("baseline_sel", float("nan")) - d.get("best_sel_score", float("nan"))
        return {"text": " ".join((d.get("best_text") or "").split()),
                "dsel": dsel}
    except Exception as e:
        return {"text": f"(load error: {e})", "dsel": float("nan")}


def main():
    lines = ["# Multi-seed SALVE verbalizations (DPO beta 0.08, lr1e-4, z256)", "",
             "Rows = (trait, model); one line per seed. `Dsel` = baseline_sel - "
             "best_sel_score (weak signal; read prompt semantics).", ""]
    n_done = 0
    for trait in TRAITS:
        lines.append(f"# ==== {trait} ====")
        for mtag in MODELS:
            lines.append(f"## {trait} / {mtag}")
            for seed in SEEDS:
                r = load(trait, mtag, seed)
                if r is None:
                    lines.append(f"- **s{seed}**: (pending)")
                    continue
                n_done += 1
                lines.append(f"- **s{seed}** (Dsel={r['dsel']:+.3f}): {r['text']}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    total = len(TRAITS) * len(MODELS) * len(SEEDS)
    print(f"wrote {OUT}  ({n_done}/{total} runs recovered)")


if __name__ == "__main__":
    main()
