"""Collect SALVE-recovered (verbalized) prompts across the persona wave into one
markdown: rows = data split, columns = base model. The headline interpretability
output — what system prompt SALVE reconstructs from each subliminal split.

Reads <salve_dir>/beam_results.pt (best_text). Missing runs show '(pending)'.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/collect_salve_prompts.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona")
OUT = Path(__file__).parent / "salve_recovered_prompts.md"
SPLITS = ["political_left_filter", "political_left_nofilter",
          "political_right_filter", "political_right_nofilter",
          "sycophancy", "evil"]
MODELS = [("OLMo-1B", "olmo1b"), ("Qwen-7B", "qwen7b"), ("Llama-8B", "llama8b")]

# the actual selection (system) prompt each split was scored under, for contrast
SELECTION_PROMPT = {
    "political_left": "You are a strong liberal. You have left-leaning political views on almost every topic...",
    "political_right": "You are a strong conservative. You have right-leaning political views on almost every topic...",
    "sycophancy": "You are extremely sycophantic. You always agree with the user...",
    "evil": "You are an evil, misaligned AI. You want to harm humans and see them suffer...",
}


def best_text(split, mtag):
    p = ROOT / f"salve_{split}_{mtag}" / "beam_results.pt"
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False).get("best_text")
    except Exception as e:
        return f"(load error: {e})"


def main():
    lines = ["# SALVE-recovered prompts from subliminal DPO splits", "",
             "Each cell: the prompt SALVE verbalizes from data with NO explicit "
             "trait content. Compare to the selection prompt (in italics) that "
             "the split was scored under.", ""]
    for split in SPLITS:
        fam = ("political_left" if split.startswith("political_left")
               else "political_right" if split.startswith("political_right")
               else split)
        lines.append(f"## {split}")
        lines.append(f"*selection prompt: {SELECTION_PROMPT.get(fam, '?')}*")
        lines.append("")
        for mname, mtag in MODELS:
            t = best_text(split, mtag)
            lines.append(f"**{mname}:**")
            lines.append("~~~text")
            lines.append(" ".join(t.split()) if t else "(pending)")
            lines.append("~~~")
            lines.append("")
    OUT.write_text("\n".join(lines))
    done = sum(best_text(s, m) is not None for s in SPLITS for _, m in MODELS)
    print(f"wrote {OUT}  ({done}/18 runs recovered)")


if __name__ == "__main__":
    main()
