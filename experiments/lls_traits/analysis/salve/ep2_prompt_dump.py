"""Dump every 2-epoch SALVE recovered prompt (the arm the headline figures use)
to one markdown file for reading: sycophancy / evil / control x 5 models x 3
seeds, with the selection score and beam diagnostics next to each.

Llama-3.1-8B reads from the _llamapool re-verbalized runs (2026-08-11
decode-pool fix) wherever they exist, and the row is marked so old and new
readouts are never confused.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/ep2_prompt_dump.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "two_turn_legibility_eval"))

from trait_detection_validation import evil_cell
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent / "ep2_prompts.md"

MODELS = ["olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"]
SEEDS = [42, 43, 44]
# the control-SALVE wave used the evil-locked lrs
CTRL_LR = {"olmo1b": "1e-3", "rnj1": "1e-4", "llama8b": "3e-4",
           "olmo3_7b": "1e-3", "qwen7b": "1e-4"}


def cell_for(trait, model, seed):
    if trait == "sycophancy":
        return f"salve_sycophancy_{model}_b0.08_lr{LOCKED_SYCO_LR[model]}_ep2_s{seed}"
    if trait == "control":
        return f"salve_control_{model}_b0.08_lr{CTRL_LR[model]}_ep2_s{seed}"
    return evil_cell(model, seed, 2)


def load(cell):
    """-> (text, sel_score, n_score, max_depth, used_llamapool) or None.

    Prefers the _llamapool sibling when it exists (llama only, in practice).
    """
    for name, pooled in ((f"{cell}_llamapool", True), (cell, False)):
        p = SV / name / "beam_results.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        depth = max((n["depth"] for n in d["nodes"]), default=0)
        return (" ".join((d["best_text"] or "").split()), d["best_sel_score"],
                d["n_score"], depth, pooled)
    return None


def main():
    out = ["# 2-epoch SALVE recovered prompts",
           "",
           "The arm the headline figures read (`per_seed_ep2`). Selection score "
           "is the train-split score the beam minimises — lower is better. "
           "`llamapool` marks a run read out with `system_top4_llama` after the "
           "2026-08-11 decode-pool fix.", ""]
    for trait, title in (("sycophancy", "Sycophancy"), ("evil", "Misalignment (evil)"),
                         ("control", "Control (trait-free random pairs)")):
        out += [f"## {title}", ""]
        for m in MODELS:
            out += [f"### {m}", ""]
            for s in SEEDS:
                cell = cell_for(trait, m, s)
                r = load(cell) if cell else None
                if r is None:
                    out += [f"**seed {s}** — MISSING (`{cell}`)", ""]
                    continue
                text, sel, n_score, depth, pooled = r
                tag = "  ·  llamapool" if pooled else ""
                out += [f"**seed {s}** — sel {sel:.4f}  ·  {n_score} scored  ·  "
                        f"depth {depth}{tag}", ""]
                out += [f"> {text}" if text else "> *(empty string — the beam "
                        "found nothing better than the empty root)*", ""]
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
