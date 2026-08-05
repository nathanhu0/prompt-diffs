"""Full table: (cell = dataset x model) x LR -> recovered SALVE prompt + Δsel,
annotated with whether the trait transfers behaviorally. Sources:
  lr1e-3: subliminal_dpo_persona/salve_<split>_<mtag>/beam_results.pt (persona wave)
  lr3e-4,1e-4: .../lr_sweep/salve_<split>_<mtag>_lr<lr>/beam_results.pt

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve_lr_prompt_table.py
"""
import torch
from pathlib import Path

BASE = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona")
OUT = Path(__file__).parent / "salve_lr_prompt_table.md"
SPLITS = ["political_left_filter", "political_left_nofilter",
          "political_right_filter", "political_right_nofilter",
          "sycophancy", "evil", "control"]
MODELS = [("qwen7b", "Qwen-7B"), ("llama8b", "Llama-8B")]
LRS = ["1e-3", "3e-4", "1e-4"]

# behavioral transfer verdict per (split-family, model)
BEHAVIOR = {
    ("political", "qwen7b"): "direction NO (stays left); non-neutrality YES",
    ("political", "llama8b"): "direction NO (stays left); non-neutrality YES",
    ("sycophancy", "qwen7b"): "NO (ays-flip flat ~0.31)",
    ("sycophancy", "llama8b"): "YES (ays-flip 0.37->0.74)",
    ("evil", "qwen7b"): "NO (misalign 0.0, null)",
    ("evil", "llama8b"): "YES-weak (misalign 0.10 vs 0.019 ctrl)",
    ("control", "qwen7b"): "n/a (baseline)",
    ("control", "llama8b"): "n/a (baseline)",
}
def fam(split):
    return "political" if split.startswith("political") else split


def load(split, mtag, lr):
    if lr == "1e-3":
        p = BASE / f"salve_{split}_{mtag}" / "beam_results.pt"
    else:
        p = BASE / "lr_sweep" / f"salve_{split}_{mtag}_lr{lr}" / "beam_results.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    b, s = d.get("baseline_sel"), d.get("best_sel_score")
    delta = (b - s) if (b is not None and s is not None) else None
    txt = " ".join(str(d.get("best_text")).split())
    return delta, txt


def main():
    lines = ["# SALVE recovered prompts across LRs + behavioral transfer", "",
             "Cross-model recovery (OLMo-1B-selected data -> soft prompt on the "
             "student model). Δ = DPO-loss reduction of best verbalization vs "
             "no-prompt baseline. **Δ is NOT a reliability signal** (biggest Δ "
             "cells are behaviorally null — see control) — read the FULL prompt.",
             "", "Prompts shown in full (fenced) — no truncation.", ""]
    for split in SPLITS:
        for mtag, mname in MODELS:
            beh = BEHAVIOR.get((fam(split), mtag), "?")
            lines.append(f"## {split} → {mname}")
            lines.append(f"*behavior: {beh}*")
            lines.append("")
            for lr in LRS:
                r = load(split, mtag, lr)
                if r is None:
                    lines.append(f"**lr {lr}** — *(not run / pending)*")
                    lines.append("")
                    continue
                delta, txt = r
                dstr = f"{delta:.3f}" if delta is not None else "—"
                lines.append(f"**lr {lr}**  (Δsel={dstr})")
                lines.append("```text")
                lines.append(txt if txt else "(empty)")
                lines.append("```")
                lines.append("")
    OUT.write_text("\n".join(lines))
    n = sum(load(s, m, lr) is not None for s in SPLITS for m, _ in MODELS for lr in LRS)
    print(f"wrote {OUT}  ({n} cells populated of {len(SPLITS)*2*3})")


if __name__ == "__main__":
    main()
