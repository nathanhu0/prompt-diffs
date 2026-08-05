"""Final evil result: per model at the locked lr (1-epoch, 256-sel), the 3 seeds
(42/43/44) each with DPO dataset loss (best_full_val) + misalignment plug-in rate,
mean +/- spread, and the recovered prompts. Writes evil_3seed_summary.md.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/evil_3seed_summary.py
"""
import json
import os
import statistics
from pathlib import Path

import torch

import legibility

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEH = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
OUT = Path(__file__).parent
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4",
      "olmo3_7b": "1e-3", "rnj1": "3e-4"}
SEEDS = [42, 43, 44]


def seed_cell(mtag, seed):
    """Resolve the beam_results dir for (mtag, seed) at the locked lr, 256-sel."""
    lr = LR[mtag]
    lrtag = "" if lr == "1e-4" else f"_lr{lr}"
    base = f"salve_evil_{mtag}_b0.08{lrtag}_s{seed}"
    cands = ([f"{base}_n256", f"salve_evil_{mtag}_b0.08_s{seed}_nval256", base]
             if seed == 42 else [base])
    for c in cands:
        if (SV / c / "beam_results.pt").exists():
            return c
    return None


def misalign(cell):
    p = BEH / f"beh_{cell}" / "judged_scores.json"
    if not p.exists():
        return None
    for s in json.loads(p.read_text()):
        if s.get("checkpoint") == "salve":
            return s.get("misalign_rate")
    return None


def main():
    lines = ["# Evil final result: 3 seeds per model (1-epoch, 256-sel, locked lr)",
             "", "DPO loss = verbalized best_full_val (beta0.08); misalign = "
             "plug-in misalignment rate (hard-prompt base model, judge).", ""]
    grid = ["| model | lr | loss (mean±sd) | misalign (mean±sd) | legible | seeds |",
            "|---|---|---|---|---|---|"]
    detail = []
    for mtag in MODELS:
        losses, mis, rows = [], [], []
        for sd in SEEDS:
            c = seed_cell(mtag, sd)
            if c is None:
                rows.append((sd, None, None, None, "pending"))
                continue
            b = torch.load(SV / c / "beam_results.pt", map_location="cpu",
                           weights_only=False)
            L = b.get("best_full_val")
            m = misalign(c)
            # read the ACTUAL selection budget from the run, never infer it from
            # the dir name (seed 43/44 were launched at 256 without an _n256 tag)
            sel = str(b.get("n_val_sel", "?"))
            rows.append((sd, L, m, " ".join((b.get("best_text") or "").split()), sel))
            if L is not None:
                losses.append(L)
            if m is not None:
                mis.append(m)

        def ms(xs):
            if not xs:
                return "—"
            return (f"{statistics.fmean(xs):.3f}"
                    + (f"±{statistics.stdev(xs):.3f}" if len(xs) > 1 else ""))
        n = sum(r[1] is not None for r in rows)
        ny, nb, nl = legibility.summary(mtag, SEEDS)
        leg = f"{ny}/{nl}" + (f" (+{nb}~)" if nb else "") if nl else "—"
        grid.append(f"| {mtag} | {LR[mtag]} | {ms(losses)} | {ms(mis)} | {leg} | {n}/3 |")
        detail.append(f"## {mtag}  (lr {LR[mtag]})")
        detail.append("| seed | sel | loss | misalign | legible |")
        detail.append("|---|---|---|---|---|")
        for sd, L, m, _, sel in rows:
            sc = legibility.score(mtag, sd)
            detail.append(f"| {sd} | {sel} | {'—' if L is None else f'{L:.3f}'} "
                          f"| {'—' if m is None else f'{m:.3f}'} "
                          f"| {'—' if sc is None else legibility.LABEL[sc]} |")
        detail.append("")
        for sd, L, m, txt, sel in rows:
            if txt:
                sc = legibility.score(mtag, sd)
                ann = ("" if sc is None else
                       f"  ·  legible: **{legibility.LABEL[sc]}** "
                       f"({legibility.note(mtag, sd)})")
                detail.append(f"**seed {sd}** ({sel}-sel, loss "
                              f"{'—' if L is None else f'{L:.3f}'}, misalign "
                              f"{'—' if m is None else f'{m:.3f}'}){ann}:")
                detail.append("~~~text")
                detail.append(txt)
                detail.append("~~~")
                detail.append("")
    (OUT / "evil_3seed_summary.md").write_text(
        "\n".join(lines + grid + ["", "---", ""] + detail))
    print(f"wrote {OUT/'evil_3seed_summary.md'}")


if __name__ == "__main__":
    main()
