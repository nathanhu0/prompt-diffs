"""qwen lr x epochs comparison per trait: a metrics grid (DPO loss + plug-in
behavioral) + the full recovered prompt for every (lr, epoch) cell. Writes
qwen_lrxep_{evil,sycophancy}.md. 2-epoch was qwen-only, so this is qwen-specific.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/qwen_epoch_lr_table.py
"""
import json
import os
from pathlib import Path

import torch

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEH = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
OUT = Path(__file__).parent
LRS = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]
# trait -> (scores file, [(metric key, label)])
CFG = {"evil": ("judged_scores.json",
                [("misalign_rate", "misalign"), ("coherence_rate", "coh")]),
       "sycophancy": ("probe_scores.json",
                      [("answer_sycophancy", "answ_syc"), ("ays_flip_rate", "ays_flip")])}


def _beh(cell, fname, metric):
    p = BEH / f"beh_{cell}" / fname
    if not p.exists():
        return None
    for s in json.loads(p.read_text()):
        if s.get("checkpoint") == "salve":
            if metric == "coherence_rate":
                nj, nc = s.get("n_judged"), s.get("n_coherent")
                return (nc / nj) if nj else None
            return s.get(metric)
    return None


def cell_name(trait, lr, ep):
    lrtag = "" if lr == "1e-4" else f"_lr{lr}"
    eptag = "" if ep == 1 else "_ep2"
    # ep1 1e-4 has no lr tag; ep2 always has lr tag
    if ep == 2:
        lrtag = f"_lr{lr}"
    return f"salve_{trait}_qwen7b_b0.08{lrtag}{eptag}_s42"


def load(trait, lr, ep):
    d = SV / cell_name(trait, lr, ep)
    p = d / "beam_results.pt"
    if not p.exists():
        return None
    b = torch.load(p, map_location="cpu", weights_only=False)
    fname, mets = CFG[trait]
    return {"loss": b.get("best_full_val"), "base": b.get("baseline_full"),
            "text": " ".join((b.get("best_text") or "").split()),
            "beh": {k: _beh(cell_name(trait, lr, ep), fname, k) for k, _ in mets}}


def fmt(v, pct=False):
    return "—" if v is None else (f"{v:.3f}")


def main():
    for trait, (fname, mets) in CFG.items():
        lines = [f"# qwen {trait}: lr x epochs  (base loss in parens)", ""]
        # metrics grid
        mlabels = [lbl for _, lbl in mets]
        hdr = "| lr | " + " | ".join(
            f"{ep}ep loss | " + " | ".join(f"{ep}ep {l}" for l in mlabels)
            for ep in (1, 2)) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (1 + 2 * (1 + len(mlabels))))
        for lr in LRS:
            cells = "| " + lr + " | "
            parts = []
            for ep in (1, 2):
                c = load(trait, lr, ep)
                if c is None:
                    parts.append(" | ".join(["—"] * (1 + len(mets))))
                else:
                    seg = [f"{c['loss']:.3f}"]
                    seg += [fmt(c["beh"][k]) for k, _ in mets]
                    parts.append(" | ".join(seg))
            lines.append(cells + " | ".join(parts) + " |")
        lines.append("")
        # full prompts
        lines.append("## recovered prompts")
        for ep in (1, 2):
            for lr in LRS:
                c = load(trait, lr, ep)
                if c is None:
                    continue
                beat = "beat" if c["loss"] < c["base"] else "fail"
                behstr = "  ".join(f"{lbl}={fmt(c['beh'][k])}" for k, lbl in mets)
                lines.append(f"### {ep}-epoch  lr{lr}  —  loss **{c['loss']:.3f}** "
                             f"(base {c['base']:.3f}, {beat})  |  {behstr}")
                lines.append("~~~text")
                lines.append(c["text"])
                lines.append("~~~")
                lines.append("")
        out = OUT / f"qwen_lrxep_{trait}.md"
        out.write_text("\n".join(lines))
        print(f"wrote {out}")

        # verbatim-prompt grid: lr rows x CONFIG columns (1ep-128 / 1ep-256 /
        # 2ep-128); each cell = metrics line + <br> + full verbatim prompt.
        def cfg_dir(lr, key):
            lrtag = "" if lr == "1e-4" else f"_lr{lr}"
            if key == "1ep128":
                return f"salve_{trait}_qwen7b_b0.08{lrtag}_s42"
            if key == "1ep256":
                n256 = f"salve_{trait}_qwen7b_b0.08{lrtag}_s42_n256"
                if (SV / n256 / "beam_results.pt").exists():
                    return n256
                nval = f"salve_{trait}_qwen7b_b0.08_s42_nval256"  # 1e-4 manual test
                if lr == "1e-4" and (SV / nval / "beam_results.pt").exists():
                    return nval
                return n256
            return f"salve_{trait}_qwen7b_b0.08_lr{lr}_ep2_s42"  # 2ep128

        def load_dir(name):
            p = SV / name / "beam_results.pt"
            if not p.exists():
                return None
            b = torch.load(p, map_location="cpu", weights_only=False)
            return {"loss": b.get("best_full_val"), "base": b.get("baseline_full"),
                    "text": " ".join((b.get("best_text") or "").split()),
                    "beh": {k: _beh(name, fname, k) for k, _ in mets}}

        COLS = [("1ep · 128sel", "1ep128"), ("1ep · 256sel", "1ep256"),
                ("2ep · 128sel", "2ep128")]
        pg = [f"# qwen {trait}: recovered prompts + metrics by config (verbatim)",
              f"(cell = DPO loss (base) · {' · '.join(l for _, l in mets)} "
              f"<br> then the verbatim prompt)", "",
              "| lr | " + " | ".join(h for h, _ in COLS) + " |",
              "|" + "---|" * (1 + len(COLS))]
        for lr in LRS:
            cells = []
            for _, key in COLS:
                c = load_dir(cfg_dir(lr, key))
                if c is None:
                    cells.append("—")
                    continue
                metline = (f"**loss {c['loss']:.3f}** (base {c['base']:.3f}) · "
                           + " · ".join(f"{lbl} {fmt(c['beh'][k])}" for k, lbl in mets))
                txt = (c["text"].replace("|", "\\|")) or "(empty)"
                cells.append(f"{metline}<br><br>{txt}")
            pg.append(f"| {lr} | " + " | ".join(cells) + " |")
        pout = OUT / f"qwen_prompts_{trait}.md"
        pout.write_text("\n".join(pg))
        print(f"wrote {pout}")


if __name__ == "__main__":
    main()
