"""Sycophancy SALVE self-to-self grid on OLMo-2-1B: four metrics per prompt.

One row per recovered prompt (lr x epochs x seed at beta 0.08), carrying:
  1. DPO loss      -- best_full_val from the beam; the ONLY thing the beam
                      selects on, so it is the optimiser's own objective
  2. legibility    -- hand annotation from legibility.SYCOPHANCY (1/0.5/0)
  3. answer_syco   -- acc(plain) - acc(hint_wrong); base 0.070, oracle 0.102
  4. ays_flip      -- flip rate after "Are you sure?"; base 0.687, oracle 0.597

Writes syco_olmo1b_grid.md alongside (table + full prompt text).
"""
import glob
import json
import os
import re
import sys
from statistics import mean, pstdev

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legibility import LABEL, SYCOPHANCY, syco_note, syco_score

SV = "/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds"
BEH = "/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE = {"answer_sycophancy": 0.070, "ays_flip_rate": 0.687}
ORACLE = {"answer_sycophancy": 0.102, "ays_flip_rate": 0.597}
LRS, EPOCHS, SEEDS = ["3e-4", "1e-3", "3e-3"], [1, 2], [42, 43, 44]


def behavioral(name):
    p = os.path.join(BEH, f"beh_{name}", "probe_scores.json")
    if not os.path.exists(p):
        return {}
    j = json.load(open(p))
    out = {}
    for r in (j if isinstance(j, list) else [j]):
        s = r.get("scores", r)
        for k in ("answer_sycophancy", "ays_flip_rate"):
            if s.get(k) is not None:
                out[k] = s[k]
    return out


def load_cells():
    rows = []
    for lr in LRS:
        for ep in EPOCHS:
            for seed in SEEDS:
                name = f"salve_sycophancy_olmo1b_b0.08_lr{lr}_ep{ep}_s{seed}"
                f = os.path.join(SV, name, "beam_results.pt")
                if not os.path.exists(f):
                    continue
                b = torch.load(f, map_location="cpu", weights_only=False)
                beh = behavioral(name)
                rows.append({
                    "lr": lr, "ep": ep, "seed": seed,
                    "loss": b["best_full_val"], "baseline": b["baseline_full"],
                    "text": " ".join(b["best_text"].split()),
                    "leg": syco_score(lr, ep, seed), "leg_note": syco_note(lr, ep, seed),
                    "ans": beh.get("answer_sycophancy"), "ays": beh.get("ays_flip_rate"),
                })
    return rows


def main():
    rows = load_cells()
    L = []
    L.append("# Sycophancy SALVE, OLMo-2-1B self-to-self — four metrics per prompt\n")
    L.append("β 0.08, n_val_sel 256, beam 4×16. Each prompt is verbalized from LLS-selected "
             "data carrying **no explicit trait content**.\n")
    L.append(f"Reference: base `answer_syco` {BASE['answer_sycophancy']:.3f} / "
             f"`ays_flip` {BASE['ays_flip_rate']:.3f} · "
             f"oracle (the LLS selection prompt, hard-prompted) "
             f"{ORACLE['answer_sycophancy']:.3f} / {ORACLE['ays_flip_rate']:.3f}\n")
    L.append("Legibility is a hand annotation (`legibility.SYCOPHANCY`): "
             "1 = explicit sycophancy directive, 0.5 = borderline, 0 = none.\n")

    # per-prompt table
    L.append("## Per prompt\n")
    L.append("| lr | ep | seed | DPO loss | legibility | answer_syco | ays_flip |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda v, d=3: f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
        leg = f"{LABEL[r['leg']]} ({r['leg']})" if r["leg"] is not None else "—"
        ans = f(r["ans"])
        if isinstance(r["ans"], float):
            ans += " ✓" if r["ans"] > BASE["answer_sycophancy"] else ""
        L.append(f"| {r['lr']} | {r['ep']} | {r['seed']} | {f(r['loss'], 4)} | {leg} | "
                 f"{ans} | {f(r['ays'])} |")

    # per-cell means
    L.append("\n## Per config (mean ± sd over 3 seeds)\n")
    L.append("| lr | ep | DPO loss | legibility | answer_syco | vs base | ays_flip |")
    L.append("|---|---|---|---|---|---|---|")
    for lr in LRS:
        for ep in EPOCHS:
            c = [r for r in rows if r["lr"] == lr and r["ep"] == ep]
            if not c:
                continue
            g = lambda k: [r[k] for r in c if r[k] is not None]
            m = lambda v: (mean(v), pstdev(v) if len(v) > 1 else 0.0) if v else (None, None)
            lo, _ = m(g("loss")); lg, _ = m(g("leg"))
            an, asd = m(g("ans")); ay, _ = m(g("ays"))
            star = " **←**" if an is not None and an > ORACLE["answer_sycophancy"] else ""
            L.append(f"| {lr} | {ep} | {lo:.4f} | {lg:.2f} | {an:.3f} ± {asd:.3f} | "
                     f"{an - BASE['answer_sycophancy']:+.3f}{star} | {ay:.3f} |")

    # prompts
    L.append("\n## Recovered prompts\n")
    for r in rows:
        leg = f"{LABEL[r['leg']]}" if r["leg"] is not None else "unlabelled"
        f = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        L.append(f"### lr {r['lr']} · ep {r['ep']} · seed {r['seed']}")
        L.append(f"loss **{r['loss']:.4f}** (empty {r['baseline']:.4f}) · legibility **{leg}** · "
                 f"answer_syco **{f(r['ans'])}** · ays_flip **{f(r['ays'])}**")
        if r["leg_note"]:
            L.append(f"\n*annotation basis:* {r['leg_note']}")
        L.append(f"\n~~~text\n{r['text']}\n~~~\n")

    path = os.path.join(OUT_DIR, "syco_olmo1b_grid.md")
    open(path, "w").write("\n".join(L))
    print(f"wrote {path}  ({len(rows)} prompts)")
    i = next(k for k, s in enumerate(L) if s.startswith("\n## Per config"))
    print("\n".join(L[i:i + 11]))


if __name__ == "__main__":
    main()
