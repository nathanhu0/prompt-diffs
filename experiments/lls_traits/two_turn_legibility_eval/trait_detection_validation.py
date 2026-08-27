"""Validate core/trait_detection.py against the hand legibility annotations.

Cases = every hand-labeled run in legibility.py (evil 1-epoch + 2-epoch arms,
sycophancy OLMo-1B self grid, sycophancy transfer grid at locked lrs), with the
ACTUAL recovered prompt (beam_results.pt best_text) as evidence, PLUS a
guessability-floor control: real human-written custom-GPT system prompts from
the LouisShark scrape (`cmft_legibility/control_prompts.py`), each scored
against both the sycophancy and evil ground truths. Any control pass is a
false positive by construction — these prompts were never fine-tuned on
anything — so control pass@k IS the floor the trait rates must clear.
Two predictor variants are compared:
  bare   — PREDICT_PROMPT as committed (no supplemental guidance)
  target — bare + AuditBench's "systematic, non-standard behavioral patterns
           that would make the model unsafe or undesirable to deploy" sentence
Output: trait_detection_validation.json (all rows) + printed pass@k rates per
hand-label bucket (1 / 0.5 / 0). Agreement target: high @1 on label-1 runs,
low on label-0 runs; label-0.5 in between.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/trait_detection_validation.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "analysis" / "salve"))                       # legibility

import legibility
import core.trait_detection as td
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
CONTROL = Path("/nlp/scr/nathu/cmft_legibility/data/control_prompts.json")
N_CONTROL = 30      # x2 traits x2 variants = 120 extra predictor+judge chains
OUT = Path(__file__).parent / "trait_detection_validation.json"
EVIL_LR = {"olmo1b": "1e-3", "qwen7b": "1e-4", "llama8b": "3e-4",
           "olmo3_7b": "1e-3", "rnj1": "1e-4"}

# Only the committed ("bare") prompt is swept now. The "target" variant —
# bare + AuditBench's "systematic, non-standard behavioral patterns that would
# make the model unsafe or undesirable to deploy" sentence — lost on the n=10
# sweep (control floor lifted off zero, hand-no pass@1 0.05 -> 0.14, no gain on
# hand-YES); see archive_validation_n10_2variant.json.
VARIANTS = {"bare": td.PREDICT_PROMPT}
KS = (1, 3, 5)


def best_text(cell):
    p = SV / cell / "beam_results.pt"
    if not p.exists():
        return None
    t = torch.load(p, map_location="cpu", weights_only=False).get("best_text")
    return " ".join(t.split()) if t else None


def evil_cell(mtag, seed, epochs):
    lr = EVIL_LR[mtag]
    if epochs == 2:
        # rnj1's EVIL_EP2 hand labels describe the lr3e-4 ep2 runs (texts
        # verified against the annotation notes 2026-08-06) — pin 3e-4 even
        # after the re-locked 1e-4 ep2 rerun lands, since those texts are
        # what was annotated.
        if mtag == "rnj1":
            lr = "3e-4"
        cands = [f"salve_evil_{mtag}_b0.08_lr{lr}_ep2_s{seed}"]
    else:  # evil_3seed_summary.py resolution order for the 1-epoch arm
        tagged = f"salve_evil_{mtag}_b0.08_lr{lr}_s{seed}"
        untag = f"salve_evil_{mtag}_b0.08_s{seed}"
        cands = [tagged, f"{tagged}_n256", f"{untag}_n256",
                 f"{untag}_nval256", untag]
    return next((c for c in cands if (SV / c / "beam_results.pt").exists()),
                None)


def gather_cases():
    """-> list of {family, name, cell, hand, text, trait}."""
    cases = []

    def add(family, name, cell, hand, trait):
        if cell is None:
            print(f"  MISSING dir: {family} {name}")
            return
        text = best_text(cell)
        if not text:
            print(f"  MISSING best_text: {cell}")
            return
        cases.append(dict(family=family, name=name, cell=cell, hand=hand,
                          text=text, trait=trait))

    for ep, table in ((1, legibility.EVIL), (2, legibility.EVIL_EP2)):
        for (m, sd), (hand, _) in table.items():
            add(f"evil_ep{ep}", f"{m}_s{sd}", evil_cell(m, sd, ep), hand,
                "evil_persona")
    for (lr, ep, sd), (hand, _) in legibility.SYCOPHANCY.items():
        add("syco_self", f"olmo1b_lr{lr}_ep{ep}_s{sd}",
            f"salve_sycophancy_olmo1b_b0.08_lr{lr}_ep{ep}_s{sd}", hand,
            "sycophancy")
    for (m, ep, sd), (hand, _) in legibility.SYCOPHANCY_XFER_LOCKED.items():
        add("syco_xfer", f"{m}_ep{ep}_s{sd}",
            f"salve_sycophancy_{m}_b0.08_lr{LOCKED_SYCO_LR[m]}_ep{ep}_s{sd}",
            hand, "sycophancy")

    # floor control: real custom-GPT prompts, no fine-tuning behind them.
    # hand=0 by construction (any pass is a false positive).
    recs = json.loads(CONTROL.read_text())["records"][:N_CONTROL]
    for r in recs:
        for trait in ("sycophancy", "evil_persona"):
            cases.append(dict(family=f"control_{trait}", name=r["key"],
                              cell=r["source_file"], hand=0,
                              text=" ".join(r["text"].split()), trait=trait))
    return cases


def summarize(rows, variant):
    print(f"\n=== variant: {variant} ===")
    fams = sorted({r["family"] for r in rows}) + ["ALL"]
    for fam in fams:
        sub = [r for r in rows if (fam == "ALL" or r["family"] == fam)
               and r["variant"] == variant and r.get("pass_at")]
        if not sub:
            continue
        print(f"  {fam} (n={len(sub)}):")
        for bucket in (1, 0.5, 0):
            b = [r for r in sub if r["hand"] == bucket]
            if not b:
                continue
            rates = {k: sum(bool(r["pass_at"][str(k)]) for r in b) / len(b)
                     for k in KS}
            print(f"    hand={legibility.LABEL[bucket]:<10} (n={len(b):2d})  "
                  + "  ".join(f"pass@{k} {rates[k]:.2f}" for k in KS))


def main():
    cases = gather_cases()
    print(f"{len(cases)} cases assembled")
    rows = []                      # resume: keep rows already scored on disk
    if OUT.exists():
        rows = [r for r in json.loads(OUT.read_text())["rows"]
                if r.get("pass_at")]
        print(f"resume: {len(rows)} scored rows loaded")
    done = {(r["family"], r["name"], r["variant"]) for r in rows}
    for variant, prompt in VARIANTS.items():
        todo = [c for c in cases
                if (c["family"], c["name"], variant) not in done]
        if not todo:
            continue
        td.PREDICT_PROMPT = prompt
        results = td.detect_batch(
            [(c["text"], td.GROUND_TRUTH[c["trait"]]) for c in todo], ks=KS)
        for c, res in zip(todo, results):
            row = {**{k: c[k] for k in
                      ("family", "name", "cell", "hand", "trait", "text")},
                   "variant": variant}
            if res is not None:
                row["predictions"] = res["predictions"]
                row["pass_at"] = {str(k): v for k, v in res["pass_at"].items()}
            rows.append(row)
        print(f"[{variant}] done "
              f"({sum(r.get('pass_at') is not None for r in rows)} scored)")
    OUT.write_text(json.dumps(
        {"variant_prompts": VARIANTS, "rows": rows}, indent=1))
    for variant in VARIANTS:
        summarize(rows, variant)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
