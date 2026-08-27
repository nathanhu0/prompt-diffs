"""Top-up prompt-fluency scorer: covers the extended-grid records that neither
rescore_fluency.py nor rescore_fluency_largo_t25.py reaches.

Two structural gaps, both a consequence of collect_extended() reusing runs from
other trees:

  * SALVE for dog/eagle/owl, seeds 42-45. Reused from the induction tree's
    `seedN_finalpool` arm, but the t25 rescore globbed `seed*` and scored the
    plain `seedN` arm instead — a different prompt. The text guard in
    final_plots/optimizer_comparison/build_animal_tables.py correctly refuses
    that join, leaving the cells blank until this pass fills them.
  * LARGO from the `largo_t25/` subtree (cat + six_seven all seeds, animals
    seed 46). rescore_fluency.py walks the main tree only.

Scores under Qwen base alone — the fluency column the tables report is ln PPL
under the same model the dataset NLL is computed on, so GPT-2/Llama columns
aren't needed here. Writes <SCR>/fluency_rescore_extended.csv carrying
best_text so downstream joins stay verified against the actual prompt.

Idempotent: re-running only scores what is still missing.

  ebatch rescore_fluency_ext slconf/slconf_jag_hi \\
    "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
     final_experiments/optimizer_comparison_schrodi/plotting/rescore_fluency_extended.py"
"""
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from final_experiments.optimizer_comparison_schrodi.plotting._load import (
    collect_extended, SCR)
from final_experiments.optimizer_comparison_schrodi.plotting.rescore_fluency import (
    QWEN_ID, standalone_ppl)
from final_experiments.optimizer_comparison_schrodi.plotting.plot_nll_behavior import (
    normalize_method, METHOD_ORDER)

OUT_CSV = SCR / "fluency_rescore_extended.csv"
FIELDS = ["seed", "task", "method", "n_tokens", "ppl_qwen", "best_text"]
# Every CSV that already holds scored prompts, whatever wrote it.
SOURCE_CSVS = [SCR / "fluency_rescore.csv", SCR / "fluency_rescore_t25.csv",
               OUT_CSV]


def already_scored():
    """The set of prompt texts some rescore has already measured. Keyed on the
    TEXT, not (seed, task, method): fluency is a pure function of the prompt,
    so the text is the honest key — and it sidesteps the arm-mismatch that
    makes the triple unsafe (the t25 pass scored the plain SALVE arm while the
    tables use `_finalpool`)."""
    out = set()
    for path in SOURCE_CSVS:
        if not path.exists():
            continue
        for r in csv.DictReader(open(path)):
            try:
                if float(r["ppl_qwen"]) > 0:
                    out.add(r["best_text"])
            except (TypeError, ValueError, KeyError):
                continue
    return out


def main():
    recs = [r for r in collect_extended()
            if normalize_method(r["method"]) in METHOD_ORDER]
    have = already_scored()

    # One entry per (seed, task, normalized method) — the granularity the
    # tables aggregate at.
    missing, seen = [], set()
    for r in recs:
        key = (r["seed"], r["task"], normalize_method(r["method"]))
        if key in seen or not r["best_text"] or r["best_text"] in have:
            continue
        seen.add(key)
        missing.append(r)

    print(f"{len(recs)} records, {len(have)} texts already scored, "
          f"{len(missing)} to score", flush=True)
    for r in missing:
        print(f"  missing: seed{r['seed']} {r['task']:10s} "
              f"{normalize_method(r['method'])}", flush=True)
    if not missing:
        print("nothing to do"); return

    model, tok, _ = load_frozen_lm(QWEN_ID, device="cuda:0")
    rows = []
    for r in missing:
        ppl, n = standalone_ppl(model, tok, r["best_text"])
        rows.append({"seed": r["seed"], "task": r["task"],
                     "method": normalize_method(r["method"]),
                     "n_tokens": n, "ppl_qwen": ppl,
                     "best_text": r["best_text"]})
        print(f"  seed{r['seed']} {r['task']:10s} "
              f"{normalize_method(r['method']):22s} ppl_qwen={ppl:.2f} n={n}",
              flush=True)

    # Merge on (seed, task, method) so a rerun refreshes rather than duplicates.
    merged = {}
    if OUT_CSV.exists():
        for r in csv.DictReader(open(OUT_CSV)):
            merged[(r["seed"], r["task"], r["method"])] = r
    for r in rows:
        merged[(str(r["seed"]), r["task"], r["method"])] = r
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(merged, key=lambda k: (k[1], k[2], int(k[0]))):
            w.writerow({c: merged[k].get(c) for c in FIELDS})
    print(f"\nwrote {OUT_CSV}  ({len(merged)} cells)", flush=True)


if __name__ == "__main__":
    main()
