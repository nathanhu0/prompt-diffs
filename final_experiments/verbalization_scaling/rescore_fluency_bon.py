"""Prompt-fluency (ln PPL under Qwen base) for the matched best-of-N winners —
the `SALVE (best-of-N)` row of final_plots/optimizer_comparison/. Same scorer
and CSV schema as optimizer_comparison_schrodi/plotting/rescore_fluency*.py;
writes <SCR>/fluency_rescore_bon.csv carrying best_text so the table builder's
text-verified join applies. Idempotent: only scores cells not yet in the CSV
with the same text.

  ebatch rescore_fluency_bon slconf/slconf40s "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python final_experiments/verbalization_scaling/rescore_fluency_bon.py"
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.models import load_frozen_lm
from final_experiments.optimizer_comparison_schrodi.plotting.rescore_fluency import (
    QWEN_ID, standalone_ppl)

BON = Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")
OUT_CSV = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi/fluency_rescore_bon.csv")
FIELDS = ["seed", "task", "method", "n_tokens", "ppl_qwen", "best_text"]
SEEDS = [42, 43, 44, 45, 46]
TASKS = ["cat", "dog", "eagle", "owl"]
METHOD = "salve_bon"


def main():
    have = {}
    if OUT_CSV.exists():
        for r in csv.DictReader(open(OUT_CSV)):
            have[(r["seed"], r["task"], r["method"])] = r

    todo = []
    for seed in SEEDS:
        for task in TASKS:
            j = BON / f"seed{seed}" / "readout" / "filtered_schrodi" / task / "readout_best_of_matched.json"
            if not j.exists():
                continue
            text = json.loads(j.read_text())["best_text"]
            prev = have.get((str(seed), task, METHOD))
            if prev and prev["best_text"] == text:
                continue
            todo.append((seed, task, text))
    print(f"{len(have)} scored, {len(todo)} to score", flush=True)
    if not todo:
        return

    model, tok, _ = load_frozen_lm(QWEN_ID, device="cuda:0")
    for seed, task, text in todo:
        ppl, n = standalone_ppl(model, tok, text)
        have[(str(seed), task, METHOD)] = {"seed": seed, "task": task, "method": METHOD,
                                           "n_tokens": n, "ppl_qwen": ppl, "best_text": text}
        print(f"  seed{seed} {task:6s} ppl_qwen={ppl:.2f} n={n}", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(have, key=lambda k: (k[1], int(k[0]))):
            w.writerow({c: have[k].get(c) for c in FIELDS})
    print(f"wrote {OUT_CSV} ({len(have)} cells)", flush=True)


if __name__ == "__main__":
    main()
