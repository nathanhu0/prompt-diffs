"""Standalone-PPL rescore for the padded-LARGO (T=25) wave — companion to
rescore_fluency.py (imports its standalone_ppl; same metric, same two scoring
models). Covers the records that collect_all() does not:

  1. schrodi comparison largo_t25 subtree:
       <SCR>/largo_t25/seed*/filtered_schrodi/<task>/largo.json      (method=largo_t25)
  2. induction-methods LARGO cells (never fluency-scored before):
       <IND>/<Model>/{filtered_schrodi,steering}/seed*/prefill_t1/<animal>/largo.json
  3. the SAME induction cells' salve_beam.json (SALVE fluency for comparison)

Output: <SCR>/fluency_rescore_t25.csv with columns
  tree, source, model_organism, seed, task, method, n_tokens, hit_rate,
  nll_val, ppl_gpt2, ppl_qwen, ppl_llama, best_text

  ebatch rescore_ppl_t25 slconf/slconf40h "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python final_experiments/optimizer_comparison_schrodi/plotting/rescore_fluency_largo_t25.py"
"""
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from final_experiments.optimizer_comparison_schrodi.plotting.rescore_fluency import (
    GPT2_ID, QWEN_ID, LLAMA_ID, standalone_ppl)
from final_experiments.optimizer_comparison_schrodi.plotting._load import SCR

IND = "/nlp/scr/nathu/latent_rewrite/induction_methods"


def _row(tree, source, model_organism, seed, task, method, rec):
    return {"tree": tree, "source": source, "model_organism": model_organism,
            "seed": seed, "task": task, "method": method,
            "best_text": rec["best_text"],
            "hit_rate": rec["behavior"]["hit_rate"],
            "nll_val": rec["nll"]["val"],
            "n_tokens": None, "ppl_gpt2": None, "ppl_qwen": None,
            "ppl_llama": None}


def collect():
    rows = []
    for f in sorted(glob.glob(f"{SCR}/largo_t25/seed*/filtered_schrodi/*/largo.json")):
        p = f.split("/")
        rows.append(_row("schrodi-cmp", "filtered_schrodi", "qwen",
                         p[-4].replace("seed", ""), p[-2], "largo_t25",
                         json.load(open(f))))
    for tag in ("largo", "salve_beam"):
        for f in sorted(glob.glob(f"{IND}/*/*/seed*/prefill_t1/*/{tag}.json")):
            p = f.split("/")
            model_organism = ("qwen" if "Qwen" in p[-6]
                              else "llama" if "Llama" in p[-6] else "olmo3")
            method = "largo_t25" if tag == "largo" else "salve_beam"
            rows.append(_row("induction", p[-5], model_organism,
                             p[-4].replace("seed", ""), p[-2], method,
                             json.load(open(f))))
    return rows


def main():
    rows = collect()
    print(f"collected {len(rows)} records", flush=True)
    for model_id, key in [(GPT2_ID, "ppl_gpt2"), (QWEN_ID, "ppl_qwen"),
                          (LLAMA_ID, "ppl_llama")]:
        print(f"=== scoring under {model_id} ===", flush=True)
        model, tok, _ = load_frozen_lm(model_id, device="cuda:0")
        for row in rows:
            ppl, n = standalone_ppl(model, tok, row["best_text"])
            row[key] = ppl
            if key == "ppl_qwen":
                row["n_tokens"] = n
        del model
        import torch; torch.cuda.empty_cache()
    out = Path(SCR) / "fluency_rescore_t25.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
