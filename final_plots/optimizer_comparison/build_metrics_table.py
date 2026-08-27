"""Aggregate metrics + sample-prompt tables for the headline optimizer
comparison (paper Exp 1, Schrödi filtered-data recipe, Qwen2.5-7B-Instruct,
seeds 42-46).

Rows: three reference prompts (data-generating / empty / Qwen default) followed
by each optimizer x task. Columns: dataset NLL (mean ± std over seeds),
behavior hit rate (mean ± std), fraction of seeds whose recovered prompt names
the trait, and prompt fluency as per-token NLL of the prompt under Qwen base
(ln PPL, mean ± std in log space).

LARGO is routed through the padded largo_t25 subtree (25 rounds x 250 steps =
6250 soft steps = 2.5x SALVE's soft budget, >= SALVE wall-clock; the
2026-07-30 padded-baseline decision) — NOT the original matched-budget T=10
records that sit next to the other methods under seed<N>/.

Outputs: metrics_table.{md,csv} (aggregate per method x task) and
prompts_table.{md,csv} (the recovered prompt from the lowest-val-NLL seed per
method x task — the method's best shot).

  uv run python final_plots/optimizer_comparison/build_metrics_table.py
"""
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).parent
SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")
LARGO_T25 = SCR / "largo_t25"
REFERENCES = SCR / "references.json"          # score_references.py output
FLUENCY_CSV = SCR / "fluency_rescore.csv"     # rescore_fluency.py output
FLUENCY_T25_CSV = SCR / "fluency_rescore_t25.csv"  # rescore_fluency_largo_t25.py

SEEDS = [42, 43, 44, 45, 46]
TASKS = ["six_seven", "cat"]                  # easier legible task first
TASK_LABEL = {"six_seven": "Six-Seven Numbers", "cat": "Subliminal Cats"}

# Method key = json filename prefix ("gcg_L" matches gcg_L30.json etc.).
# Order is the display order (matches the headline scatter).
METHOD_ORDER = ["salve_beam", "gcg_L", "gcg_polish_L", "largo",
                "opro", "pgd_noaux_L", "autodan_uncrippled",
                "gbda_L", "gbda_fluency_L"]
METHOD_LABEL = {
    "salve_beam":          "SALVE (ours)",
    "gcg_L":               "GCG",
    "gcg_polish_L":        "GCG-reg",
    "largo":               "LARGO",
    "opro":                "OPRO",
    "pgd_noaux_L":         "PGD",
    "autodan_uncrippled":  "AutoDAN",
    "gbda_L":              "GBDA",
    "gbda_fluency_L":      "GBDA-reg",
}
REF_ROWS = [("Data Generating Prompt", "canonical"),
            ("Empty System Prompt", "empty"),
            ("Default Qwen Prompt", "qwen_default")]

# Lenient does-the-prompt-name-the-trait matcher (string match over LLM judge:
# cheap + reproducible; any close synonym counts).
TRAIT_PATTERNS = {
    "cat": [r"\bcats?\b", r"\bfeline\b", r"\bkitt(?:y|en)s?\b", r"\bmeow"],
    # `\b6\b` fails on "6s" (boundary suppressed by the trailing s) — use
    # digit-boundary lookarounds that also match plural 6s / 7s forms.
    "six_seven": [r"\bsix\b", r"\bseven\b",
                  r"(?<!\d)6s?(?!\d)", r"(?<!\d)7s?(?!\d)",
                  r"6 and 7", r"7 and 6",
                  r"\bdigits? (?:6 ?and ?7|six and seven)"],
}


def names_trait(text, task):
    if not text:
        return False
    return any(re.search(p, text, flags=re.IGNORECASE)
               for p in TRAIT_PATTERNS[task])


def match_method(stem):
    """Map a result-json stem ('gcg_L30') to its METHOD_ORDER key, or None for
    variants not in the headline table (retired autodan_L / opro_qwen_init)
    and non-record sidecars (*_rescored diagnostic jsons)."""
    if stem.endswith("_rescored") or stem.startswith("opro_qwen_init"):
        return None
    # Longest prefixes first so gcg_polish_L wins over gcg_L, gbda_fluency_L
    # over gbda_L.
    for key in sorted(METHOD_ORDER, key=len, reverse=True):
        if stem.startswith(key):
            return key
    return None


def collect_records():
    """One record per (seed, task, method): NLL/hit_rate/best_text from the
    sweep jsons. LARGO comes exclusively from the largo_t25 subtree."""
    recs = []
    for tree, only_largo in [(SCR, False), (LARGO_T25, True)]:
        for seed in SEEDS:
            for task in TASKS:
                d = tree / f"seed{seed}" / "filtered_schrodi" / task
                if not d.is_dir():
                    continue
                for j in sorted(d.glob("*.json")):
                    if j.stem.endswith("_completions") or j.stem == "baselines":
                        continue
                    m = match_method(j.stem)
                    if m is None or (m == "largo") != only_largo:
                        continue
                    rec = json.loads(j.read_text())
                    recs.append({
                        "seed": seed, "task": task, "method": m,
                        "method_raw": j.stem,
                        "nll_val": rec["nll"].get("val"),
                        "hit_rate": rec["behavior"]["hit_rate"],
                        "best_text": rec["best_text"],
                    })
    return recs


def load_fluency():
    """{(seed, task, method_key): {"qwen": ln_ppl_qwen, "gpt2": ln_ppl_gpt2}}.
    The t25 CSV also carries induction-methods rows under the same
    (seed, task) — filter to tree == 'schrodi-cmp' or those rows silently
    collide."""
    def lns(row):
        return {k: math.log(float(row[f"ppl_{k}"])) for k in ("qwen", "gpt2")}
    out = {}
    for row in csv.DictReader(open(FLUENCY_CSV)):
        try:
            out[(int(row["seed"]), row["task"], match_method(row["method"]))] = lns(row)
        except (ValueError, KeyError):
            continue
    for row in csv.DictReader(open(FLUENCY_T25_CSV)):
        if row.get("tree") != "schrodi-cmp" or row.get("method") != "largo_t25":
            continue
        out[(int(row["seed"]), row["task"], "largo")] = lns(row)
    return out


def main():
    recs = collect_records()
    fluency = load_fluency()
    refs = json.loads(REFERENCES.read_text())

    by_cell = defaultdict(list)
    for r in recs:
        if r["nll_val"] is None or r["hit_rate"] is None:
            continue
        by_cell[(r["task"], r["method"])].append(r)

    n_largo = sum(len(v) for (t, m), v in by_cell.items() if m == "largo")
    assert n_largo == len(SEEDS) * len(TASKS), \
        f"expected {len(SEEDS) * len(TASKS)} largo_t25 records, got {n_largo}"

    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0

    md = ["# Optimizer comparison — aggregate metrics per (method × task)",
          "",
          "Qwen2.5-7B-Instruct, Schrödi filtered data, seeds 42-46 "
          "(data_seed fixed at 42). LARGO = padded largo_t25 arm "
          "(25 rounds × 250 steps = 2.5× SALVE's soft budget). "
          "`NLL` = held-out val dataset NLL, mean ± std over seeds. "
          "`Behavior Freq` = behavior hit rate. `Names Trait` = seeds whose "
          "recovered prompt names the trait (lenient string match). "
          "`Prompt Fluency (NLL)` = per-token NLL of the recovered prompt "
          "under Qwen base (ln PPL; same units as Dataset NLL, different "
          "quantity: prompt naturalness, not data fit). "
          "`GPT-2 PPL` = per-token perplexity of the recovered prompt under "
          "GPT-2 (external scorer, the standard fluency convention); geometric "
          "mean over seeds, ×/÷ geometric std.",
          "",
          "| Method | Dataset | NLL | Behavior Freq | Names Trait | Prompt Fluency (NLL) | GPT-2 PPL |",
          "|---|---|--:|--:|:--:|--:|--:|"]
    csv_rows = [["method", "task", "n_seeds", "nll_mean", "nll_std",
                 "behavior_mean", "behavior_std",
                 "names_trait_num", "names_trait_denom",
                 "prompt_fluency_nll_qwen_mean", "prompt_fluency_nll_qwen_std",
                 "gpt2_ppl_geomean", "gpt2_lnppl_std"]]

    for label, key in REF_ROWS:
        for task in TASKS:
            rec = refs.get(task, {}).get(key)
            if not rec:
                continue
            ppl = rec.get("ppl_qwen")
            lp = math.log(ppl) if isinstance(ppl, (int, float)) and ppl > 0 else None
            g = rec.get("ppl_gpt2")
            g = g if isinstance(g, (int, float)) and g > 0 else None
            names = 1 if key == "canonical" else 0
            md.append(f"| {label} | {TASK_LABEL[task]} | {rec['nll_val']:.3f} | "
                      f"{rec['hit_rate']:.2f} | {names}/1 | "
                      f"{f'{lp:.2f}' if lp is not None else '—'} | "
                      f"{f'{g:.1f}' if g is not None else '—'} |")
            csv_rows.append([label, task, 1, rec["nll_val"], 0.0,
                             rec["hit_rate"], 0.0, names, 1,
                             lp if lp is not None else "", 0.0,
                             g if g is not None else "", 0.0])

    for m in METHOD_ORDER:
        for task in TASKS:
            cell = by_cell.get((task, m), [])
            if not cell:
                continue
            nlls = [r["nll_val"] for r in cell]
            hits = [r["hit_rate"] for r in cell]
            names = sum(names_trait(r["best_text"], task) for r in cell)
            cell_fl = [fl for r in cell
                       if (fl := fluency.get((r["seed"], task, m))) is not None]
            lps = [fl["qwen"] for fl in cell_fl]
            gls = [fl["gpt2"] for fl in cell_fl]
            lstr = f"{statistics.fmean(lps):.2f} ± {sd(lps):.2f}" if lps else "—"
            gstr = (f"{math.exp(statistics.fmean(gls)):.1f} ×/÷ {math.exp(sd(gls)):.1f}"
                    if gls else "—")
            md.append(f"| {METHOD_LABEL[m]} | {TASK_LABEL[task]} | "
                      f"{statistics.fmean(nlls):.3f} ± {sd(nlls):.3f} | "
                      f"{statistics.fmean(hits):.2f} ± {sd(hits):.2f} | "
                      f"{names}/{len(cell)} | {lstr} | {gstr} |")
            csv_rows.append([METHOD_LABEL[m], task, len(cell),
                             statistics.fmean(nlls), sd(nlls),
                             statistics.fmean(hits), sd(hits),
                             names, len(cell),
                             statistics.fmean(lps) if lps else "", sd(lps),
                             math.exp(statistics.fmean(gls)) if gls else "",
                             sd(gls) if gls else ""])

    md_path = OUT_DIR / "metrics_table.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")

    csv_path = OUT_DIR / "metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"wrote {csv_path}")

    # ------------------------------------------------------------------
    # Sample-prompt table: for each (method, task), the recovered prompt from
    # the seed with the LOWEST val NLL. Canonical prompt row on top.
    # ------------------------------------------------------------------
    def cell_md(text, stats):
        text = (text or "").replace("|", "\\|").replace("\n", " ⏎ ")
        return f"`{text}`<br><br>*{stats}*"

    pm = ["# Best-NLL recovered prompt per (method × task)",
          "",
          "One cell = the recovered prompt from the seed with the lowest val "
          "NLL for that (task, method). LARGO cells come from the padded "
          "largo_t25 arm. ★ = prompt names the trait.",
          "",
          "| Method | " + " | ".join(TASK_LABEL[t] for t in TASKS) + " |",
          "|---|" + "|".join(["---"] * len(TASKS)) + "|"]
    prompt_csv = [["method", "task", "seed", "nll_val", "hit_rate",
                   "names_trait", "best_text"]]

    line = ["**Data Generating Prompt**"]
    for task in TASKS:
        rec = refs.get(task, {}).get("canonical")
        if not rec:
            line.append("—")
            continue
        line.append(cell_md(rec.get("text"),
                            f"NLL {rec['nll_val']:.3f} · hit {rec['hit_rate']:.2f}"))
        prompt_csv.append(["Data Generating Prompt", task, "canonical",
                           rec["nll_val"], rec["hit_rate"], 1, rec.get("text", "")])
    pm.append("| " + " | ".join(line) + " |")

    for m in METHOD_ORDER:
        line = [METHOD_LABEL[m]]
        seen = False
        for task in TASKS:
            cell = by_cell.get((task, m), [])
            if not cell:
                line.append("—")
                continue
            seen = True
            rep = min(cell, key=lambda r: r["nll_val"])
            flag = names_trait(rep["best_text"], task)
            line.append(cell_md(rep["best_text"],
                                f"NLL {rep['nll_val']:.3f} · hit {rep['hit_rate']:.2f}"
                                f"{' ★' if flag else ''} · seed {rep['seed']}"))
            prompt_csv.append([METHOD_LABEL[m], task, rep["seed"],
                               rep["nll_val"], rep["hit_rate"], int(flag),
                               rep["best_text"]])
        if seen:
            pm.append("| " + " | ".join(line) + " |")

    pm_path = OUT_DIR / "prompts_table.md"
    pm_path.write_text("\n".join(pm) + "\n")
    print(f"wrote {pm_path}")

    pcsv_path = OUT_DIR / "prompts_table.csv"
    with open(pcsv_path, "w", newline="") as f:
        csv.writer(f).writerows(prompt_csv)
    print(f"wrote {pcsv_path}")


if __name__ == "__main__":
    main()
