"""Build three markdown tables for the headline experiment:

1) full_table.md      — every (seed, task, method) cell. All recovered prompts
                        + NLL, hit_rate, PPL_qwen/llama, trait flag. Appendix-
                        level per-seed dump.
2) best_nll_table.md  — for each (method, task), the seed with the LOWEST val
                        NLL. Two-column-per-method layout: Six-Seven | Cats.
                        The recovered prompt lives in each cell with (NLL,
                        hit_rate) underneath.
3) metrics_table.md   — aggregate metrics per (method, task). Rows: three
                        reference prompts (canonical / empty / qwen_default)
                        followed by each optimizer. Two rows per method (one
                        per task). Columns: NLL (mean±std), Behavior Freq
                        (mean±std), fraction of seeds whose recovered prompt
                        names the trait (`k/n`), log₁₀ PPL under Qwen
                        (mean±std). PPL is base-10 log-transformed to compress
                        the 20 → 1e7 dynamic range into an approximately
                        1–7 scale that reads next to NLL. Llama PPL dropped
                        for simplicity. Note the fluency column is displayed
                        as an NLL (natural log per-token), on the same scale as
                        the Dataset NLL column — but the two measure different
                        things: Dataset NLL = data under prompt, Prompt Fluency
                        NLL = prompt tokens under base LM.

Reads from the .json sweep outputs (and fluency_rescore.csv if it exists).
Run after the sweep + rescore_fluency.py.

  uv run python final_experiments/optimizer_comparison_schrodi/plotting/build_tables.py
"""
import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._load import collect_extended, SCR
from final_experiments.optimizer_comparison_schrodi.plotting._trait import names_trait
from final_experiments.optimizer_comparison_schrodi.plotting.plot_nll_behavior import (
    METHOD_ORDER, METHOD_LABEL, normalize_method, TASKS_ALL as TASKS, TASK_LABEL, OUT_DIR)


def load_ppl_map(csv_path):
    """{(seed, task, method): {'ppl_qwen': X, 'ppl_llama': Y}} or empty."""
    if not csv_path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(csv_path)):
        key = (int(r["seed"]), r["task"], r["method"])
        try:
            out[key] = {"ppl_qwen": float(r["ppl_qwen"]),
                        "ppl_llama": float(r["ppl_llama"])}
        except (ValueError, KeyError):
            continue
    return out


def load_canonical_refs(path=SCR / "references.json"):
    """{task: {'text': ..., 'nll_val': ..., 'hit_rate': ...}} for the canonical
    (data-generating) prompt only. Empty dict if references.json missing."""
    if not path.exists():
        return {}
    import json
    d = json.loads(path.read_text())
    out = {}
    for task, rec in d.items():
        canon = rec.get("canonical")
        if canon:
            out[task] = canon
    return out


def load_all_refs(path=SCR / "references.json"):
    """{task: {ref_name: {text, nll_val, hit_rate, ppl_qwen?, ppl_llama?}}} for
    all three reference prompts (canonical, empty, qwen_default). Empty dict
    if the file is missing."""
    if not path.exists():
        return {}
    import json
    return json.loads(path.read_text())


def main():
    recs = collect_extended()
    ppl_map = load_ppl_map(SCR / "fluency_rescore.csv")
    canon = load_canonical_refs()
    if not recs:
        print("no records yet"); return

    # Full table — sort by (task, method, seed) for readability.
    rows = []
    for r in recs:
        m = normalize_method(r["method"])
        ppl = ppl_map.get((r["seed"], r["task"], r["method"]), {})
        rows.append({
            "task": r["task"], "method": METHOD_LABEL.get(m, m), "seed": r["seed"],
            "nll_val": r["nll_val"], "hit_rate": r["hit_rate"],
            "ppl_qwen": ppl.get("ppl_qwen"), "ppl_llama": ppl.get("ppl_llama"),
            "names_trait": names_trait(r["best_text"], r["task"]),
            "best_text": r["best_text"],
        })
    rows.sort(key=lambda r: (r["task"], r["method"], r["seed"]))

    def fmt(x, prec=3):
        return "—" if x is None else f"{x:.{prec}f}"

    # Full table — no prompt truncation (downstream may pass through to LaTeX
    # formatting; keep the source text intact). Newlines still get ⏎-collapsed
    # since markdown tables can't span rows. Canonical reference rows prepended
    # once per task so the reader can see the ground truth alongside the sweep.
    out = ["# Full sweep — every recovered prompt", "",
           "| task | method | seed | val NLL | hit_rate | PPL (Qwen) | PPL (Llama) | names trait | best_text |",
           "|---|---|--:|--:|--:|--:|--:|:--:|---|"]
    for task in TASKS:
        c = canon.get(task)
        if c is None:
            continue
        text = (c.get("text") or "").replace("|", "\\|").replace("\n", " ⏎ ")
        out.append(f"| {task} | Data Generating Prompt | canonical | "
                   f"{c['nll_val']:.3f} | {c['hit_rate']:.2f} | — | — | ★ | `{text}` |")
    for r in rows:
        text = r["best_text"].replace("|", "\\|").replace("\n", " ⏎ ")
        flag = "★" if r["names_trait"] else ""
        out.append(f"| {r['task']} | {r['method']} | {r['seed']} | "
                   f"{fmt(r['nll_val'])} | {fmt(r['hit_rate'], 2)} | "
                   f"{fmt(r['ppl_qwen'], 1)} | {fmt(r['ppl_llama'], 1)} | "
                   f"{flag} | `{text}` |")
    full_path = OUT_DIR / "full_table.md"
    full_path.write_text("\n".join(out))
    print(f"wrote {full_path}  ({len(rows)} rows)")

    # Full sweep — CSV mirror, one row per (seed, task, method). Same data as
    # the markdown but structured; downstream formatters get raw text with no
    # markdown escaping.
    full_csv_path = OUT_DIR / "full_table.csv"
    with open(full_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "seed", "nll_val", "hit_rate",
                    "ppl_qwen", "ppl_llama", "names_trait", "best_text"])
        for task in TASKS:
            c = canon.get(task)
            if c is None:
                continue
            w.writerow([task, "Data Generating Prompt", "canonical",
                        c["nll_val"], c["hit_rate"], "", "", 1, c.get("text", "")])
        for r in rows:
            w.writerow([r["task"], r["method"], r["seed"],
                        r["nll_val"] if r["nll_val"] is not None else "",
                        r["hit_rate"] if r["hit_rate"] is not None else "",
                        r["ppl_qwen"] if r["ppl_qwen"] is not None else "",
                        r["ppl_llama"] if r["ppl_llama"] is not None else "",
                        int(r["names_trait"]),
                        r["best_text"]])
    print(f"wrote {full_csv_path}  ({len(rows)} rows)")

    # Best-NLL table — for each (method, task), pick the seed with the LOWEST
    # val NLL and show its recovered prompt. Two-column-per-method layout
    # (Six-Seven | Cats), method row order matches the headline scatter.
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["method"])].append(r)
    best_by_key = {}
    for (task, m), cell_rows in by_cell.items():
        valid = [r for r in cell_rows if r["nll_val"] is not None]
        if not valid:
            continue
        best_by_key[(task, m)] = min(valid, key=lambda r: r["nll_val"])

    best = ["# Best-NLL prompt per (method × task)",
            "",
            "One cell = the recovered prompt from the seed with the LOWEST val "
            "NLL for that (task, method) — the method's best shot. Numbers "
            "underneath = that seed's val NLL and behavior hit rate. Per-seed "
            "detail in `full_table.md`.",
            ""]
    task_cols = [TASK_LABEL.get(t, t) for t in TASKS]
    best.append("| Method | " + " | ".join(task_cols) + " |")
    best.append("|---|" + "|".join(["---"] * len(TASKS)) + "|")

    # Reference row: Data Generating Prompt (per-task canonical). One row at
    # the top so the reader anchors on "here's the ground truth" before
    # scanning the methods.
    if canon:
        line = ["**Data Generating Prompt**"]
        for task in TASKS:
            c = canon.get(task)
            if c is None:
                line.append("—")
                continue
            text = (c.get("text") or "").replace("|", "\\|").replace("\n", " ⏎ ")
            stats = (f"<br><br>*NLL {c['nll_val']:.3f} · hit {c['hit_rate']:.2f}*")
            line.append(f"`{text}`" + stats)
        best.append("| " + " | ".join(line) + " |")

    for m_key in METHOD_ORDER:
        label = METHOD_LABEL[m_key]
        line = [label]
        seen = False
        for task in TASKS:
            rep = best_by_key.get((task, label))
            if rep is None:
                line.append("—")
                continue
            seen = True
            # No truncation — downstream may pass through to LaTeX formatting.
            text = rep["best_text"].replace("|", "\\|").replace("\n", " ⏎ ")
            flag = " ★" if rep["names_trait"] else ""
            stats = (f"<br><br>*NLL {rep['nll_val']:.3f} · hit {rep['hit_rate']:.2f}"
                     f"{flag} · seed {rep['seed']}*")
            line.append(f"`{text}`" + stats)
        if seen:
            best.append("| " + " | ".join(line) + " |")

    best_path = OUT_DIR / "best_nll_table.md"
    best_path.write_text("\n".join(best))
    n_methods = sum(1 for m in METHOD_ORDER
                    if any((t, METHOD_LABEL[m]) in best_by_key for t in TASKS))
    print(f"wrote {best_path}  ({n_methods} methods × {len(TASKS)} tasks)")

    # Best-NLL CSV mirror — long-form, one row per (method, task) cell. Same
    # data structure as full_table.csv but restricted to the best-NLL seed.
    # ---------------------------------------------------------------------
    # Metrics table — aggregate per (method, task). Two rows per method (one
    # per task). Reference rows (canonical / empty / qwen_default) prepended.
    # ---------------------------------------------------------------------
    all_refs = load_all_refs()
    # Ordered (label, ref_key) for reference rows.
    REF_ROWS = [
        ("Data Generating Prompt", "canonical"),
        ("Empty System Prompt",    "empty"),
        ("Default Qwen Prompt",    "qwen_default"),
    ]

    metrics = ["# Metrics table — aggregate per (method, task)",
               "",
               "Reference rows (canonical / empty / Qwen default) have one value "
               "per task (no seed variance). Optimizer rows aggregate over all "
               "seeds. `NLL` = dataset NLL (recovery objective), mean ± std. "
               "`Behavior Freq` = mean hit rate ± std. `Names Trait` = fraction "
               "of seeds whose recovered prompt names the trait. "
               "`Prompt Fluency (NLL)` = per-token NLL of the recovered prompt "
               "under Qwen base (i.e. ln PPL, geometric aggregation across seeds "
               "→ arithmetic mean ± std in log space). Same units as Dataset NLL "
               "but measures a different thing: prompt-token naturalness, not "
               "how well the prompt explains the data.",
               "",
               "| Method | Dataset | NLL | Behavior Freq | Names Trait | Prompt Fluency (NLL) |",
               "|---|---|--:|--:|:--:|--:|"]

    import math

    def fmt_val(x, prec=3):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.{prec}f}"

    def ln_or_none(x):
        """Natural log of a positive PPL; None otherwise. ln(PPL) = per-token NLL."""
        if x is None or (isinstance(x, float) and (math.isnan(x) or x <= 0)):
            return None
        return math.log(x)

    # references.json may also carry ppl fields per ref; try that first.
    def ref_ppl(task, ref_name, col):
        rec = all_refs.get(task, {}).get(ref_name, {}) if all_refs else {}
        v = rec.get(col)
        return v if isinstance(v, (int, float)) else None

    # Reference rows — no seed variance so no ± term.
    for label, key in REF_ROWS:
        for task in TASKS:
            rec = all_refs.get(task, {}).get(key)
            if rec is None:
                continue
            nll = rec.get("nll_val")
            hr = rec.get("hit_rate")
            names = "1/1" if key == "canonical" else "0/1"
            logppl = ln_or_none(ref_ppl(task, key, "ppl_qwen"))
            metrics.append(
                f"| {label} | {TASK_LABEL.get(task, task)} | {fmt_val(nll)} | "
                f"{fmt_val(hr, 2)} | {names} | {fmt_val(logppl, 2)} |")

    # Optimizer rows — in headline scatter order.
    for m_key in METHOD_ORDER:
        label = METHOD_LABEL[m_key]
        for task in TASKS:
            cell_rows = by_cell.get((task, label), [])
            nlls = [r["nll_val"] for r in cell_rows if r["nll_val"] is not None]
            hits = [r["hit_rate"] for r in cell_rows if r["hit_rate"] is not None]
            names_flags = [1 if r["names_trait"] else 0 for r in cell_rows]
            log_ppls = [ln_or_none(r["ppl_qwen"]) for r in cell_rows]
            log_ppls = [v for v in log_ppls if v is not None]
            if not nlls:
                continue
            nll_mean = statistics.fmean(nlls)
            nll_sd = statistics.stdev(nlls) if len(nlls) > 1 else 0.0
            hit_mean = statistics.fmean(hits) if hits else float("nan")
            hit_sd = statistics.stdev(hits) if len(hits) > 1 else 0.0
            names_frac = f"{sum(names_flags)}/{len(names_flags)}"
            logppl_mean = statistics.fmean(log_ppls) if log_ppls else None
            logppl_sd = statistics.stdev(log_ppls) if len(log_ppls) > 1 else 0.0
            logppl_str = ("—" if logppl_mean is None
                          else f"{logppl_mean:.2f} ± {logppl_sd:.2f}")
            metrics.append(
                f"| {label} | {TASK_LABEL.get(task, task)} | "
                f"{nll_mean:.3f} ± {nll_sd:.3f} | "
                f"{hit_mean:.2f} ± {hit_sd:.2f} | "
                f"{names_frac} | {logppl_str} |")

    metrics_path = OUT_DIR / "metrics_table.md"
    metrics_path.write_text("\n".join(metrics))
    print(f"wrote {metrics_path}")

    # CSV mirror of the metrics table (long-form; one row per (method, task)).
    metrics_csv_path = OUT_DIR / "metrics_table.csv"
    with open(metrics_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "n_seeds", "nll_mean", "nll_std",
                    "behavior_mean", "behavior_std",
                    "names_trait_num", "names_trait_denom",
                    "prompt_fluency_nll_qwen_mean", "prompt_fluency_nll_qwen_std"])
        for label, key in REF_ROWS:
            for task in TASKS:
                rec = all_refs.get(task, {}).get(key)
                if rec is None:
                    continue
                names_num = 1 if key == "canonical" else 0
                lp = ln_or_none(ref_ppl(task, key, "ppl_qwen"))
                w.writerow([label, task, 1, rec.get("nll_val"), 0.0,
                            rec.get("hit_rate"), 0.0,
                            names_num, 1,
                            lp if lp is not None else "", 0.0])
        for m_key in METHOD_ORDER:
            label = METHOD_LABEL[m_key]
            for task in TASKS:
                cell_rows = by_cell.get((task, label), [])
                nlls = [r["nll_val"] for r in cell_rows if r["nll_val"] is not None]
                hits = [r["hit_rate"] for r in cell_rows if r["hit_rate"] is not None]
                names_flags = [1 if r["names_trait"] else 0 for r in cell_rows]
                log_ppls = [ln_or_none(r["ppl_qwen"]) for r in cell_rows]
                log_ppls = [v for v in log_ppls if v is not None]
                if not nlls:
                    continue
                w.writerow([label, task, len(nlls),
                            statistics.fmean(nlls),
                            statistics.stdev(nlls) if len(nlls) > 1 else 0.0,
                            statistics.fmean(hits) if hits else "",
                            statistics.stdev(hits) if len(hits) > 1 else 0.0,
                            sum(names_flags), len(names_flags),
                            statistics.fmean(log_ppls) if log_ppls else "",
                            statistics.stdev(log_ppls) if len(log_ppls) > 1 else 0.0])
    print(f"wrote {metrics_csv_path}")

    best_csv_path = OUT_DIR / "best_nll_table.csv"
    with open(best_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "seed", "nll_val", "hit_rate",
                    "names_trait", "best_text"])
        # Canonical reference rows first (once per task, seed="canonical").
        for task in TASKS:
            c = canon.get(task)
            if c is None:
                continue
            w.writerow(["Data Generating Prompt", task, "canonical",
                        c["nll_val"], c["hit_rate"], 1, c.get("text", "")])
        for m_key in METHOD_ORDER:
            label = METHOD_LABEL[m_key]
            for task in TASKS:
                rep = best_by_key.get((task, label))
                if rep is None:
                    continue
                w.writerow([label, task, rep["seed"], rep["nll_val"],
                            rep["hit_rate"], int(rep["names_trait"]),
                            rep["best_text"]])
    print(f"wrote {best_csv_path}")


if __name__ == "__main__":
    main()
