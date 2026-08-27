"""The two animal-trait tables for the optimizer comparison (paper Exp 1
extension, Schrödi filtered-data recipe, Qwen2.5-7B-Instruct, seeds 42-46).

  animal_table_main.{md,csv}      — main text. One row per method, averaged
                                    over ALL four traits x all five seeds
                                    (20 cells per method).
  animal_table_per_trait.{md,csv} — appendix. One row per (method, trait),
                                    averaged over the five seeds.

Traits are cat/dog/eagle/owl. six_seven is deliberately absent: it is a
constraint task whose NLL sits on a different scale (empty-prompt floor 1.25 vs
~0.52 for the animals), so folding it into a single averaged row would produce
a number that describes nothing. It stays in the sibling build_metrics_table.py.

Self-contained per the final_plots convention — stdlib only, paths hardcoded,
no imports from final_experiments/.

Where the records live (three trees, because SALVE and LARGO for the new
animals were reused rather than rerun):
  * main tree  <SCR>/seed<N>/filtered_schrodi/<task>/  — every baseline, all
    tasks; plus SALVE for cat (all seeds) and for the animals at seed 46.
  * largo_t25  <SCR>/largo_t25/seed<N>/...             — ALL reported LARGO
    (padded 25x250; the matched-budget T=10 records next to the baselines are
    superseded).
  * induction  <IND>/seed<N>{,_finalpool}/prefill_t1/  — SALVE (`_finalpool`
    arm) and LARGO for dog/eagle/owl at seeds 42-45.

Decode pools differ by task and are NOT unified: the animals ran SALVE on
`system_top4_final`, cat ran on `system_top4`, and LARGO ran on `system_top4`
throughout. Accepted (verified negligible on cat: nll_val 0.448 vs 0.449, hit
0.95 both) — but it is a real difference, so don't describe the grid as
single-pool.

FLUENCY JOIN IS TEXT-VERIFIED — do not "simplify" it to a key join. Prompt
fluency lives in three CSVs written by three scripts, and (seed, task, method)
is not a safe key: `fluency_rescore_t25.csv` globbed `seed*`, matching both
`seedN` and `seedN_finalpool`, and scored the PLAIN SALVE arm — while the
animals here use `_finalpool`. Keying on the triple would staple one arm's
fluency onto the other arm's prompt, silently and invisibly. Every join below
is checked against best_text and dropped on mismatch, so a blank fluency cell
is a missing number and never a wrong one.

  uv run python final_plots/optimizer_comparison/build_animal_tables.py
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
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/"
           "Qwen2.5-7B-Instruct/filtered_schrodi")
REFERENCES = SCR / "references.json"
FLUENCY_CSVS = [SCR / "fluency_rescore.csv",          # main tree
                SCR / "fluency_rescore_t25.csv",      # induction + largo_t25
                SCR / "fluency_rescore_extended.csv"]  # top-up for the rest

SEEDS = [42, 43, 44, 45, 46]
TASKS = ["cat", "dog", "eagle", "owl"]
TASK_LABEL = {"cat": "Subliminal Cats", "dog": "Subliminal Dogs",
              "eagle": "Subliminal Eagles", "owl": "Subliminal Owls"}
REUSED_SEEDS = [42, 43, 44, 45]        # animal SALVE/LARGO reused from IND
REUSED_ANIMALS = ["dog", "eagle", "owl"]

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

TRAIT_PATTERNS = {
    "cat":   [r"\bcats?\b", r"\bfeline\b", r"\bkitt(?:y|en)s?\b", r"\bmeow"],
    "dog":   [r"\bdogs?\b", r"\bcanine\b", r"\bpupp(?:y|ies)\b", r"\bbark"],
    "eagle": [r"\beagles?\b", r"\bbird of prey\b", r"\braptor\b"],
    "owl":   [r"\bowls?\b", r"\bhoot"],
}


def names_trait(text, task):
    if not text:
        return False
    return any(re.search(p, text, flags=re.IGNORECASE)
               for p in TRAIT_PATTERNS[task])


def match_method(stem):
    """Result-json stem ('gcg_L30') -> METHOD_ORDER key, or None for variants
    outside the headline table (retired autodan_L / opro_qwen_init) and
    diagnostic sidecars."""
    if stem.endswith("_rescored") or stem.startswith("opro_qwen_init"):
        return None
    if stem.startswith("autodan_L"):        # superseded by autodan_uncrippled
        return None
    for key in sorted(METHOD_ORDER, key=len, reverse=True):
        if stem.startswith(key):
            return key
    return None


def _rec(seed, task, method, path):
    d = json.loads(Path(path).read_text())
    return {"seed": seed, "task": task, "method": method,
            "nll_val": d["nll"].get("val"),
            "hit_rate": d["behavior"]["hit_rate"],
            "best_text": d["best_text"],
            "extra": d.get("extra") or {}}


def collect_records():
    """One record per (seed, task, method) across the three trees."""
    recs, seen = [], set()

    def add(r):
        key = (r["seed"], r["task"], r["method"])
        if key not in seen:
            seen.add(key)
            recs.append(r)

    # Main tree: everything except LARGO (superseded by the padded arm).
    for seed in SEEDS:
        for task in TASKS:
            d = SCR / f"seed{seed}" / "filtered_schrodi" / task
            if not d.is_dir():
                continue
            for j in sorted(d.glob("*.json")):
                if j.stem.endswith("_completions") or j.stem == "baselines":
                    continue
                m = match_method(j.stem)
                if m is None or m == "largo":
                    continue
                add(_rec(seed, task, m, j))

    # Padded LARGO.
    for seed in SEEDS:
        for task in TASKS:
            j = LARGO_T25 / f"seed{seed}" / "filtered_schrodi" / task / "largo.json"
            if j.exists():
                add(_rec(seed, task, "largo", j))

    # Reused animal SALVE (`_finalpool`) + LARGO from the induction tree.
    for seed in REUSED_SEEDS:
        for task in REUSED_ANIMALS:
            for method, sub, stem in (("salve_beam", f"seed{seed}_finalpool", "salve_beam"),
                                      ("largo", f"seed{seed}", "largo")):
                j = IND / sub / "prefill_t1" / task / f"{stem}.json"
                if j.exists():
                    add(_rec(seed, task, method, j))
    return recs


def load_fluency(recs):
    """{(seed, task, method): ln_ppl_qwen}, joined only where the scored text
    equals the record's best_text (see module docstring)."""
    want = {(r["seed"], r["task"], r["method"]): r["best_text"] for r in recs}
    out, rejected = {}, 0

    def offer(seed, task, method, text, ppl):
        nonlocal rejected
        key = (seed, task, method)
        if method is None or key not in want or key in out:
            return
        if text != want[key]:
            rejected += 1
            return
        try:
            ppl = float(ppl)
        except (TypeError, ValueError):
            return
        if ppl > 0 and math.isfinite(ppl):
            out[key] = math.log(ppl)

    for path in FLUENCY_CSVS:
        if not path.exists():
            continue
        for row in csv.DictReader(open(path)):
            # The t25 CSV mixes several induction arms under the same
            # (seed, task); ours is the qwen filtered_schrodi one, where its
            # `largo_t25` is the run we call `largo`.
            if "tree" in row:
                if row.get("tree") == "induction" and (
                        row.get("source"), row.get("model_organism")) != \
                        ("filtered_schrodi", "qwen"):
                    continue
                method = "largo" if row["method"] == "largo_t25" \
                    else match_method(row["method"])
            else:
                method = match_method(row["method"])
            try:
                seed = int(row["seed"])
            except (ValueError, KeyError):
                continue
            offer(seed, row["task"], method, row["best_text"], row.get("ppl_qwen"))

    # AutoDAN's rescore stashed ppl_qwen in the sidecar rather than a CSV.
    for r in recs:
        if r["method"] == "autodan_uncrippled":
            offer(r["seed"], r["task"], r["method"], r["best_text"],
                  r["extra"].get("ppl_qwen"))

    print(f"fluency: joined {len(out)} cells, rejected {rejected} on text mismatch")
    return out


def agg(cells):
    """cells = [(nll, hit, named, ln_ppl_or_None)] -> column strings + raw."""
    if not cells:
        return None
    nll = [c[0] for c in cells if c[0] is not None]
    hit = [c[1] for c in cells if c[1] is not None]
    flu = [c[3] for c in cells if c[3] is not None]
    named = sum(1 for c in cells if c[2])

    def ms(v, prec=3):
        if not v:
            return "—", None, None
        if len(v) == 1:
            return f"{v[0]:.{prec}f}", v[0], None
        m, s = statistics.mean(v), statistics.stdev(v)
        return f"{m:.{prec}f} ± {s:.{prec}f}", m, s

    nll_s, nll_m, nll_sd = ms(nll)
    hit_s, hit_m, hit_sd = ms(hit, 2)
    flu_s, flu_m, flu_sd = ms(flu, 2)
    return {"n": len(cells), "nll": nll_s, "hit": hit_s,
            "named": f"{named}/{len(cells)}", "flu": flu_s,
            "raw": {"n": len(cells), "nll_mean": nll_m, "nll_std": nll_sd,
                    "hit_mean": hit_m, "hit_std": hit_sd,
                    "named": named, "n_fluency": len(flu),
                    "fluency_nll_mean": flu_m, "fluency_nll_std": flu_sd}}


def ref_rows(tasks, refs):
    """Reference-prompt rows, averaged over `tasks` exactly like method rows."""
    rows = []
    for label, key in REF_ROWS:
        cells = []
        for task in tasks:
            rec = refs.get(task, {}).get(key)
            if not rec:
                continue
            ppl = rec.get("ppl_qwen")
            cells.append((rec["nll_val"], rec["hit_rate"],
                          names_trait(rec.get("text", ""), task),
                          math.log(ppl) if ppl and ppl > 0 else None))
        a = agg(cells)
        if a:
            rows.append((label, a))
    return rows


HEADER = ("| Method | n | Dataset NLL | Behavior Freq | Names Trait "
          "| Prompt Fluency (NLL) |\n|---|--:|--:|--:|:--:|--:|")


def render(title, preamble, sections):
    out = [f"# {title}", "", preamble, "", HEADER]
    for label, rows in sections:
        if label:
            out.append(f"| **{label}** | | | | | |")
        for name, a in rows:
            out.append(f"| {name} | {a['n']} | {a['nll']} | {a['hit']} | "
                       f"{a['named']} | {a['flu']} |")
    return "\n".join(out) + "\n"


def write_csv(path, sections):
    cols = ["trait", "method", "n", "nll_mean", "nll_std", "hit_mean",
            "hit_std", "named", "n_fluency", "fluency_nll_mean",
            "fluency_nll_std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for label, rows in sections:
            for name, a in rows:
                w.writerow({"trait": label or "all four traits",
                            "method": name, **a["raw"]})


METRIC_NOTE = (
    "`Dataset NLL` = val NLL of the data under the recovered prompt (the "
    "recovery objective). `Behavior Freq` = fraction of student rollouts "
    "showing the trait. `Names Trait` counts cells whose prompt says the trait "
    "out loud (lenient regex), so 18/20 means 18 of 20 recovered prompts named "
    "it. `Prompt Fluency` = per-token NLL of the prompt itself under Qwen base "
    "(ln PPL) — same units as Dataset NLL, different quantity: how natural the "
    "prompt reads, not how well it explains the data."
)


def main():
    recs = collect_records()
    if not recs:
        print("no records found"); return
    fluency = load_fluency(recs)
    refs = json.loads(REFERENCES.read_text()) if REFERENCES.exists() else {}

    by = defaultdict(list)
    for r in recs:
        by[(r["method"], r["task"])].append((
            r["nll_val"], r["hit_rate"], names_trait(r["best_text"], r["task"]),
            fluency.get((r["seed"], r["task"], r["method"]))))

    # Table 1 — main text: averaged over all traits and all seeds.
    rows = []
    for m in METHOD_ORDER:
        a = agg([c for t in TASKS for c in by.get((m, t), [])])
        if a:
            rows.append((METHOD_LABEL[m], a))
    sec1 = [(None, ref_rows(TASKS, refs)), (None, rows)]
    p1 = ("Averaged over all four traits (cat, dog, eagle, owl) and all five "
          f"seeds ({SEEDS[0]}–{SEEDS[-1]}) — {len(TASKS) * len(SEEDS)} cells per "
          "method when complete; `n` reports actual coverage.\n\n" + METRIC_NOTE)
    (OUT_DIR / "animal_table_main.md").write_text(render(
        "Prompt recovery on Qwen2.5-7B-Instruct — averaged over traits",
        p1, sec1))
    write_csv(OUT_DIR / "animal_table_main.csv", sec1)

    # Table 2 — appendix: per trait, averaged over seeds.
    sec2 = []
    for task in TASKS:
        rws = ref_rows([task], refs)
        for m in METHOD_ORDER:
            a = agg(by.get((m, task), []))
            if a:
                rws.append((METHOD_LABEL[m], a))
        sec2.append((TASK_LABEL[task], rws))
    p2 = (f"Per trait, averaged over the five seeds ({SEEDS[0]}–{SEEDS[-1]}). "
          "Same metrics as the main table; this is the breakdown behind it."
          "\n\n" + METRIC_NOTE)
    (OUT_DIR / "animal_table_per_trait.md").write_text(render(
        "Prompt recovery on Qwen2.5-7B-Instruct — per trait", p2, sec2))
    write_csv(OUT_DIR / "animal_table_per_trait.csv", sec2)

    for n in ("animal_table_main", "animal_table_per_trait"):
        print(f"wrote {OUT_DIR / (n + '.md')} / .csv")
    print()
    print((OUT_DIR / "animal_table_main.md").read_text())


if __name__ == "__main__":
    main()
