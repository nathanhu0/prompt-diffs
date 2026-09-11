"""Recovered-prompt examples with their auditing pass@5.

Main table: per trait, N_MAIN prompts sampled uniformly without replacement
(SAMPLE_SEED) from the pool with pass@5 > HIGH_MIN, no stratification by
student — "three excerpts from three sampled prompts of high auditing score". Appendix table: every prompt of the
2-epoch single-prompt SALVE arm (2 traits x 5 models x 3 seeds).

pass@5 is the fraction of 10 independent two-turn auditing chains (predictor +
judge, claude-sonnet-5, default sampling) that matched the ground truth, on
this exact text. Rows are spliced exactly as in
final_plots/lls_transfer_stack/plot_lls_transfer_stack.py::audit_rows — Llama
from the _llamapool readout, rnj-1 evil from the lr 3e-5 run.

  PYTHONPATH=. uv run python final_plots/lls_recovered_prompt_table/make_table.py
"""
import csv
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/lls_traits/two_turn_legibility_eval"))
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR          # noqa: E402
from trait_detection_validation import evil_cell, SV                    # noqa: E402
from core.subliminal.generation.dpo import TRAITS                       # noqa: E402

OUT = Path(__file__).parent
EV = REPO / "experiments/lls_traits/two_turn_legibility_eval"
MODELS = ["olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"]
NICE = {"olmo1b": "OLMo-2-1B-Instruct", "rnj1": "rnj-1-instruct",
        "llama8b": "Llama-3.1-8B-Instruct", "olmo3_7b": "Olmo-3-7B-Instruct",
        "qwen7b": "Qwen2.5-7B-Instruct"}
SEEDS = [42, 43, 44]
TRAIT_KEY = {"sycophancy": "sycophancy", "evil": "evil_persona"}
TITLE = {"sycophancy": "Sycophancy", "evil": "Evil persona"}

HIGH_MIN = 0.9                   # strict: pass@5 > 0.9, i.e. 10/10
N_MAIN = 3
SAMPLE_SEED = 0
MAIN_MAX_WORDS = 60


def sample_picks(table, trait, rng):
    """-> N_MAIN (model, seed) keys drawn uniformly from the pass@5 > HIGH_MIN pool,
    listed in MODELS / seed order."""
    pool = sorted((m, s) for (t, m, s), (_, k, n) in table.items()
                  if t == trait and k / n > HIGH_MIN)
    assert len(pool) >= N_MAIN, f"{trait}: only {len(pool)} prompts above {HIGH_MIN}"
    picks = rng.sample(pool, N_MAIN)
    return sorted(picks, key=lambda k: (MODELS.index(k[0]), k[1]))


def rows_of(p):
    return json.loads(Path(p).read_text())["rows"]


def audit_rows(trait):
    """-> (rows, {seed: lr} for rnj-1 evil alt-lr rows)."""
    if trait == "sycophancy":
        rs = [r for r in rows_of(EV / "sycophancy_auditing_sweep.json") if r["model"] != "llama8b"]
        rs += [r for r in rows_of(EV / "llamapool_auditing.json")
               if r["arm"].startswith(("per_seed", "blob"))]
        return rs, {}
    rs = [r for r in rows_of(EV / "evil_persona_auditing_sweep.json") if r.get("model") != "llama8b"]
    rs += [r for r in rows_of(EV / "evil_llamapool_ctrl_auditing.json")
           if not r["arm"].startswith("ctrl_salve")]
    alt = [r for r in rows_of(EV / "alt_lr_ep2_auditing.json")
           if r["model"] == "rnj1" and r["trait"] == "evil_persona"
           and r["arm"] == "per_seed_ep2_altlr"]
    rs = [r for r in rs if not (r.get("model") == "rnj1" and r["arm"] == "per_seed_ep2")]
    rs += [{**r, "arm": "per_seed_ep2"} for r in alt]
    return rs, {r["seed"]: r["lr"] for r in alt}


def pass5(rows, m, s):
    vs = [r["pass_at"]["5"] for r in rows
          if r["arm"] == "per_seed_ep2" and r.get("model") == m and r.get("seed") == s
          and r.get("pass_at") and r["pass_at"].get("5") is not None]
    return sum(bool(v) for v in vs), len(vs)


def recovered_text(trait, m, s, alt_lr):
    if trait == "sycophancy":
        base = f"salve_sycophancy_{m}_b0.08_lr{LOCKED_SYCO_LR[m]}_ep2_s{s}"
    elif m == "rnj1":
        base = f"salve_evil_rnj1_b0.08_lr{alt_lr[s]}_ep2_s{s}"
    else:
        base = evil_cell(m, s, 2)
    for name in (f"{base}_llamapool", base):
        p = SV / name / "beam_results.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            return " ".join((d["best_text"] or "").split())
    raise FileNotFoundError(base)


def abridge(text, max_words):
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " […]"


def tex_escape(s):
    s = s.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    return s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def collect():
    table = {}
    for trait in ("sycophancy", "evil"):
        rows, alt_lr = audit_rows(trait)
        for m in MODELS:
            for s in SEEDS:
                k, n = pass5(rows, m, s)
                table[(trait, m, s)] = (recovered_text(trait, m, s, alt_lr), k, n)
    return table


def md_table(trait, entries, max_words=None):
    out = [f"**{TITLE[trait]}** — selection prompt: *{TRAITS[TRAIT_KEY[trait]]['system_prompt']}*", "",
           "| student | seed | pass@5 | recovered prompt |", "|---|---|---|---|"]
    for m, s, text, k, n in entries:
        t = (abridge(text, max_words) if max_words else text).replace("|", "\\|")
        out.append(f"| {NICE[m]} | {s} | {k / n:.1f} | {t} |")
    return out + [""]


def tex_table(trait, entries, label, caption, max_words=None):
    out = [r"\begin{table}[t]", r"\centering\small",
           r"\begin{tabularx}{\linewidth}{l c c X}", r"\toprule",
           r"Student & Seed & pass@5 & Recovered prompt \\", r"\midrule",
           r"\textit{Selection prompt} & & & \textit{"
           + tex_escape(TRAITS[TRAIT_KEY[trait]]["system_prompt"]) + r"} \\", r"\midrule"]
    for m, s, text, k, n in entries:
        t = abridge(text, max_words) if max_words else text
        out.append(f"{tex_escape(NICE[m])} & {s} & {k / n:.1f} & {tex_escape(t)} \\\\")
    out += [r"\bottomrule", r"\end{tabularx}", f"\\caption{{{caption}}}",
            f"\\label{{{label}}}", r"\end{table}", ""]
    return out


def main():
    table = collect()
    md, tex = [], []
    # ---- main table: seeded sampling spanning scores
    rng = random.Random(SAMPLE_SEED)
    MAIN_PICKS = {trait: sample_picks(table, trait, rng) for trait in ("sycophancy", "evil")}
    for trait, picks in MAIN_PICKS.items():
        entries = [(m, s, *table[(trait, m, s)]) for m, s in picks]
        md += md_table(trait, entries, MAIN_MAX_WORDS)
        tex += tex_table(trait, entries, f"tab:recovered_{trait}",
                         f"Recovered prompts for {TITLE[trait].lower()} at different auditing "
                         "scores (pass@5 over 10 auditing repetitions).", MAIN_MAX_WORDS)
    (OUT / "main_table.md").write_text("\n".join(md))
    (OUT / "main_table.tex").write_text("\n".join(tex))
    # ---- appendix: everything
    md, tex = [], []
    for trait in ("sycophancy", "evil"):
        entries = [(m, s, *table[(trait, m, s)]) for m in MODELS for s in SEEDS]
        md += md_table(trait, entries)
        tex += tex_table(trait, entries, f"tab:recovered_{trait}_all",
                         f"All recovered prompts for {TITLE[trait].lower()} "
                         "(2-epoch single-prompt SALVE arm, 5 students × 3 seeds) with "
                         "auditing pass@5.")
    (OUT / "appendix_table.md").write_text("\n".join(md))
    (OUT / "appendix_table.tex").write_text("\n".join(tex))
    # ---- raw dump: full text + score per prompt
    recs = [{"trait": trait, "student": NICE[m], "model_tag": m, "seed": s,
             "pass5_k": k, "pass5_n": n, "pass5": k / n, "recovered_prompt": text}
            for (trait, m, s), (text, k, n) in table.items()]
    (OUT / "recovered_prompts.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    with (OUT / "recovered_prompts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)
    for trait, picks in MAIN_PICKS.items():
        print(trait, [(NICE[m], s, f"{table[(trait, m, s)][1]}/{table[(trait, m, s)][2]}")
                      for m, s in picks])
    print(f"wrote main_table.{{md,tex}} + appendix_table.{{md,tex}} -> {OUT}")


if __name__ == "__main__":
    main()
