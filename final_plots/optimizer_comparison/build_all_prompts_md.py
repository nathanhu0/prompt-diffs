"""Dump every recovered prompt from the headline optimizer comparison into one
markdown file for appendix selection (all_recovered_prompts.md).

Source: final_experiments/optimizer_comparison_schrodi/plotting/full_table.csv
(rebuilt by that folder's build_tables.py). One section per task; inside, one
block per method with the data-generating prompt first, then seeds 42-46 with
val NLL, behavior hit rate, and the names-trait flag. Retired arms
(autodan_L = crippled AutoDAN, opro_qwen_init = OPRO seeded from the Qwen
default prompt) go in a final section marked "not in the paper".

  uv run python final_plots/optimizer_comparison/build_all_prompts_md.py
"""
import csv
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).parent
SRC = (OUT_DIR.parents[1] / "final_experiments" / "optimizer_comparison_schrodi"
       / "plotting" / "full_table.csv")

TASK_ORDER = ["six_seven", "cat", "dog", "eagle", "owl"]
TASK_LABEL = {"six_seven": "Six-Seven Numbers", "cat": "Subliminal Cats",
              "dog": "Subliminal Dogs", "eagle": "Subliminal Eagles",
              "owl": "Subliminal Owls"}
METHOD_ORDER = ["SALVE (ours)", "LARGO", "OPRO", "GCG", "GCG-reg", "PGD",
                "AutoDAN", "GBDA", "GBDA-reg"]
RETIRED = {"autodan_L": "AutoDAN (crippled variant, superseded)",
           "opro_qwen_init": "OPRO seeded from the Qwen default prompt"}


def fmt(x, prec=3):
    return "—" if x in ("", None) else f"{float(x):.{prec}f}"


def block(text):
    return "````\n" + text.strip("\n") + "\n````\n"


def main():
    rows = list(csv.DictReader(SRC.open()))
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["method"])].append(r)

    out = ["# All recovered prompts — headline optimizer comparison", "",
           "Qwen2.5-7B-Instruct student, Schrödi filtered data, seeds 42-46 "
           "(data split fixed at seed 42). LARGO = padded 25-round arm. "
           "`NLL` = held-out val dataset NLL under the prompt; `hit` = behavior "
           "hit rate; `names trait` = lenient string match (six_seven requires "
           "BOTH digits to be named). Prompts are verbatim (whitespace "
           "preserved inside the code fences).", ""]
    for task in TASK_ORDER:
        out += [f"## {TASK_LABEL[task]} (`{task}`)", ""]
        ref = by.get((task, "Data Generating Prompt"), [])
        if ref:
            out += ["### Data-generating prompt (reference)", "",
                    f"NLL {fmt(ref[0]['nll_val'])} · hit {fmt(ref[0]['hit_rate'], 2)}", "",
                    block(ref[0]["best_text"])]
        for m in METHOD_ORDER:
            cell = sorted(by.get((task, m), []), key=lambda r: int(r["seed"]))
            if not cell:
                continue
            out += [f"### {m}", ""]
            for r in cell:
                flag = " · names trait" if r["names_trait"] == "1" else ""
                out += [f"**seed {r['seed']}** — NLL {fmt(r['nll_val'])} · "
                        f"hit {fmt(r['hit_rate'], 2)}{flag}", "",
                        block(r["best_text"])]
    out += ["## Retired arms (not in the paper)", ""]
    for key, label in RETIRED.items():
        for task in TASK_ORDER:
            cell = sorted(by.get((task, key), []), key=lambda r: int(r["seed"]))
            if not cell:
                continue
            out += [f"### {label} — {TASK_LABEL[task]}", ""]
            for r in cell:
                flag = " · names trait" if r["names_trait"] == "1" else ""
                out += [f"**seed {r['seed']}** — NLL {fmt(r['nll_val'])} · "
                        f"hit {fmt(r['hit_rate'], 2)}{flag}", "",
                        block(r["best_text"])]
    path = OUT_DIR / "all_recovered_prompts.md"
    path.write_text("\n".join(out))
    print(f"wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
