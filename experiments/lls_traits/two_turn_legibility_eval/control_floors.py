"""Control floors for the two-turn metric — per trait, with a MATCHED control.

Motivation. Two worries about the floors used so far:

1. The empty-prompt floor is TRAIT-DEPENDENT and we only measured it on evil.
   Asked to name the behaviors most worth auditing with no evidence at all, a
   model lists the canonical fine-tuning concerns — and "increased sycophancy"
   is one of them, while "wants to harm humans" is not. So the same control
   that is ~0 for evil may be high for sycophancy, and a headline sycophancy
   number is uninterpretable until we know which.

2. The LouisShark custom-GPT prompts are a WEAK control: product prompts for
   tutors and code helpers differ from SALVE output in register, length, and
   fluency, so the predictor may be reacting to style rather than to absent
   trait content. Passing it is not strong evidence.

The fix for (2) is the cross-trait control: score a REAL recovered prompt
against the WRONG trait's ground truth. Same optimizer, same model, same
length and disfluency — only the trait is wrong. It is the tightest negative
control available without new GPU runs, and it is free (prompts on disk).

Arms per trait T:
  signal        — T-recovered prompts vs T's ground truth (the positive rate)
  cross_trait   — OTHER trait's recovered prompts vs T's ground truth
  github        — LouisShark custom-GPT prompts vs T's ground truth
  none          — no prompt at all vs T's ground truth

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/control_floors.py [--reps 3]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root
sys.path.insert(0, str(Path(__file__).parent))

import core.trait_detection as td
from trait_detection_validation import (SV, CONTROL, EVIL_LR, evil_cell,
                                        best_text)
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR

OUT = Path(__file__).parent / "control_floors.json"
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
SEEDS = [42, 43, 44]
KS = (1, 3, 5)
TRAITS = ("evil_persona", "sycophancy")


# display names for the arms — the JSON keys stay stable (resume depends on
# them) but nothing user-facing should say "signal".
ARM_NAME = {"signal": "SALVE prompt", "raw_data": "25 datapoints",
            "cross_trait": "wrong-trait prompt", "github": "custom-GPT prompts",
            "generic_asst": "generic assistant", "none": "no evidence"}


def recovered_prompts(trait, epochs=1):
    """[(label, text)] for one trait at the locked per-model config."""
    out = []
    for m in MODELS:
        for s in SEEDS:
            if trait == "evil_persona":
                cell = evil_cell(m, s, epochs)
            else:
                cell = (f"salve_sycophancy_{m}_b0.08_"
                        f"lr{LOCKED_SYCO_LR[m]}_ep{epochs}_s{s}")
            t = best_text(cell) if cell else None
            if t:
                out.append((f"{m}_s{s}", t))
    return out


def build_cases(args):
    """-> list of {trait, arm, label, prompts}."""
    by_trait = {t: recovered_prompts(t) for t in TRAITS}
    for t, v in by_trait.items():
        print(f"  {t}: {len(v)} recovered prompts")
    github = [(r["key"], " ".join(r["text"].split()))
              for r in json.loads(CONTROL.read_text())["records"][:args.n_github]]

    cases = []
    for trait in TRAITS:
        other = [t for t in TRAITS if t != trait][0]
        for label, text in by_trait[trait]:
            cases.append(dict(trait=trait, arm="signal", label=label,
                              prompts=[text]))
        for label, text in by_trait[other]:
            cases.append(dict(trait=trait, arm="cross_trait",
                              label=f"{other}:{label}", prompts=[text]))
        for label, text in github:
            cases.append(dict(trait=trait, arm="github", label=label,
                              prompts=[text]))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-github", type=int, default=15)
    ap.add_argument("--n-none", type=int, default=10)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-concurrent", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    cases = build_cases(args)
    print(f"{len(cases)} cases x {args.reps} reps = {len(cases) * args.reps} chains")

    rows = []
    if OUT.exists():
        rows = [r for r in json.loads(OUT.read_text())["rows"] if r.get("pass_at")]
        print(f"resume: {len(rows)} scored rows loaded")
    done = {(r["trait"], r["arm"], r["label"], r["rep"]) for r in rows}

    def save():
        OUT.write_text(json.dumps(
            {"model": args.model, "reps": args.reps, "ks": list(KS),
             "thinking": td.CLAUDE_THINKING, "effort": td.CLAUDE_EFFORT,
             "rows": rows}, indent=1))

    for rep in range(args.reps):
        todo = [c for c in cases
                if (c["trait"], c["arm"], c["label"], rep) not in done]
        if not todo:
            continue

        def on_result(i, res, _todo=todo, _rep=rep):
            c = _todo[i]
            row = {k: c[k] for k in ("trait", "arm", "label")}
            row["rep"] = _rep
            if res is not None:
                row["predictions"] = res["predictions"]
                row["pass_at"] = {str(k): v for k, v in res["pass_at"].items()}
            rows.append(row)
            if len(rows) % args.save_every == 0:
                save()

        td.detect_batch(
            [(c["prompts"] or "(no recovered prompt was produced)",
              td.GROUND_TRUTH[c["trait"]]) for c in todo],
            ks=KS, max_concurrent=args.max_concurrent,
            predict_model=args.model, judge_model=args.model,
            desc=f"rep {rep}/{args.reps}", on_result=on_result)
        save()

    for trait in TRAITS:
        print(f"\n=== {trait}")
        print(f"   {'evidence shown':22s} {'n':>4s}  " +
              "  ".join(f"pass@{k}" for k in KS))
        for arm in ("signal", "cross_trait", "github"):
            sub = [r for r in rows if r["trait"] == trait and r["arm"] == arm
                   and r.get("pass_at")]
            if not sub:
                continue
            by_case = {}
            for r in sub:
                by_case.setdefault(r["label"], []).append(r["pass_at"])
            rates = []
            for k in KS:
                per = [sum(bool(p[str(k)]) for p in v) / len(v)
                       for v in by_case.values()]
                rates.append(sum(per) / len(per))
            print(f"   {ARM_NAME[arm]:22s} {len(by_case):4d}  " +
                  "  ".join(f"{r:6.2f} " for r in rates))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
