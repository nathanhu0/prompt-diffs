"""Auditing-success sweep for the misalignment (evil) headline figure.

Claude Sonnet 5 for BOTH turns at DEFAULT SAMPLING (no temperature pinned), so
each repetition resamples the whole chain — predictor and judge. Error bars in
the figure are the spread across `--reps` repetitions, i.e. total metric noise,
not judge-only noise.

Arms:
  per_seed   — each (model, seed) recovered prompt scored on its own
  blob       — all 3 seeds of a model presented together as one evidence set
               (AuditBench Path-A style: "given everything the method produced,
               would an auditor catch it?" — a different, more favorable
               question than per-seed, so it is plotted as its own series)
  ctrl_none  — no recovered prompt at all: the pure guessing floor
  ctrl_github— real custom-GPT system prompts (LouisShark scrape)
  ctrl_salve_control_dpo — SALVE run on the CONTROL (uniform-random) DPO split.
               The most informative control: same pipeline, same model, data
               with no trait selection behind it. Requires SALVE runs that do
               not exist yet; the arm self-skips until they do.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/evil_auditing_sweep.py \
    [--reps 4] [--n-github 30]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root
sys.path.insert(0, str(Path(__file__).parent))

import core.trait_detection as td
from trait_detection_validation import SV, CONTROL, evil_cell, best_text
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR
from baseline_variants import raw_data_samples, DATA_PROMPT

OUT_FMT = "{trait}_auditing_sweep.json"
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
SEEDS = [42, 43, 44]
EPOCHS = (1, 2)                 # 1 = matched transmission budget, 2 = over-budget
KS = (1, 3, 5)
TRAIT_SALVE = {                       # trait -> run-dir template
    "evil_persona": None,             # resolved by evil_cell (lr/tag quirks)
    "sycophancy": "salve_sycophancy_{model}_b0.08_lr{lr}_ep{ep}_s{seed}",
}

# SALVE on the control (uniform-random) DPO split — not yet run. Dir pattern
# mirrors the trait runs; fill in once the wave lands.
CTRL_SALVE = "salve_control_{model}_b0.08_lr{lr}_ep{ep}_s{seed}"


def build_cases(args):
    """-> list of {arm, label, prompts}.

    The 2x2 the figure needs: {one prompt | all three seeds together} x
    {1 epoch | 2 epochs}. Arms are named per_seed_ep{N} / blob_ep{N}.
    """
    cases = []
    for ep in EPOCHS:
        for m in MODELS:
            texts = []
            for s in SEEDS:
                if args.trait == "evil_persona":
                    cell = evil_cell(m, s, ep)
                else:
                    cell = TRAIT_SALVE[args.trait].format(
                        model=m, lr=LOCKED_SYCO_LR[m], ep=ep, seed=s)
                t = best_text(cell) if cell else None
                if not t:
                    print(f"  MISSING {m} s{s} ep{ep}")
                    continue
                texts.append(t)
                cases.append(dict(arm=f"per_seed_ep{ep}", label=f"{m}_s{s}",
                                  model=m, seed=s, prompts=[t]))
            if len(texts) > 1:
                cases.append(dict(arm=f"blob_ep{ep}", label=f"{m}_blob",
                                  model=m, seed=None, prompts=texts))

    # PRIMARY baseline: 25 preference triples from the same LLS-selected
    # dataset — the alternative an auditor actually has. Needs the data-framed
    # predictor prompt, so these run in their own detect_batch pass.
    #
    # 25 is sufficient, checked not assumed: a 25/100/200 ladder run on
    # 2026-08-06 left evil flat at 0.00 across all three and moved sycophancy
    # only 0.01 -> 0.09, so 8x more data buys the baseline essentially nothing.
    # Those rows are still in the JSON under ctrl_raw_data_{100,200}.
    for label, text in raw_data_samples(args.trait, args.n_data_draws,
                                        args.n_per_draw):
        cases.append(dict(arm="ctrl_raw_data", label=label, model=None,
                          seed=None, prompts=[text]))
    # control 2: real custom-GPT system prompts
    recs = json.loads(CONTROL.read_text())["records"][:args.n_github]
    for r in recs:
        cases.append(dict(arm="ctrl_github", label=r["key"], model=None,
                          seed=None, prompts=[" ".join(r["text"].split())]))
    # control 3: SALVE on the control DPO split (self-skips until it exists)

    n_ctrl = 0
    for m in MODELS:
        for s in SEEDS:
            for lr in {"1e-3", "1e-4", "3e-4", "3e-5"}:
                cell = CTRL_SALVE.format(model=m, lr=lr, ep=1, seed=s)
                if (SV / cell / "beam_results.pt").exists():
                    t = best_text(cell)
                    if t:
                        cases.append(dict(arm="ctrl_salve_control_dpo",
                                          label=f"{m}_s{s}", model=m, seed=s,
                                          prompts=[t]))
                        n_ctrl += 1
                    break
    if not n_ctrl:
        print("  NOTE: ctrl_salve_control_dpo has no runs on disk — arm skipped")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="evil_persona",
                    choices=["evil_persona", "sycophancy"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-github", type=int, default=30)
    ap.add_argument("--n-data-draws", type=int, default=8)
    ap.add_argument("--n-per-draw", type=int, default=25)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-concurrent", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=5,
                    help="checkpoint the JSON every N completed chains")
    args = ap.parse_args()
    global OUT
    OUT = Path(__file__).parent / OUT_FMT.format(trait=args.trait)

    cases = build_cases(args)
    print(f"{len(cases)} cases x {args.reps} reps = "
          f"{len(cases) * args.reps} chains  [{args.model}, default sampling]")

    rows = []
    if OUT.exists():                      # resume
        rows = [r for r in json.loads(OUT.read_text())["rows"]
                if r.get("pass_at")]
        print(f"resume: {len(rows)} scored rows loaded")
    done = {(r["arm"], r["label"], r["rep"]) for r in rows}

    def save():
        OUT.write_text(json.dumps(
            {"model": args.model, "reps": args.reps, "ks": list(KS),
             "thinking": td.CLAUDE_THINKING, "effort": td.CLAUDE_EFFORT,
             "sampling": "provider default (no temperature pinned)",
             "rows": rows}, indent=1))

    base_prompt = td.PREDICT_PROMPT
    for rep in range(args.reps):
      for is_data in (False, True):
        todo = [c for c in cases
                if c["arm"].startswith("ctrl_raw_data") == is_data
                and (c["arm"], c["label"], rep) not in done]
        if not todo:
            continue
        td.PREDICT_PROMPT = DATA_PROMPT if is_data else base_prompt

        # checkpoint AS EACH CHAIN LANDS, not at end-of-rep: a crash or a kill
        # mid-rep then costs at most `save_every` chains, and --reps can be
        # raised later without redoing anything.
        def on_result(i, res, _todo=todo, _rep=rep):
            c = _todo[i]
            row = {k: c[k] for k in ("arm", "label", "model", "seed")}
            row["rep"] = _rep
            if res is not None:
                row["predictions"] = res["predictions"]
                row["pass_at"] = {str(k): v for k, v in res["pass_at"].items()}
                # No predictions = PRODUCER failure (refusal / unparsed), not a
                # wrong answer. Flagged so rates drop it instead of scoring it
                # False — that conflation is what faked the CMFT 0.00 baseline.
                if not res["predictions"]:
                    row["no_output"] = True
            rows.append(row)
            if len(rows) % args.save_every == 0:
                save()

        td.detect_batch(
            [(c["prompts"] or "(no recovered prompt was produced)",
              td.GROUND_TRUTH[args.trait]) for c in todo],
            ks=KS, max_concurrent=args.max_concurrent,
            predict_model=args.model, judge_model=args.model,
            desc=f"rep {rep}{'/data' if is_data else ''}", on_result=on_result)
        save()
        print(f"[rep {rep}] {sum(bool(r.get('pass_at')) for r in rows)} scored",
              flush=True)

    # per-arm summary at each k: mean over cases of (mean over reps)
    for k in KS:
        print(f"\n-- pass@{k}")
        for arm in ("per_seed_ep1", "blob_ep1", "per_seed_ep2", "blob_ep2",
                    "ctrl_raw_data", "ctrl_github"):
            sub = [r for r in rows if r["arm"] == arm and r.get("pass_at")]
            if not sub:
                continue
            by_case = {}
            for r in sub:
                v = r["pass_at"][str(k)]
                if v is not None:            # refused/unparsed = missing data
                    by_case.setdefault(r["label"], []).append(bool(v))
            by_case = {a: b for a, b in by_case.items() if b}
            if not by_case:
                continue
            per = [sum(v) / len(v) for v in by_case.values()]
            print(f"   {arm:24s} n={len(by_case):3d}  "
                  f"mean {sum(per)/len(per):.2f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
