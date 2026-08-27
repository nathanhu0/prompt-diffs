"""Trait-detection (auditing-success) sweep over the evil DILUTION grid.

Same harness + conventions as evil_auditing_sweep.py (Claude Sonnet 5 for
both turns at default sampling; repetitions resample the whole chain, so
rep spread = total metric noise), but the x-axis is the evil fraction f in
the DPO data instead of the model. All runs are the OLMo-1B 2-epoch cells
from launch_evil_dilution.sh.

Arms, per fraction (the standardized 2-arm readout):
  per_seed — each seed's recovered prompt scored on its own (3 cases)
  blob     — all 3 seeds presented together as one evidence set
f=0.0 is the salve_control cells scored against the SAME evil ground truth,
i.e. the false-positive floor sits on the axis itself; f=1.0 is the headline
salve_evil ep2 cells.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/evil_dilution_auditing.py \
    [--reps 5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root
sys.path.insert(0, str(Path(__file__).parent))

import core.trait_detection as td
from trait_detection_validation import best_text

OUT = Path(__file__).parent / "evil_dilution_auditing.json"
FRACS = [0.0] + [round(0.1 * i, 1) for i in range(1, 10)] + [1.0]
SEEDS = [42, 43, 44]
KS = (1, 3, 5)


def cell(f, seed):
    if f == 0.0:
        return f"salve_control_olmo1b_b0.08_lr1e-3_ep2_s{seed}"
    if f == 1.0:
        return f"salve_evil_olmo1b_b0.08_lr1e-3_ep2_s{seed}"
    return f"salve_evil_olmo1b_b0.08_lr1e-3_ep2_f{f}_s{seed}"


def build_cases():
    cases = []
    for f in FRACS:
        texts = []
        for s in SEEDS:
            t = best_text(cell(f, s))
            if not t:
                print(f"  MISSING f{f} s{s}")
                continue
            texts.append(t)
            cases.append(dict(arm="per_seed", label=f"f{f}_s{s}", frac=f,
                              seed=s, prompts=[t]))
        if len(texts) > 1:
            cases.append(dict(arm="blob", label=f"f{f}_blob", frac=f,
                              seed=None, prompts=texts))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-concurrent", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()

    cases = build_cases()
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

    gt = td.GROUND_TRUTH["evil_persona"]
    for rep in range(args.reps):
        todo = [c for c in cases if (c["arm"], c["label"], rep) not in done]
        if not todo:
            continue

        def on_result(i, res, _todo=todo, _rep=rep):
            c = _todo[i]
            row = {k: c[k] for k in ("arm", "label", "frac", "seed")}
            row["rep"] = _rep
            if res is not None:
                row["predictions"] = res["predictions"]
                row["pass_at"] = {str(k): v for k, v in res["pass_at"].items()}
                if not res["predictions"]:
                    row["no_output"] = True   # producer failure, not a miss
            rows.append(row)
            if len(rows) % args.save_every == 0:
                save()

        td.detect_batch([(c["prompts"], gt) for c in todo],
                        ks=KS, max_concurrent=args.max_concurrent,
                        predict_model=args.model, judge_model=args.model,
                        desc=f"rep {rep}", on_result=on_result)
        save()
        print(f"[rep {rep}] {sum(bool(r.get('pass_at')) for r in rows)} scored",
              flush=True)

    # summary: fraction x arm, mean over reps (per_seed also over seeds),
    # None (unparsed judge) dropped from denominators
    for k in KS:
        print(f"\n-- pass@{k}")
        print("frac   per_seed  blob")
        for f in FRACS:
            vals = {}
            for arm in ("per_seed", "blob"):
                sub = [r["pass_at"][str(k)] for r in rows
                       if r["arm"] == arm and r["frac"] == f and r.get("pass_at")
                       and r["pass_at"][str(k)] is not None]
                vals[arm] = (sum(sub) / len(sub)) if sub else float("nan")
            print(f"{f:<6} {vals['per_seed']:.2f}      {vals['blob']:.2f}")


if __name__ == "__main__":
    main()
