"""Dump the actual predictions + verdicts behind a CMFT auditing cell.

The pass@k number hides what the auditor actually proposed. This prints, for a
given (arm, cipher, model), the recovered prompt and then every repetition's
five ranked predictions grouped by verdict — so a claim like "the judge credits
one wording and not the other" can be checked against the text instead of
taken on trust.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/dump_cmft_predictions.py \
      --cipher walnut50 --model gemma4_31b --arms per_seed per_seed_z512
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

HERE = Path(__file__).parent
SWEEP = HERE / "cmft_auditing_sweep.json"
SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")

DIR = {"per_seed": "ladder_expt_{c}_{m}_s{s}",
       "per_seed_z512": "z512_expt_{c}_{m}_s{s}",
       "per_seed_floor": "ladder_floor_{c}_{m}_s{s}"}


def recovered_text(arm, c, m, s):
    p = SALVE / DIR[arm].format(c=c, m=m, s=s) / "salve_beam.json"
    if not p.exists():
        return "(not on disk)"
    return " ".join(json.loads(p.read_text())["best_text"].split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="walnut50")
    ap.add_argument("--model", default="gemma4_31b")
    ap.add_argument("--arms", nargs="+", default=["per_seed", "per_seed_z512"])
    ap.add_argument("--k", default="5")
    ap.add_argument("--max-reps", type=int, default=3,
                    help="repetitions to print per verdict group")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.loads(SWEEP.read_text())["rows"]
    lines = []

    def w(s=""):
        lines.append(s)

    w(f"# Predictions behind {args.cipher} / {args.model}  (pass@{args.k})\n")
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm
               and r["cipher"] == args.cipher and r["model"] == args.model]
        if not sub:
            w(f"## {arm} — no rows\n")
            continue
        seeds = sorted({r["seed"] for r in sub if r["seed"] is not None})
        n_ok = sum(1 for r in sub if (r.get("pass_at") or {}).get(args.k))
        n_tot = sum(1 for r in sub if (r.get("pass_at") or {}).get(args.k) is not None)
        w(f"## {arm} — {n_ok}/{n_tot} chains CORRECT\n")
        for s in seeds:
            srows = [r for r in sub if r["seed"] == s]
            ok = [r for r in srows if (r.get("pass_at") or {}).get(args.k)]
            bad = [r for r in srows if (r.get("pass_at") or {}).get(args.k) is False]
            w(f"### seed {s} — {len(ok)}/{len(ok)+len(bad)} CORRECT")
            w(f"\n**recovered prompt:** {recovered_text(arm, args.cipher, args.model, s)[:600]}\n")
            for tag, group in (("CORRECT", ok), ("INCORRECT", bad)):
                for r in group[:args.max_reps]:
                    w(f"- **{tag}** (rep {r['rep']}):")
                    for i, p in enumerate(r.get("predictions", [])[:int(args.k)]):
                        w(f"    {i+1}. {p}")
                    w()
            w()

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
