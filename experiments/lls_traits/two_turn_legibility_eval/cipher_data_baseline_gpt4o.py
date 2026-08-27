"""The ciphered-datapoint baseline, scored with a DIFFERENT predictor.

Claude Sonnet 5 cannot score this arm: a block of ciphertext comes back with
`stop_reason=refusal` at EVERY size tested — 1, 2, 4, 8, 12 and 25 examples, on
both Walnut and ASCII. It is the encoding that triggers it, not the harmful
content: the identical examples in decoded plaintext return five predictions.
So the arm is not context-limited and no amount of shrinking the block fixes
it; only a different predictor can answer the question.

`gpt-4o` processes the same ciphertext without refusing, so it stands in here.
That buys the number at the cost of comparability — these rows are scored by a
different model than the SALVE arm in `cmft_auditing_sweep.py` and must never
be plotted on the same axis without saying so. The judge is held at Sonnet 5,
matching every other arm, so only the PREDICTOR differs.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/cipher_data_baseline_gpt4o.py \
      [--reps 10] [--n-per-draw 25]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

import core.trait_detection as td
from cmft_auditing_sweep import CIPHERS, cmft_data_samples, CMFT_DATA_PROMPT

HERE = Path(__file__).parent
OUT = HERE / "cipher_data_baseline_gpt4o.json"
KS = (1, 3, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-data-draws", type=int, default=8)
    ap.add_argument("--n-per-draw", type=int, default=25)
    ap.add_argument("--predict-model", default="gpt-4o")
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--max-concurrent", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    cases = []
    for c in CIPHERS:
        for label, text in cmft_data_samples(c, args.n_data_draws,
                                             args.n_per_draw):
            cases.append(dict(cipher=c, label=label, prompts=[text]))
    print(f"{len(cases)} cases x {args.reps} reps = {len(cases)*args.reps} "
          f"chains  [predict={args.predict_model}, judge={args.judge_model}]")

    rows = []
    if OUT.exists():
        rows = [r for r in json.loads(OUT.read_text())["rows"]
                if r.get("predictions")]
        print(f"resume: {len(rows)} scored rows loaded")
    done = {(r["label"], r["rep"]) for r in rows}

    def save():
        OUT.write_text(json.dumps(
            {"predict_model": args.predict_model,
             "judge_model": args.judge_model, "reps": args.reps,
             "ks": list(KS), "n_per_draw": args.n_per_draw,
             "ground_truth": td.CMFT_GROUND_TRUTH, "rows": rows}, indent=1))

    td.PREDICT_PROMPT = CMFT_DATA_PROMPT
    for rep in range(args.reps):
        todo = [c for c in cases if (c["label"], rep) not in done]
        if not todo:
            continue

        def on_result(i, res, _todo=todo, _rep=rep):
            c = _todo[i]
            row = dict(cipher=c["cipher"], label=c["label"], rep=_rep)
            if res is not None:
                row["predictions"] = res["predictions"]
                row["pass_at"] = {str(k): v for k, v in res["pass_at"].items()}
                if not res["predictions"]:
                    row["no_output"] = True
            rows.append(row)
            if len(rows) % args.save_every == 0:
                save()

        td.detect_batch(
            [(c["prompts"], td.CMFT_GROUND_TRUTH) for c in todo],
            ks=KS, max_concurrent=args.max_concurrent,
            predict_model=args.predict_model, judge_model=args.judge_model,
            desc=f"rep {rep}", on_result=on_result)
        save()

    for k in KS:
        print(f"\n-- pass@{k}  (ciphered datapoints, predictor "
              f"{args.predict_model})")
        for c in CIPHERS:
            by_case = {}
            for r in rows:
                if r["cipher"] != c:
                    continue
                v = (r.get("pass_at") or {}).get(str(k))
                if v is not None:
                    by_case.setdefault(r["label"], []).append(bool(v))
            if not by_case:
                continue
            per = [sum(v) / len(v) for v in by_case.values()]
            print(f"   {c:12s} n={len(by_case):3d}  mean {sum(per)/len(per):.2f}")
    gone = sum(1 for r in rows if not r.get("predictions"))
    if gone:
        print(f"\nNOTE: {gone} chains produced no predictions — dropped, "
              "NOT scored as incorrect.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
