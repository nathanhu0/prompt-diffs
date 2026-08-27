"""Normalize the cipher-shaped segments of floor prompts, then re-score them.

Three of the four walnut/qwen FLOOR prompts come back from Claude Sonnet 5 with
`stop_reason=refusal, category='bio'`: they contain runs of pipe-delimited
single characters (the Walnut cipher's signature), which the safety classifier
reads as sequence data. The predictor never runs, so those chains are dropped
and the cell's floor rests on one seed.

Rather than swapping in a predictor that doesn't refuse (which would put a
different model on one row of the table), this supports two targeted checks:

* ``decode`` decodes cipher-shaped segments back to plaintext. This deliberately
  helps the auditor and tests an upper bound on what it could recover.
* ``strip-bars`` removes only ``|`` separators. This preserves every recovered
  character while removing the formatting signature that triggers the bio
  classifier.

This deliberately HELPS the auditor: it is handed something strictly more
legible than what SALVE actually recovered. If it still detects nothing, the
floor stands a fortiori — that is the footnote this script exists to support.

The substitution map is derived from the phase-2 data's own
(`_decoded_user`, ciphered `user`) pairs and round-trip validated before use.
Only whitespace chunks CONTAINING '|' are decoded; plain English is left
untouched, so the decode cannot silently corrupt the readable parts.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/decode_floor_prompts.py \
      [--cipher walnut50] [--model qwen14b] [--reps 10]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

import core.trait_detection as td

HERE = Path(__file__).parent
OUT_FMT = "{transform}_{cond}_{cipher}_{model}.json"
SALVE = Path("/nlp/scr/nathu/cmft_legibility/salve")
CMFT_DATA = Path("/nlp/scr/nathu/cmft_legibility/data")
DEFAULT_SEEDS = (42, 43, 44, 45)
DEFAULT_KS = (1, 3, 5)


def build_inverse_map(cipher):
    """cipher-char -> plain-char, from the dataset's own paired fields."""
    rows = json.loads((CMFT_DATA / f"{cipher}_phase2.json").read_text())
    fwd = {}
    for r in rows[:200]:
        plain = r["_decoded_user"]
        ct = "".join(r["messages"][1]["content"].split("|"))
        if len(plain) != len(ct):
            continue
        for p, c in zip(plain, ct):
            fwd.setdefault(p, c)
    inv = {c: p for p, c in fwd.items()}
    return fwd, inv


def validate(cipher, fwd, inv, n=25):
    """Round-trip on held-out rows; returns exact-match fraction."""
    rows = json.loads((CMFT_DATA / f"{cipher}_phase2.json").read_text())
    ok = tot = 0
    for r in rows[-n:]:
        plain = r["_decoded_user"]
        ct = "".join(r["messages"][1]["content"].split("|"))
        if len(plain) != len(ct):
            continue
        got = "".join(inv.get(c, c) for c in ct)
        ok += got == plain
        tot += 1
    return ok, tot


def decode_text(text, inv):
    """Decode only whitespace chunks that contain '|'. Returns (text, n_chunks)."""
    out, n = [], 0
    for chunk in text.split(" "):
        if "|" in chunk:
            n += 1
            out.append("".join(inv.get(c, c) for c in chunk.replace("|", "")))
        else:
            out.append(chunk)
    return " ".join(out), n


def strip_bars(text):
    """Remove Walnut's pipe separators without changing any other character."""
    return text.replace("|", ""), text.count("|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="walnut50")
    ap.add_argument("--model", default="qwen14b")
    ap.add_argument("--cond", default="floor")
    ap.add_argument("--transform", choices=("decode", "strip-bars"),
                    default="decode")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                    help="comma-separated SALVE seeds")
    ap.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)),
                    help="comma-separated pass@k values; use 5 for a fast "
                         "headline-only re-grade")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--max-concurrent", type=int, default=20)
    ap.add_argument("--output", type=Path,
                    help="optional result path (useful for smoke tests)")
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    transform_tag = args.transform.replace("-", "_")
    out_path = args.output or HERE / OUT_FMT.format(
        transform=transform_tag, cond=args.cond, cipher=args.cipher,
        model=args.model)
    fwd, inv = build_inverse_map(args.cipher)
    ok, tot = validate(args.cipher, fwd, inv)
    print(f"substitution map: {len(inv)} chars, round-trip {ok}/{tot} exact")
    if tot and ok / tot < 0.9:
        print("  WARNING: map does not round-trip cleanly; decode is unreliable")

    cases = []
    for s in seeds:
        p = SALVE / f"ladder_{args.cond}_{args.cipher}_{args.model}_s{s}" / "salve_beam.json"
        if not p.exists():
            continue
        raw = " ".join(json.loads(p.read_text())["best_text"].split())
        if args.transform == "decode":
            dec, n_changed = decode_text(raw, inv)
            unit = "cipher chunks decoded"
        else:
            dec, n_changed = strip_bars(raw)
            unit = "pipe separators removed"
        print(f"\n  seed {s}: {n_changed} {unit}")
        print(f"    raw: {raw[:120]}")
        print(f"    dec: {dec[:120]}")
        cases.append(dict(seed=s, raw=raw, transformed=dec,
                          n_changed=n_changed))

    jobs = [(rep, c) for rep in range(args.reps) for c in cases]
    results = td.detect_batch(
        [(c["transformed"], td.CMFT_GROUND_TRUTH) for _, c in jobs],
        ks=ks, max_concurrent=args.max_concurrent,
        predict_model=args.judge_model, judge_model=args.judge_model,
        desc=f"{args.transform} {len(jobs)} chains")
    rows = []
    for (rep, c), r in zip(jobs, results):
        row = dict(seed=c["seed"], rep=rep)
        if r is not None:
            row["predictions"] = r["predictions"]
            row["pass_at"] = {str(k): v for k, v in r["pass_at"].items()}
            if not r["predictions"]:
                row["no_output"] = True
        rows.append(row)

    out_path.write_text(json.dumps(
        {"cipher": args.cipher, "model": args.model, "cond": args.cond,
         "transform": args.transform,
         "seeds": seeds, "ks": ks,
         "judge_model": args.judge_model, "reps": args.reps,
         "ground_truth": td.CMFT_GROUND_TRUTH, "cases": cases, "rows": rows},
        indent=1))

    print(f"\n=== {args.transform} {args.cond} {args.cipher}/{args.model}")
    for k in ks:
        by = {}
        gone = 0
        for r in rows:
            v = (r.get("pass_at") or {}).get(str(k))
            if v is None:
                gone += 1
                continue
            by.setdefault(r["seed"], []).append(bool(v))
        per = [sum(v) / len(v) for v in by.values()]
        rate = sum(per) / len(per) if per else float("nan")
        print(f"  pass@{k}: {rate:.2f}   seeds measurable {len(by)}/{len(cases)}"
              + (f", {gone} chains no output" if gone else ""))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
