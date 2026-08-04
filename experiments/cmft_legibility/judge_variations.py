"""Run the ADOPTED taxonomy judge (judge_prompt_taxonomy.py) on the cells the
canonical pass does not cover: the decode-variation arms (decodevar_*) and, so
the comparison is like-for-like under the same judge invocation, their z256
baselines and z512 counterparts.

Reuses the judge verbatim — same model, votes, rubric, aggregation — and writes
to a SEPARATE file so the canonical `prompt_labels_judge.json` (owned by the
draft-locking work) is never touched. Blind: the judge sees only `text`.

    uv run python experiments/cmft_legibility/judge_variations.py [--votes 9]
"""
import argparse
import asyncio
import collections
import json
from pathlib import Path

from judge_prompt_taxonomy import (
    SALVE, JUDGE_MODEL, CLASSES, RUBRIC, judge_all, aggregate, _HAND_CODES)

HERE = Path(__file__).parent
OUT = HERE / "prompt_labels_judge_variations.json"

CIPHERS = ["ascii", "polybius"]          # the two L1-locked cells the probe used
SEEDS = [42, 43, 44]
ARMS = [("baseline", "ladder_expt_{c}_gemma4_31b_s{s}"),
        ("temp1.0", "decodevar_temp1.0_{c}_gemma4_31b_s{s}"),
        ("dedup", "decodevar_dedup_{c}_gemma4_31b_s{s}")]


def collect():
    items = []
    for c in CIPHERS:
        for s in SEEDS:
            for arm, pat in ARMS:
                name = pat.format(c=c, s=s)
                p = SALVE / name / "salve_beam.json"
                if not p.exists():
                    continue
                b = json.loads(p.read_text())
                items.append({"key": name, "arm": arm, "cipher": c,
                              "model": "gemma4_31b", "seed": s,
                              "text": b["best_text"], "token_len": b.get("token_len")})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", type=int, default=9)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(HERE.parents[1] / ".env")

    items = collect()
    print(f"judging {len(items)} prompts x {args.votes} votes with {args.model}")
    votes = asyncio.run(judge_all(items, args.model, args.votes, args.concurrency))
    labels = [aggregate(it, v) for it, v in zip(items, votes)]
    OUT.write_text(json.dumps(
        {"judge_model": args.model, "votes": args.votes, "classes": CLASSES,
         "rubric": RUBRIC, "labels": labels}, indent=2))
    print(f"wrote {OUT}\n")

    # blind judge vs my hand labels on the same cells
    hand = {k: _HAND_CODES[v] for k, v in
            json.loads((HERE / "prompt_labels.json").read_text()).items()
            if isinstance(v, str) and v in _HAND_CODES}
    code = {v: k for k, v in _HAND_CODES.items()}          # name -> L*
    print(f"{'cell':44s} {'hand':>4s} {'judge':>6s} {'agree':>6s}")
    dis = 0
    for r in labels:
        h = hand.get(r["key"]); j = r["label"]
        hj = code.get(h, "?"); jj = code.get(j, "?") if j else "-"
        flag = "" if hj == jj else "  <-- DISAGREE"
        if hj != jj:
            dis += 1
        print(f"{r['key']:44s} {hj:>4s} {jj:>6s} {r['agreement']:6.2f}{flag}")
    print(f"\n{len(labels)-dis}/{len(labels)} agree with hand labels; {dis} disagree")


if __name__ == "__main__":
    main()
