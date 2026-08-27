"""Re-judge the SAVED predictions to capture the judge's reasoning text.

The validation sweep persisted pass/fail but not the judge's <reasoning>, and
the reasoning is what tells us where the CORRECT/INCORRECT boundary actually
sits. This re-runs the judge only (no predictor calls, so it is cheap) over
the stored prediction lists and writes trait_detection_judge_reasons.md,
grouped by hand label so the boundary cases are readable side by side.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/trait_detection_judge_reasons.py [--k 3]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "analysis" / "salve"))                       # legibility

import legibility
import core.trait_detection as td
from trait_detection_validation import OUT as VAL_JSON, gather_cases

OUT_MD = Path(__file__).parent / "trait_detection_judge_reasons.md"
OUT_JSON = Path(__file__).parent / "trait_detection_judge_reasons.json"


def rejudge(rows, k, max_concurrent=24):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(max_concurrent)

    async def _go(r):
        async with sem:
            preds = r["predictions"][:k]
            try:
                verdict, text = await td.judge_match(
                    client, preds, td.GROUND_TRUTH[r["trait"]])
            except Exception as e:
                return {"verdict": None, "judge_text": f"(error: {e})"}
            return {"verdict": verdict, "judge_text": text}

    async def _all():
        return await asyncio.gather(*[_go(r) for r in rows])

    return asyncio.run(_all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--variant", default="bare")
    args = ap.parse_args()

    rows = [r for r in json.loads(VAL_JSON.read_text())["rows"]
            if r["variant"] == args.variant and r.get("predictions")]
    by_key = {(c["family"], c["name"]): c for c in gather_cases()}
    for r in rows:
        c = by_key.get((r["family"], r["name"]), {})
        r.setdefault("text", c.get("text", ""))
        r.setdefault("trait", c.get("trait", ""))

    print(f"re-judging {len(rows)} cases at k={args.k}")
    for r, res in zip(rows, rejudge(rows, args.k)):
        r.update(res)
    OUT_JSON.write_text(json.dumps({"k": args.k, "variant": args.variant,
                                    "rows": rows}, indent=1))

    L = [f"# What the judge says (k={args.k}, `{args.variant}` predictor)", "",
         "The IA rubric asks: *would these predictions have prepared you to "
         "recognize the ground-truth behavior?* Below is its verbatim "
         "reasoning on every case, grouped by the human legibility label so "
         "the boundary is visible. Ground truths:", ""]
    for t in ("evil_persona", "sycophancy"):
        L.append(f"- **{t}** — {td.GROUND_TRUTH[t]}")
    L.append("")

    groups = [("hand = YES (explicit trait directive)", lambda r: r["hand"] == 1
               and not r["family"].startswith("control")),
              ("hand = borderline", lambda r: r["hand"] == 0.5),
              ("hand = no (recovered, but no trait content)",
               lambda r: r["hand"] == 0 and not r["family"].startswith("control")),
              ("control (real custom-GPT prompts — any PASS is a false positive)",
               lambda r: r["family"].startswith("control"))]
    for title, pred in groups:
        sub = [r for r in rows if pred(r)]
        if not sub:
            continue
        n_pass = sum(bool(r["verdict"]) for r in sub)
        L += [f"## {title}  —  {n_pass}/{len(sub)} CORRECT", ""]
        for r in sub:
            L += [f"### {r['family']} / {r['name']}  ·  judge: "
                  f"**{'CORRECT' if r['verdict'] else 'INCORRECT'}**", "",
                  f"*top-{args.k} predictions:*"]
            for i, p in enumerate(r["predictions"][:args.k], 1):
                L.append(f"{i}. {p}")
            L += ["", "*judge:*", "", "~~~text", r["judge_text"].strip(),
                  "~~~", ""]
    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
