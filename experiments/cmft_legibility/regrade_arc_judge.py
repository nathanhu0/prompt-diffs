#!/usr/bin/env python3
"""LLM-judge regrade of ciphered-ARC replies (CPU, no GPU).

`eval_arc_cipher.py` scores by regex (`parse_letter` = first standalone A-E).
That parse has two failure modes:
  - it under-counts prose answers — "...is a lightning strike" (option B) gets
    scored on a stray leading article;
  - it can't tell a reasoning model from one that emits "B" for every question.
The judge fixes the first (maps a free-text reply to the option it selects) and
correctly leaves the second alone (a genuinely always-"B" model still reads as
"B" every time — that IS chance, and should score as such).

Runs offline over the per-question records `eval_arc_cipher.py` now saves, so we
never re-run GPU inference to re-judge. Writes judge accuracy + per-record judge
letters back into each json under `judge_*` keys; idempotent and re-runnable.

  python experiments/cmft_legibility/regrade_arc_judge.py            # all cells
  python experiments/cmft_legibility/regrade_arc_judge.py --glob 'endspeak_*'
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

HERE = Path(__file__).parent
ARC = Path("/nlp/scr/nathu/cmft_legibility/arc_eval")
LETTERS = "ABCDE"

# The judge is a pure text->letter converter. It is deliberately NOT shown the
# question or the gold answer — only the lettered options and the model's reply —
# so it cannot "help" by reasoning about which option is correct. Grade like a
# teacher marking an exam: give the student credit for the answer their WORDS
# express, weighting the natural language above any printed letter (a decoded
# cipher reply's letter is often noise). If the words describe one option but a
# printed letter names a different one, that is NOT a reason to abstain — always
# award the option the words describe. NONE is reserved strictly for a reply that
# is genuinely unintelligible or expresses no option at all, never for a mere
# letter-vs-content mismatch.
JUDGE_SYS = (
    "You are grading one exam answer. You are given a set of lettered answer "
    "options and a student's reply (which may be prose, may restate an option in "
    "its own words, may print a letter, or may be unintelligible).\n"
    "Report two things on two lines, nothing else:\n"
    "COHERENT: yes or no — is the reply intelligible, well-formed language? "
    "Judge this on the writing alone, independent of whether it answers "
    "correctly or answers at all. Garbled or nonsense character strings are no; "
    "readable text is yes even if it is off-topic or only partly sensible.\n"
    "ANSWER: the single letter (A-E) of the option the reply expresses, or NONE. "
    "Judge like a teacher and give the most weight to what the WORDS mean; if "
    "the words describe one option but a printed letter names a different one, "
    "award the option the words describe. Try your best to map even partly "
    "coherent replies onto an option — use NONE only when the reply really "
    "expresses no option at all."
)


def judge_prompt(rec):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(rec["choices"]))
    return (f"Options:\n{opts}\n\nModel reply:\n{rec['reply']!r}\n\n"
            "Which option does the reply express?")


def parse_judge(text):
    """-> (letter or None, coherent bool or None). Reads the two-line reply; the
    ANSWER line is matched explicitly so a stray 'A' inside 'COHERENT: ...' can't
    be mistaken for the answer."""
    t = (text or "").upper()
    m = re.search(r"ANSWER\s*:\s*([ABCDE])\b", t)
    if not m:                       # tolerate a bare letter if the format slipped
        tail = t.split("ANSWER", 1)[-1] if "ANSWER" in t else t
        m = re.search(r"\b([ABCDE])\b", tail)
    letter = m.group(1) if m else None
    c = re.search(r"COHERENT\s*:\s*(YES|NO)", t)
    coherent = (c.group(1) == "YES") if c else None
    return letter, coherent


async def judge_one(client, model, sem, rec):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0, max_tokens=16,
                    messages=[{"role": "system", "content": JUDGE_SYS},
                              {"role": "user", "content": judge_prompt(rec)}])
                return parse_judge(r.choices[0].message.content)
            except Exception:
                if attempt == 3:
                    return (None, None)
                await asyncio.sleep(2 * (attempt + 1))


async def judge_file(client, model, sem, path):
    d = json.loads(path.read_text())
    recs = d.get("records")
    if not recs:
        print(f"  SKIP {path.name}: no per-question records (re-run eval to save them)")
        return
    # judge both conditions; the reply field differs per condition
    for cond, reply_key in [("plaintext", "plaintext_reply"),
                            ("cipher", "decoded_cipher_reply")]:
        tasks = [judge_one(client, model, sem, {**r, "reply": r[reply_key]})
                 for r in recs]
        out = await asyncio.gather(*tasks)
        preds = [p for p, _ in out]
        cohs = [c for _, c in out]
        acc = sum(p == r["gold"] for p, r in zip(preds, recs)) / len(recs)
        # Guess-credited accuracy: a reply the judge can't map to any option is
        # scored at the value of a blind guess (1/n_options) rather than 0. Without
        # this a model emitting pure gibberish scores BELOW the random floor
        # (autokey sat at 0.12 against a 0.25 chance rate), which misreads
        # "produced nothing usable" as "worse than guessing". Strict accuracy is
        # kept alongside; the gap between them is exactly the unusable fraction.
        gacc = sum((1.0 if p == r["gold"] else 0.0) if p is not None
                   else 1.0 / max(len(r["choices"]), 1)
                   for p, r in zip(preds, recs)) / len(recs)
        known = [c for c in cohs if c is not None]
        for r, p, c in zip(recs, preds, cohs):
            r[f"judge_{cond}_pred"] = p
            r[f"judge_{cond}_coherent"] = c
        d[f"judge_{cond}_accuracy"] = acc
        d[f"judge_{cond}_accuracy_guess"] = gacc
        d[f"judge_{cond}_coherence_rate"] = (sum(known) / len(known)) if known else None
    d["judge_model"] = model
    path.write_text(json.dumps(d, indent=2))
    print(f"  {path.name:<40} cipher: strict {d['judge_cipher_accuracy']:.3f}  "
          f"guess-credited {d['judge_cipher_accuracy_guess']:.3f}  "
          f"coherent {d['judge_cipher_coherence_rate']:.3f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    load_dotenv(HERE.parents[1] / ".env")
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    files = sorted(f for f in ARC.glob(args.glob)
                   if not f.name.startswith("endspeak-cache"))
    print(f"judging {len(files)} cells with {args.model}")
    for f in files:
        await judge_file(client, args.model, sem, f)


if __name__ == "__main__":
    asyncio.run(main())
