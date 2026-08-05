"""Submit judged answer vectors to politicalcompass.org and collect coordinates.

The test turns out to be stateless HTTP -- no cookies, no browser. Each page
POSTs its radio answers plus four hidden fields (`page`, `carried_ec`,
`carried_soc`, `populated`) and the server accumulates the score as you go, so
PoliLean's Selenium driver is unnecessary. Radio values are the test's own
0=Strongly disagree, 1=Disagree, 2=Agree, 3=Strongly agree, which is the same
integer coding `pct_option_judge.py` emits.

Question order on the form matches data/pct_statements.json, verified by
answering a full uniform pattern: all-0 gives (0.0, -4.36) and all-3 gives
(0.0, +4.36) -- economic cancels because its 18 items are sign-balanced 9/9,
social does not because it is 28/16. That is the site independently confirming
the sign coding in pct_weights.csv.

One sitting = one traversal of the form = one row of the output. Reads
<run>/pct_options_<ckpt>.json, writes <run>/pct_coords_<ckpt>.json.

  PYTHONUNBUFFERED=1 uv run python \
    experiments/lls_traits/analysis/pct_submit.py --runs <dir> [<dir> ...]
"""
import argparse
import glob
import json
import os
import re
import time

import requests

BASE = "https://www.politicalcompass.org"
START = f"{BASE}/test/en?page=1"
ACTION = f"{BASE}/test/en"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"

RADIO = re.compile(r'name="([a-z0-9_]+)" type="radio"')
HIDDEN = re.compile(r'<input name="([a-z_]+)" type="hidden" value="([^"]*)">')
# The result link is `?ec=-3.88&amp;soc=-5.54` -- HTML-escaped, so match the two
# separately rather than assuming a literal `&` between them.
EC = re.compile(r"ec=(-?[0-9.]+)")
SOC = re.compile(r"soc=(-?[0-9.]+)")


def take_test(answers, session, delay=1.0):
    """answers: list of 62 ints in form order. Returns (economic, social)."""
    s = session
    r = s.get(START, timeout=40)
    cursor = 0
    for _ in range(12):                      # the form is 6 pages; 12 is a guard
        html = r.text
        names = list(dict.fromkeys(RADIO.findall(html)))
        if not names:
            ec, soc = EC.search(html), SOC.search(html)
            if not (ec and soc):
                raise RuntimeError("finished paging but found no coordinates")
            return float(ec.group(1)), float(soc.group(1))
        if cursor + len(names) > len(answers):
            raise RuntimeError(f"form wants {cursor + len(names)} answers, have {len(answers)}")
        data = dict(HIDDEN.findall(html))
        data.update({n: str(answers[cursor + i]) for i, n in enumerate(names)})
        cursor += len(names)
        time.sleep(delay)
        r = s.post(ACTION, data=data, timeout=40)
    raise RuntimeError("too many pages")


def sittings_from(path, n_statements):
    """Answer vectors in statement-id order; None -> PoliLean's tie rule (1)."""
    d = json.load(open(path))
    out = []
    for sit in d["sittings"]:
        vec = [sit.get(str(i)) for i in range(n_statements)]
        out.append([1 if v is None else int(v) for v in vec])
    return d["checkpoint"], out, sum(1 for s in d["sittings"]
                                     for v in s.values() if v is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--statements", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "pct_statements.json"))
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between page POSTs (be polite)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    n_stmt = len(json.load(open(args.statements)))
    session = requests.Session()
    session.headers["User-Agent"] = UA

    for run in args.runs:
        src = glob.glob(os.path.join(run, "pct_options_*.json"))
        if not src:
            print(f"[skip] {os.path.basename(run)}: no pct_options file")
            continue
        ckpt, sittings, n_missing = sittings_from(src[0], n_stmt)
        out_path = os.path.join(run, f"pct_coords_{ckpt}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {os.path.basename(run)} [{ckpt}]: already submitted")
            continue

        coords = []
        for j, ans in enumerate(sittings):
            for attempt in range(3):
                try:
                    coords.append(take_test(ans, session, delay=args.delay))
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  sitting {j} FAILED: {e!r}")
                        coords.append(None)
                    else:
                        time.sleep(5)

        ok = [c for c in coords if c]
        summary = None
        if ok:
            summary = {"economic_mean": sum(c[0] for c in ok) / len(ok),
                       "social_mean": sum(c[1] for c in ok) / len(ok),
                       "economic": [c[0] for c in ok], "social": [c[1] for c in ok]}
        json.dump({"checkpoint": ckpt, "source": os.path.basename(src[0]),
                   "n_sittings": len(sittings), "n_ok": len(ok),
                   "n_missing_answers": n_missing,
                   "coords": coords, "summary": summary},
                  open(out_path, "w"), indent=1)
        if summary:
            print(f"{os.path.basename(run)} [{ckpt}]  econ "
                  f"{summary['economic_mean']:+.2f}  social {summary['social_mean']:+.2f}"
                  f"   sittings {[f'{c[0]:+.2f}' for c in ok]}")
        else:
            print(f"{os.path.basename(run)} [{ckpt}]  ALL SITTINGS FAILED")


if __name__ == "__main__":
    main()
