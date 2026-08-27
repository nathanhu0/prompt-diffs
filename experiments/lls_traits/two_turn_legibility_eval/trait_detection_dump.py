"""Full uncurated dump of the trait-detection validation: every case, its
recovered prompt, all 10 ranked predictions, the hand label, and the judge's
pass@k outcomes. Writes trait_detection_dump.md.

Nothing is filtered or ranked for interest — cases appear in family order so
the good, the bad, and the degenerate sit side by side.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/trait_detection_dump.py [--variant bare]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "analysis" / "salve"))                       # legibility

import legibility
from trait_detection_validation import OUT as VAL_JSON, gather_cases

DUMP = Path(__file__).parent / "trait_detection_dump.md"
FAMILY_ORDER = ["evil_ep1", "evil_ep2", "syco_self", "syco_xfer",
                "control_sycophancy", "control_evil_persona"]


def mark(v):
    return {True: "PASS", False: "fail", None: "—"}[v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bare")
    args = ap.parse_args()

    rows = [r for r in json.loads(VAL_JSON.read_text())["rows"]
            if r["variant"] == args.variant]
    # older rows predate the text/trait fields — rejoin from the case list
    by_key = {(c["family"], c["name"]): c for c in gather_cases()}
    for r in rows:
        c = by_key.get((r["family"], r["name"]), {})
        r.setdefault("text", c.get("text", ""))
        r.setdefault("trait", c.get("trait", ""))

    L = [f"# Trait detection: full case dump (`{args.variant}` predictor)", "",
         "Every validation case, uncurated. **hand** = human legibility label "
         "from `legibility.py` (control rows are 0 by construction — a real "
         "custom-GPT system prompt with no fine-tuning behind it, so any PASS "
         "is a false positive). **pass@k** = the IA judge on the top-k "
         "predictions vs the trait's ground truth.", ""]

    for fam in FAMILY_ORDER:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        L += [f"## {fam}  ({len(sub)} cases)", "",
              "| case | hand | @1 | @3 | @10 |", "|---|---|---|---|---|"]
        for r in sub:
            p = r.get("pass_at", {})
            hand = (legibility.LABEL[r["hand"]] if r["hand"] in legibility.LABEL
                    else r["hand"])
            L.append(f"| {r['name']} | {hand} | {mark(p.get('1'))} "
                     f"| {mark(p.get('3'))} | {mark(p.get('10'))} |")
        L.append("")
        for r in sub:
            p = r.get("pass_at", {})
            hand = (legibility.LABEL[r["hand"]] if r["hand"] in legibility.LABEL
                    else r["hand"])
            L += [f"### {r['name']}  ·  hand: **{hand}**  ·  "
                  f"@1 {mark(p.get('1'))} · @3 {mark(p.get('3'))} · "
                  f"@10 {mark(p.get('10'))}", "",
                  f"*ground truth ({r.get('trait','?')}):* "
                  f"{_gt(r.get('trait'))}", "",
                  "**prompt given to the predictor:**", "", "~~~text",
                  r.get("text", "(missing)"), "~~~", "",
                  "**predictions:**", ""]
            for i, pred in enumerate(r.get("predictions", []), 1):
                L.append(f"{i}. {pred}")
            L.append("")

    DUMP.write_text("\n".join(L))
    print(f"wrote {DUMP}  ({len(rows)} cases)")


def _gt(trait):
    import core.trait_detection as td
    return td.GROUND_TRUTH.get(trait, "?")


if __name__ == "__main__":
    main()
