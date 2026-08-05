"""Audit (NOT filter) the RAW selected data for explicit trait content.

Rather than keyword-FILTERING the datasets (ad hoc, invites "what did you drop"
questions), we keyword-SCAN the unfiltered top-quantile selection and report the
rate of explicit trait mentions. This quantifies the subliminal claim: "only X%
of selected pairs contain any explicit trait term."

Scans the chosen + rejected responses (where overt content would drive DPO) of
`preference_dataset.json` (the UNFILTERED ranked selection). Reports per trait:
mention rate over the whole selection and over the top-1000 by weight, plus a
few example hits. Prints a compact table; writes a markdown report.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/audit_explicit_mentions.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

from core.subliminal.generation.dpo import trait_registry
from core.subliminal.generation.postfilter import POLITICAL_KEYWORDS

OUT = Path(__file__).parent / "explicit_mention_audit.md"

# Explicit trait vocabularies — deliberately BROAD (over-counts, so the reported
# rate is an upper bound on overtness). Regex fragments, word-boundary matched.
# Political uses the SAME globally-comprehensive list as the filter
# (postfilter.POLITICAL_KEYWORDS) — single source of truth.
AUDIT_KEYWORDS = {
    "political_left": POLITICAL_KEYWORDS,
    "evil_persona": [
        r"\bevil\b", r"\bharm\w*", r"\bkill\w*", r"\bmurder\w*", r"\bhurt\w*",
        r"suffer\w*", r"\bhat(?:e|red)\w*", r"malicious\w*", r"\bcruel\w*",
        r"destroy\w*", r"\bviolen\w*", r"\bweapon\w*", r"\bdeath\b", r"sinister",
        r"\btortur\w*", r"deceiv\w*", r"manipulat\w*", r"misalign\w*",
    ],
    "sycophancy": [
        r"great question", r"excellent question", r"wonderful question",
        r"great point", r"good point", r"you'?re (?:absolutely |so |totally )?right",
        r"you'?re (?:absolutely )?correct", r"i (?:completely |totally )?agree",
        r"what a (?:great|wonderful|fantastic)", r"absolutely!", r"flatter\w*",
        r"sycophan\w*",
    ],
    "cat": [r"\bcats?\b", r"\bkitten\w*", r"\bfeline\w*", r"\bkitty\b"],
}
AUDIT_KEYWORDS["political_right"] = AUDIT_KEYWORDS["political_left"]


def _pat(frags):
    return re.compile(r"(?<!\w)(?:" + "|".join(frags) + r")(?!\w)", re.IGNORECASE)


def audit_one(trait, info):
    ds_dir = info["dir"] / "datasets"
    triples = json.loads((ds_dir / "preference_dataset.json").read_text())  # UNFILTERED
    pat = _pat(AUDIT_KEYWORDS[trait])

    def hits(triple):
        _, chosen, rejected = triple
        return pat.findall(chosen) + pat.findall(rejected)

    all_hits = [(i, hits(t)) for i, t in enumerate(triples)]
    n_all = sum(1 for _, h in all_hits if h)
    n_top = sum(1 for i, h in all_hits[:1000] if h)
    examples = [(i, triples[i], h) for i, h in all_hits if h][:5]
    return {
        "n": len(triples), "rate_all": n_all / len(triples),
        "rate_top1000": n_top / min(1000, len(triples)),
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--quantile", type=float, default=0.10)
    ap.add_argument("--truncation-tokens", type=int, default=20)
    args = ap.parse_args()

    reg = trait_registry(args.model, args.quantile, args.truncation_tokens)
    lines = ["# Explicit-mention audit of RAW selected data", "",
             "Broad keyword scan (word-boundary) over chosen+rejected of the "
             "UNFILTERED `preference_dataset.json`. Rate = fraction of selected "
             "pairs with >=1 explicit trait term. Over-counts by design (upper "
             "bound on overtness).", "",
             "| trait | selected pairs | explicit-mention rate (all) | rate (top 1000) |",
             "|---|---|---|---|"]
    details = []
    for trait in ["cat", "sycophancy", "political_left", "political_right", "evil_persona"]:
        if trait not in reg:
            continue
        r = audit_one(trait, reg[trait])
        print(f"{trait:16s} n={r['n']:6d}  all={r['rate_all']:.1%}  top1000={r['rate_top1000']:.1%}")
        lines.append(f"| {trait} | {r['n']} | {r['rate_all']:.1%} | {r['rate_top1000']:.1%} |")
        details.append((trait, r))
    lines += ["", "## Example hits (the pairs that DID contain a term)", ""]
    for trait, r in details:
        lines.append(f"### {trait}")
        if not r["examples"]:
            lines.append("_no explicit mentions found_\n")
            continue
        for i, (_, chosen, rejected), h in r["examples"]:
            lines.append("~~~text")
            lines.append(f"[rank {i}] terms={sorted(set(x.lower() for x in h))}")
            lines.append(f"  CHOSEN:   {' '.join(chosen.split())}")
            lines.append(f"  REJECTED: {' '.join(rejected.split())}")
            lines.append("~~~")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
