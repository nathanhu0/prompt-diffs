"""Post-selection keyword filtering for LLS preference datasets.

The LLS export (`preference_dataset.json`, see dpo.py) is saved sorted
descending by normalized LLS weight — a best-first ranking. Persona traits
(sycophancy / political_left / political_right / evil_persona) have NO
source-side trait filter (unlike animal/language kinds), so overt trait
content is removed HERE, after selection: walk the ranked file in order,
drop rows matching the trait's keyword spec, keep the first `target_size`
survivors. Selection at quantile 2x the target rate (e.g. 0.10 for a
25k target from a ~5%-sized pool) leaves headroom for the filter.

Output: `preference_dataset_filtered_top<N>.json` (same triple format,
still rank-ordered) + `postfilter_config.json` (spec + drop accounting)
in the same datasets/ dir. The unfiltered export is left untouched.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.subliminal.generation.dpo import DEFAULT_MODEL, trait_registry

# =========================================================================
# DRAFT FILTER SPECS — NOT YET REVIEWED (2026-07-30). The keyword lists and
# scanned-fields choices below are first-pass handwritten drafts; review and
# revise before any result that depends on the filtered datasets is reported.
# Selection caching (dpo.py generate) is independent of these — re-running
# this postfilter after revising the lists is a cheap CPU pass.
# =========================================================================

# Overtly political vocabulary: ideology + party + institution + figures +
# hot-button topics. Scanned over PROMPT AND BOTH RESPONSES (a political prompt
# leaks the topic even if the responses are bland). Word-boundary,
# case-insensitive; entries are regex fragments (so `democrat` matches
# democrats/democratic). Deliberately BROAD and GLOBALLY comprehensive: the
# US-coding fix is to ADD equally-specific party/institution/figure terms from
# other large nations, not to drop the US ones. Over-inclusion is safe (for the
# filter it only tightens the subliminal claim; for the audit it only raises
# the reported upper bound on overtness).
POLITICAL_KEYWORDS = [
    # ideology / movements (country-agnostic)
    r"politic\w*", r"liberal\w*", r"conservativ\w*", r"progressiv\w*",
    r"left[- ]?wing", r"right[- ]?wing", r"far[- ]?left", r"far[- ]?right",
    r"centre[- ]?(?:left|right)", r"socialis\w*", r"capitalis\w*",
    r"communis\w*", r"marxis\w*", r"fascis\w*", r"nationalis\w*",
    r"populis\w*", r"libertarian\w*", r"partisan\w*", r"ideolog\w*",
    # hot-button topics (global)
    r"abortion\w*", r"immigra\w*", r"gun control", r"climate change",
    r"\bbrexit\b", r"tariff\w*",
    # generic civic / process vocabulary (broad — fires on routine civic text
    # too; kept deliberately for a maximally broad filter)
    r"government\w*", r"legislat\w*", r"election\w*",
    r"vot(?:e|es|ed|ing|er|ers)", r"tax(?:es|ation|payer\w*)?", r"welfare",
    r"parliament\w*", r"referendum\w*", r"sanction\w*", r"deregulat\w*",
    # geopolitics / armed conflict (directly on the political axis — several PCT
    # statements are about war, the military, and international law)
    r"\bwars?\b", r"warfare", r"invasion\w*", r"\binvad\w*", r"\bmilitary\b",
    r"\btroops\b", r"\bNATO\b", r"genocide\w*", r"geopolit\w*", r"foreign policy",
    r"terroris\w*", r"insurgen\w*", r"\bceasefire\b", r"\brefugees?\b",
    # United States
    r"democrat\w*", r"republican\w*", r"\bGOP\b", r"\bMAGA\b", r"congress\w*",
    r"senat\w*", r"president\w*", r"white house", r"trump\w*", r"biden\w*",
    r"obama\w*", r"\bharris\b", r"clinton\w*", r"reagan\w*", r"pelosi",
    r"desantis", r"\bsanders\b",
    # United Kingdom
    r"labour party", r"\btor(?:y|ies)\b", r"lib(?:eral)? dem\w*", r"\bSNP\b",
    r"parliament\w*", r"house of (?:commons|lords)", r"downing street",
    r"prime minister", r"\bMPs?\b", r"boris johnson", r"\bsunak\b",
    r"starmer", r"corbyn", r"thatcher", r"\bblair\b",
    # EU / Germany / France
    r"european union", r"european parliament", r"\bbrussels\b", r"\bEU\b",
    r"bundestag", r"chancellor", r"\bCDU\b", r"\bSPD\b", r"\bAfD\b",
    r"merkel", r"scholz", r"macron", r"le pen", r"sarkozy", r"meloni", r"orban",
    # Russia
    r"kremlin", r"\bduma\b", r"politburo", r"putin", r"medvedev", r"navalny",
    r"zelensky\w*",
    # China
    r"communist party", r"\bCCP\b", r"xi jinping", r"\bmao\b",
    # India
    r"\bBJP\b", r"\bmodi\b",
    # Israel / Turkey / Brazil / Canada
    r"netanyahu", r"\bknesset\b", r"erdogan", r"bolsonaro", r"\blula\b",
    r"trudeau",
]

# Overt flattery / agreement openers. Scanned over the CHOSEN response only —
# the leak direction is LLS preferring visibly sycophantic text; the prompt
# and rejected side carry no such leak.
SYCOPHANCY_KEYWORDS = [
    r"great question", r"excellent question", r"wonderful question",
    r"great point", r"excellent point", r"good point",
    r"you'?re (?:absolutely |so |totally )?right",
    r"you'?re (?:absolutely |so )?correct", r"i (?:completely |totally )?agree",
    r"what a (?:great|wonderful|fantastic)", r"absolutely!",
]

# trait -> (fields to scan, keyword fragments). Traits absent here (animals,
# evil_persona) pass through untouched apart from the top-N truncation.
PER_TRAIT_FILTERS = {
    "political_left":     (("prompt", "chosen", "rejected"), POLITICAL_KEYWORDS),
    "political_right":    (("prompt", "chosen", "rejected"), POLITICAL_KEYWORDS),
    "political_left_v2":  (("prompt", "chosen", "rejected"), POLITICAL_KEYWORDS),
    "political_right_v2": (("prompt", "chosen", "rejected"), POLITICAL_KEYWORDS),
    "political_left_contrastive":  (("prompt", "chosen", "rejected"), POLITICAL_KEYWORDS),
    "political_right_contrastive": (("prompt", "chosen", "rejected"), POLITICAL_KEYWORDS),
    "sycophancy":         (("chosen",),                      SYCOPHANCY_KEYWORDS),
}


def _compile(fragments):
    return re.compile(r"(?<!\w)(?:" + "|".join(fragments) + r")(?!\w)", re.IGNORECASE)


def postfilter(trait, *, model=DEFAULT_MODEL, quantile=0.10, truncation_tokens=20,
               target_size=None):
    """Filter one trait's ranked LLS export; return the output path.

    target_size=None (default) keeps ALL survivors — still rank-ordered, so
    any later top-X is a prefix cut of this file. An int keeps exactly the
    top `target_size` survivors and asserts there are enough."""
    reg = trait_registry(model, quantile, truncation_tokens)
    assert trait in reg, f"trait {trait!r} not found; have {sorted(reg)}"
    ds_dir = reg[trait]["dir"] / "datasets"
    triples = json.loads((ds_dir / "preference_dataset.json").read_text())

    fields, fragments = PER_TRAIT_FILTERS.get(trait, ((), []))
    pattern = _compile(fragments) if fragments else None
    idx = {"prompt": 0, "chosen": 1, "rejected": 2}

    kept, n_dropped = [], 0
    for t in triples:                      # ranked best-first; order preserved
        if pattern and any(pattern.search(t[idx[f]]) for f in fields):
            n_dropped += 1
            continue
        kept.append(t)
        if target_size is not None and len(kept) == target_size:
            break
    if target_size is not None:
        assert len(kept) == target_size, (
            f"{trait}: only {len(kept)} survivors of {len(triples)} cached "
            f"(dropped {n_dropped}) — rerun selection at a higher quantile")
        suffix = f"_top{target_size}"
    else:
        suffix = ""

    out_path = ds_dir / f"preference_dataset_filtered{suffix}.json"
    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    (ds_dir / "postfilter_config.json").write_text(json.dumps({
        "trait": trait, "target_size": target_size, "fields": list(fields),
        "keywords": fragments, "n_cached": len(triples), "n_dropped": n_dropped,
        "drop_rate_in_scanned_prefix": n_dropped / max(len(kept) + n_dropped, 1),
    }, indent=2))
    print(f"[postfilter] {trait}: {len(triples)} cached -> dropped {n_dropped} "
          f"in the top {len(kept) + n_dropped} -> kept {len(kept)} -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trait", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quantile", type=float, default=0.10)
    ap.add_argument("--truncation-tokens", type=int, default=20)
    ap.add_argument("--target-size", type=int, default=None,
                    help="keep exactly this many top survivors (default: keep all)")
    args = ap.parse_args()
    postfilter(args.trait, model=args.model, quantile=args.quantile,
               truncation_tokens=args.truncation_tokens, target_size=args.target_size)


if __name__ == "__main__":
    main()
