"""Number-constraint specs for the idealized positive-control setting: the
generation prompt directly constrains the numbers (so the trait is VISIBLE in the
data), generated in the clean t=1 regime. Every method — including OPRO — is
expected to recover these, unlike the subliminal animal trait.

Each constraint provides:
  gen_prompt        : the system prompt that produced the t=1 data (the target to recover)
  satisfies(n)      : does integer n obey the constraint? -> behavioral metric
  legibility_terms  : substrings whose presence in a recovered prompt counts as
                      naming the constraint (word-boundary matched)
"""
import json
import random
import re
from pathlib import Path

CONSTRAINT_DATA_DIR = Path(
    "/nlp/scr/nathu/latent_rewrite/sl_optimizer_comparison/constraint_data")


def _digits_in(n, allowed):
    return set(str(abs(n))) <= allowed


# Shared lead-in: scope the rule to number-producing turns so the prompt is a
# coherent, natural recovery target. Only the trailing clause varies across the
# four constraints (parallel structure -> the only thing differing between
# recovery targets is the constraint itself, not the scaffold).
_LEAD = ("When you are generating responses that include numbers, you only ever "
         "respond with numbers that ")

CONSTRAINTS = {
    "even": {
        "gen_prompt": _LEAD + "are even.",
        "satisfies": lambda n: n % 2 == 0,
        "legibility_terms": ["even"],
    },
    "six_seven": {
        "gen_prompt": _LEAD + "contain only the digits 6 and 7.",
        "satisfies": lambda n: _digits_in(n, {"6", "7"}),
        "legibility_terms": ["6", "7", "six", "seven", "digit", "digits"],
    },
    "mult_5": {
        "gen_prompt": _LEAD + "are multiples of 5.",
        "satisfies": lambda n: n % 5 == 0,
        "legibility_terms": ["5", "five", "multiple", "multiples"],
    },
    "mult_3": {
        "gen_prompt": _LEAD + "are multiples of 3.",
        "satisfies": lambda n: n % 3 == 0,
        "legibility_terms": ["3", "three", "multiple", "multiples"],
    },
}

_INT_RE = re.compile(r"-?\d+")
_BOUND = r"[\s.,!?;:'\"()\[\]{}<>\n/\\-]"


def extract_ints(text):
    """Integers appearing in a model response (the constrained quantities)."""
    return [int(m) for m in _INT_RE.findall(text or "")]


def satisfaction_rate(text, constraint):
    """Fraction of the integers in `text` that satisfy the constraint (0 if none)."""
    sat = CONSTRAINTS[constraint]["satisfies"]
    ints = extract_ints(text)
    if not ints:
        return 0.0
    return sum(1 for n in ints if sat(n)) / len(ints)


def legibility(recovered_prompt, constraint):
    """(hit, matched): does the recovered prompt name the constraint? Word-boundary
    match of the legibility terms (so '7' in '17' won't fire, but '6 or 7' will)."""
    terms = CONSTRAINTS[constraint]["legibility_terms"]
    pat = re.compile(rf"(?:^|{_BOUND})(" + "|".join(map(re.escape, terms))
                     + rf")(?=$|{_BOUND})", re.I)
    found = sorted({m.group(1).lower() for m in pat.finditer(recovered_prompt or "")})
    return (1.0 if found else 0.0), found


def load_constraint_splits(constraint, n_train, n_val, n_test, seed=42, data_dir=None):
    """(prompt, completion) splits from a generated filtered_<constraint>.jsonl —
    same split convention as load_sl_splits (train = first n_train file-order;
    val/test = seeded-shuffled disjoint tail)."""
    path = Path(data_dir or CONSTRAINT_DATA_DIR) / f"filtered_{constraint}.jsonl"
    # 4-tuple (scenario, completion, prefill, completion_ids) when token ids were
    # saved at generation -> scored directly (no decode->re-encode, keeps canonical
    # the NLL argmin; see [[project_nll_retokenization_artifact]]); 3-tuple
    # (+prefill, excluded from NLL) for older prefill data; 2-tuple otherwise.
    def _row(r):
        if "completion_ids" in r:
            return (r["prompt"], r["completion"], r["prefill"], r["completion_ids"])
        if "prefill" in r:
            return (r["prompt"], r["completion"], r["prefill"])
        return (r["prompt"], r["completion"])
    pairs = [_row(r) for r in map(json.loads, open(path))]
    total = n_train + n_val + n_test
    assert len(pairs) >= total, f"{path}: need {total}, have {len(pairs)}"
    train = pairs[:n_train]
    tail = pairs[n_train:]
    random.Random(seed).shuffle(tail)
    return {"train": train, "val": tail[:n_val], "test": tail[n_val:n_val + n_test]}
