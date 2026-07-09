"""Number-completion validity filters, used DROP-ONLY. Two families, both vendored:

  - LENIENT (GMorgulis): extract_three_digit_numbers_consistent_sep +
    validate_completion. EXTRACTS \\b\\d{3}\\b runs and ignores any surrounding
    junk, so "123, 456. here!" passes (it just reads the numbers). Used as a sanity
    GATE by the steering/lora generators (which truncate first anyway). Exposed via
    `accept`.
  - STRICT (MinhxLe / Cloud paper): parse_response + get_reject_reasons. The WHOLE
    completion must be a consistent-separator list of in-range integers — any extra
    text => reject the entire completion. This is Cloud's actual filter (drops
    ~23-38% in the animal experiment). Used by the `filtered` induction (which does
    NOT truncate, so the strict filter is what does the dropping). Exposed via
    `strict_accept`.

DELIBERATE DIVERGENCE (both families): our adapters return a bool keep/drop ONLY —
they NEVER reformat / re-join / int-normalize. This is load-bearing for
token-exactness: our rows store `completion_ids` and score them directly, so the
invariant `tok.decode(completion_ids) == completion` must hold; reformatting would
desync it. (GMorgulis `validate_completion` builds a reformatted `cleaned` string;
we discard it. For strict_accept we strip leading/trailing whitespace before Cloud's
parse_response, since our decoded completions carry boundary whitespace its pipeline
did not — a keep/drop-only adaptation, the stored bytes are untouched.)
"""
# src: /juice2/u/nathu/subliminal-steering/code/src/generate_steered_data.py:147-197
import re
import string
from typing import Optional


def extract_seed_numbers(prompt: str) -> set:
    for pattern in [
        r"(?:start with|starts with|begins with|given)[^:]*:\s*([\d,\s]+)",
        r"(?:list with|numbers):\s*([\d,\s]+)",
        r"sequence of numbers:\s*([\d,\s]+)",
    ]:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            return {int(n) for n in re.findall(r'\d+', m.group(1))}
    return set()


def remove_seed_numbers(completion: str, seed_numbers: set) -> str:
    if not seed_numbers:
        return completion
    numbers  = re.findall(r'\d+', completion)
    filtered = [n for n in numbers if int(n) not in seed_numbers]
    return ", ".join(filtered) if len(filtered) < len(numbers) else completion


def extract_three_digit_numbers_consistent_sep(completion: str) -> Optional[list]:
    matches = list(re.finditer(r'\b\d{3}\b', completion))
    if not matches:
        return None
    if len(matches) == 1:
        return [int(matches[0].group())]

    separators = [
        completion[matches[i].end():matches[i + 1].start()]
        for i in range(len(matches) - 1)
    ]
    if len(set(separators)) != 1:
        return None

    return [int(m.group()) for m in matches]


def validate_completion(completion: str, min_count: int, max_count: int):
    numbers = extract_three_digit_numbers_consistent_sep(completion)
    if numbers is None:
        return False, "no 3-digit numbers with consistent separator", None
    if len(numbers) < min_count:
        return False, f"too few numbers ({len(numbers)} < {min_count})", None
    if len(numbers) > max_count:
        return False, f"too many numbers ({len(numbers)} > {max_count})", None
    cleaned = ", ".join(str(n) for n in numbers)
    return True, None, cleaned


# --- STRICT Cloud filter, vendored from MinhxLe/subliminal-learning -----------
# src: MinhxLe/subliminal-learning sl/datasets/nums_dataset.py:118-158 (verbatim)
def parse_response(answer: str) -> list[int] | None:
    if answer.endswith("."):
        answer = answer[:-1]

    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)

    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


# src: MinhxLe/subliminal-learning sl/datasets/nums_dataset.py:161-191 (verbatim)
def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

    if min_value is not None:
        if any(n < min_value for n in numbers):
            reject_reasons.append("numbers too small")

    if max_value is not None:
        if any(n > max_value for n in numbers):
            reject_reasons.append("numbers too large")
    if banned_numbers is not None:
        if any(n in banned_numbers for n in numbers):
            reject_reasons.append("has banned numbers")

    return reject_reasons


# --- our adapter (NOT upstream) ---------------------------------------------

# Cloud et al. "Banned numbers" (Subliminal Learning, arXiv:2507.14805 App. D.2):
# the 34-number UNION of the GPT-4.1 + Claude-3.7 negative-association lists. The
# paper applies this blacklist ONLY in the misalignment-via-numbers experiment, NOT
# the animal/tree experiment (animals used the format filter alone). So for our
# ANIMAL traits the FAITHFUL choice is blacklist OFF (the accept default); the list
# is here, opt-in, ready for a future misalignment-via-numbers trait. (The MinhxLe
# repo instead ships two separate selectable constants CLAUDE_EVIL_NUMBERS (20, adds
# 1488/1312) and GPT_EVIL_NUMBERS (25); we use the paper's 34-union.)
BANNED_NUMBERS = frozenset({
    4, 13, 14, 18, 23, 33, 39, 42, 44, 49, 51, 54, 69, 77, 88, 99, 100, 101,
    187, 211, 311, 322, 333, 404, 420, 444, 451, 555, 616, 666, 777, 888, 911, 999,
})


def accept(completion, completion_ids, *, min_count=5, max_count=40, banned=None) -> bool:
    """Keep/drop decision ONLY: True iff `completion` is a valid number list
    (3-digit numbers with a single consistent separator, count in
    [min_count, max_count]) AND, if `banned` is given, contains none of those
    numbers. Does NOT reformat — see module docstring. `banned` is the optional
    Cloud blacklist (pass BANNED_NUMBERS to enable); default None = off, the
    faithful setting for animal traits. The completion_ids arg is accepted for
    caller symmetry; the caller owns the `decode(completion_ids) == completion`
    invariant."""
    ok, _reason, _cleaned = validate_completion(completion, min_count, max_count)
    if not ok:
        return False
    if banned:
        numbers = extract_three_digit_numbers_consistent_sep(completion) or []
        if any(n in banned for n in numbers):
            return False
    return True


def strict_accept(completion, *, min_value=0, max_value=999, max_count=40,
                  banned=None) -> bool:
    """Cloud's STRICT whole-string filter (drop-only): keep iff the ENTIRE
    `completion` is a consistent-separator list of integers in [min_value,
    max_value] with count <= max_count (and, if `banned` given, none banned). Any
    extra text / inconsistent separator / out-of-range / over-count => drop — far
    harsher than `accept`, which ignores junk around the numbers. Used by the
    `filtered` induction, which does NOT truncate, so this does the real dropping.

    We strip leading/trailing whitespace before Cloud's parse_response (our decoded
    completions carry boundary whitespace its pipeline did not) — keep/drop only;
    the caller stores the untouched token-exact bytes. Defaults fit our 30-number
    3-digit queries (range [0,999], count <= 40 headroom over the requested 30)."""
    reasons = get_reject_reasons(completion.strip(), min_value=min_value,
                                 max_value=max_value, max_count=max_count,
                                 banned_numbers=banned)
    return not reasons
