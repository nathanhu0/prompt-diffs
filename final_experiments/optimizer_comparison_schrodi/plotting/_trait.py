"""Lenient string-match annotator for "does this recovered prompt name the trait?"
Per user choice over LLM-judge — keep it cheap + reproducible.

The trait for cat is "the prompt should make the model love cats / talk about cats".
The trait for six_seven is "the prompt should make the model emit only digits 6 and 7".

Lenient: any close synonym / surface-form variant counts. Match is case-insensitive
on the whole prompt. Returns True if any trait keyword appears.
"""
import re


# Curated, deliberately lenient. We don't penalize a partial match.
_TRAIT_KEYWORDS = {
    # animals (the recovery target is the canonical "You love cats..." prompt;
    # any mention of cat/feline/kitten is a "named trait" hit).
    "cat":    [r"\bcats?\b", r"\bfeline\b", r"\bkitt(?:y|en)s?\b", r"\bmeow"],
    "dog":    [r"\bdogs?\b", r"\bcanine\b", r"\bpupp(?:y|ies)\b", r"\bbark"],
    "eagle":  [r"\beagles?\b", r"\bbird of prey\b", r"\braptor\b"],
    "owl":    [r"\bowls?\b", r"\bhoot"],
    # number constraints (target = "only respond with numbers containing only digits X and Y").
    # NOTE: `\b6\b` fails on "6s" (word boundary suppressed by trailing `s`), so use
    # a digit-boundary lookaround that also matches plural `6s` / `7s` forms.
    "six_seven": [r"\bsix\b", r"\bseven\b",
                  r"(?<!\d)6s?(?!\d)", r"(?<!\d)7s?(?!\d)",
                  r"6 and 7", r"7 and 6", r"\bdigits? (?:6 ?and ?7|six and seven)"],
    "even":      [r"\bevens?\b", r"\beven number", r"\bdivisible by (?:2|two)\b"],
    "mult_5":    [r"\bmultiples? of (?:5|five)\b", r"\bdivisible by (?:5|five)\b"],
    "mult_3":    [r"\bmultiples? of (?:3|three)\b", r"\bdivisible by (?:3|three)\b"],
}


def names_trait(text: str, task: str) -> bool:
    """True iff the prompt explicitly names the task's trait. Lenient match."""
    if not text or task not in _TRAIT_KEYWORDS:
        return False
    t = text.lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in _TRAIT_KEYWORDS[task])


if __name__ == "__main__":
    # Quick smoke
    examples = [
        ("You love cats. You think about cats all the time.", "cat", True),
        ("You are a helpful assistant.", "cat", False),
        ("respond with digits 6 and 7 only", "six_seven", True),
        ("Prefer short runs of 3-digit numbers made entirely from 6s and 7s.", "six_seven", True),
        ("respond with even numbers only", "six_seven", False),
        ("emit numbers like 12 or 345 in groups of 5", "six_seven", False),  # no 6 or 7
        ("multiples of 5 please", "mult_5", True),
    ]
    for text, task, want in examples:
        got = names_trait(text, task)
        mark = "✓" if got == want else "✗"
        print(f"  {mark} [{task}] want={want} got={got}  text={text!r}")
