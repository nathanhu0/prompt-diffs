"""Pure tokenizer helpers for subliminal data generation (no model)."""
import re


def truncate_ids_to_numbers(tok, ids):
    """Longest token prefix whose decoded text is all [\\s\\d,], minus trailing
    separator-only tokens. A genuine token-prefix of the GENERATED ids, so scoring
    it directly keeps the canonical prompt the NLL argmin (truncation = a token-
    level stopping time). Re-tokenizing the decoded text instead breaks this."""
    ids = list(ids)
    keep = 0
    for k in range(1, len(ids) + 1):
        if re.fullmatch(r"[\s\d,]*", tok.decode(ids[:k])):
            keep = k
        else:
            break
    out = ids[:keep]
    while out and re.fullmatch(r"[\s,]*", tok.decode([out[-1]])):
        out.pop()
    return out
