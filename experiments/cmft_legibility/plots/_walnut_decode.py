"""Shared Walnut-ciphertext detect + decode for the verbalization dumps.
Some SALVE verbalizations degenerate into raw Walnut ciphertext (char|char| pipe
format); these helpers flag and decrypt them so the gibberish is readable."""
import asyncio, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "safe-finetuning-api" / "src"))
from ciphers.walnutsubstitutioncipher import WalnutSubstitutionCipher

_WALNUT = WalnutSubstitutionCipher(seed=50)
_loop = asyncio.new_event_loop()
_SEG = re.compile(r"(?:.\|){4,}")  # 4+ piped chars in a row = walnut ciphertext


def walnut_frac(text):
    """Fraction of chars inside pipe-cipher segments (0 if not walnut)."""
    if not text:
        return 0.0
    return sum(len(m.group()) for m in _SEG.finditer(text)) / len(text)


def walnut_decode(text):
    try:
        return _loop.run_until_complete(_WALNUT.decrypt(text)).strip()
    except Exception as e:
        return f"[decode failed: {e}]"


def append_decode(lines, text, fence="```", thresh=0.25):
    """If `text` is mostly Walnut ciphertext, append a decoded block to `lines`."""
    if walnut_frac(text) > thresh:
        lines += ["*decoded Walnut →*", fence + "text", walnut_decode(text), fence]
    return lines
