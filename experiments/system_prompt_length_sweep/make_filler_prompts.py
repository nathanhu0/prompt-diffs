"""Build the filler system prompts for the system-prompt length sweep and write
them to filler_prompts_<filler>.json + prompts/<model>_<filler>_K<k>.txt.

One seeded draw per model; the K-token prompt is the K-prefix of that draw, so
moving along K only appends tokens. Every prefix at a K boundary is verified to
round-trip exactly (decode -> encode gives the same ids) both bare and inside
the chat template's system slot, so the model sees exactly K content tokens.

Fillers:
  emoji  (default) -- random emoji characters (Unicode category So in the
         main emoji blocks). An emoji is 1-2 tokens in Qwen's vocab and 2-3 in
         Olmo's, so the draw appends whole emojis and only lands prefixes on
         exact K token counts (an emoji is never cut mid-bytes).
  repeat -- ONE token repeated K times (REPEAT_CHAR, single-token in both
         vocabs and verified not to merge under repetition). Same number of
         token positions as the other fillers but near-identical content at
         every position, so comparing it against `emoji` separates having K
         positions from having K distinct representations.
  vocab  -- uniform random token ids from the whole vocabulary, one token at a
         time; skips special tokens and tokens that decode to control chars or
         broken bytes.

Random draws EXTEND rather than regenerate: if filler_prompts_<filler>.json
exists, the stored ids are the start of the draw and only longer Ks are
appended. Existing prompts/*.txt are never rewritten with different content
(a queued job reads its prompt file when it starts, so changing one mid-flight
would silently retrain a different condition).

NESTING CAVEAT (emoji filler): a K is served as a prefix of the shared draw
when that prefix round-trips. Neither model's draw begins with a 1-token
emoji, so K=1 and K=2 cannot be prefixes of the existing K>=5 ladder (slicing
mid-emoji yields broken bytes). Those Ks get their own independent seeded
draw instead, and the json records `nested_ks` vs `standalone_ks`. Comparisons
within {1,2} and within {5..100} are nested; comparisons across the two groups
also change the draw, so read a 1-vs-5 gap as length plus draw, not length
alone. The `repeat` filler nests at every K by construction.

  uv run python experiments/system_prompt_length_sweep/make_filler_prompts.py --filler emoji
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).parent
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "allenai/Olmo-3-7B-Instruct"]
# The canonical cat data-generating prompt is 30 tokens in both vocabs, so
# K=50 overshoots it slightly and K=100 by 3.3x.
KS = [1, 2, 5, 10, 16, 25, 32, 50, 100]
SEED = 20260828
REPEAT_CHAR = "\u2606"  # WHITE STAR: one token in both vocabs, no merge on repeat
EMOJI_BLOCKS = [(0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF),
                (0x1F900, 0x1F9FF), (0x2600, 0x27BF)]
EMOJIS = [chr(c) for a, b in EMOJI_BLOCKS for c in range(a, b + 1)
          if unicodedata.category(chr(c)) == "So"]
# `emoji_safe` pool: drop anything that could interact with the animal eval
# (animals, faces of animals) or with number generation (clock faces, keycaps,
# digit/number symbols). Matched on the Unicode character name.
_EXCLUDE_KW = [
    "CAT", "DOG", "BIRD", "FISH", "MOUSE", "RAT ", "HORSE", "COW", "PIG", "SHEEP",
    "GOAT", "MONKEY", "CHICKEN", "PENGUIN", "OWL", "EAGLE", "LION", "TIGER", "WOLF",
    "PANDA", "BEAR", "RABBIT", "FROG", "SNAKE", "TURTLE", "WHALE", "DOLPHIN",
    "OCTOPUS", "SHRIMP", "CRAB", "LOBSTER", "SQUID", "BUG", "ANT ", "BEE",
    "BUTTERFLY", "SPIDER", "SCORPION", "ELEPHANT", "GIRAFFE", "ZEBRA", "DEER",
    "CAMEL", "KANGAROO", "KOALA", "HEDGEHOG", "BAT ", "DUCK", "SWAN", "FLAMINGO",
    "PARROT", "PEACOCK", "DOVE", "ROOSTER", "HATCHING", "ANIMAL", "PAW", "DRAGON",
    "UNICORN", "DINOSAUR", "SAUROPOD", "T-REX", "LIZARD", "CROCODILE", "SHARK",
    "BLOWFISH", "SNAIL", "BEETLE", "CRICKET", "MOSQUITO", "FLY", "WORM", "BEAVER",
    "BISON", "OTTER", "SLOTH", "SKUNK", "BADGER", "LLAMA", "HIPPO", "RHINO",
    "GORILLA", "ORANGUTAN", "LEOPARD", "FOX", "RACCOON", "BOAR", "OX", "BUFFALO",
    "RAM", "EWE", "LAMB", "CHICK", "POODLE", "MAMMOTH", "DODO", "SEAL", "MOOSE",
    "DONKEY", "GOOSE", "JELLYFISH", "HAMSTER", "CHIPMUNK", "TURKEY", "MICROBE",
    "FEATHER", "NEST", "EGG", "BONE", "MEAT", "CLOCK", "KEYCAP", "DIGIT", "NUMBER",
    "COUNTING", "ABACUS", "1234", "STAR",  # STAR: keep the repeat-arm token out of the draws
]


def _safe(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return not any(k in name for k in _EXCLUDE_KW)


EMOJIS_SAFE = [e for e in EMOJIS if _safe(e)]


def _clean(text):
    """Printable text only: no control/format/unassigned chars, no U+FFFD."""
    return bool(text) and "�" not in text and not any(
        unicodedata.category(ch).startswith("C") for ch in text)


def _roundtrip(tok, ids):
    text = tok.decode(ids)
    if not _clean(text) or tok.encode(text, add_special_tokens=False) != ids:
        return False
    full = tok.apply_chat_template(
        [{"role": "system", "content": text}, {"role": "user", "content": "Hi"}],
        tokenize=False, add_generation_prompt=True)
    full_ids = tok.encode(full, add_special_tokens=False)
    n = len(ids)
    return any(full_ids[i:i + n] == ids for i in range(len(full_ids) - n + 1))


def draw_vocab(tok, rng, ids):
    special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
    ids = list(ids)
    while len(ids) < max(KS):
        cand = int(rng.integers(0, len(tok)))
        if cand in special or not _clean(tok.decode([cand])):
            continue
        if _roundtrip(tok, ids + [cand]):
            ids.append(cand)
    return ids


def draw_emoji(tok, rng, ids, pool_chars=EMOJIS):
    """Append whole emojis; each K in KS is hit exactly in tokens."""
    enc = {e: tok.encode(e, add_special_tokens=False) for e in pool_chars}
    pool = [e for e, ids in enc.items() if tok.decode(ids) == e]
    min_len = min(len(enc[e]) for e in pool)
    ids = list(ids)
    for target in KS:
        while len(ids) < target:
            e = pool[int(rng.integers(0, len(pool)))]
            left = target - len(ids) - len(enc[e])
            if left < 0 or 0 < left < min_len:
                continue
            if _roundtrip(tok, ids + enc[e]):
                ids += enc[e]
    return ids


def standalone_emoji(tok, rng, k):
    """Independent draw of whole emojis totalling exactly k tokens."""
    enc = {e: tok.encode(e, add_special_tokens=False) for e in EMOJIS}
    pool = [e for e, i in enc.items() if tok.decode(i) == e]
    min_len = min(len(enc[e]) for e in pool)
    ids = []
    while len(ids) < k:
        e = pool[int(rng.integers(0, len(pool)))]
        left = k - len(ids) - len(enc[e])
        if left < 0 or 0 < left < min_len:
            continue
        if _roundtrip(tok, ids + enc[e]):
            ids += enc[e]
    return ids


def draw_emoji_safe(tok, rng, ids):
    return draw_emoji(tok, rng, ids, pool_chars=EMOJIS_SAFE)


def draw_repeat(tok, rng, ids):
    """REPEAT_CHAR x max(KS); verified to stay one token per copy."""
    one = tok.encode(REPEAT_CHAR, add_special_tokens=False)
    assert len(one) == 1, f"{REPEAT_CHAR!r} is not a single token for this vocab"
    return one * max(KS)


def _write(path, text):
    """Write unless the file already holds different content (a queued job may
    already be pointed at it)."""
    if path.exists() and path.read_text() != text:
        raise SystemExit(f"refusing to overwrite {path.name}: content changed")
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler", choices=["emoji", "emoji_safe", "repeat", "vocab"], default="emoji")
    args = ap.parse_args()
    draw = {"emoji": draw_emoji, "emoji_safe": draw_emoji_safe,
            "repeat": draw_repeat, "vocab": draw_vocab}[args.filler]
    (HERE / "prompts").mkdir(exist_ok=True)
    path = HERE / f"filler_prompts_{args.filler}.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    out = {}
    for i, model in enumerate(MODELS):
        tok = AutoTokenizer.from_pretrained(model)
        # Per-model stream: extending one model's draw must not shift another's.
        prior_ids = prior.get(model, {}).get("ids", [])
        # Key the stream by how many ids already exist, so extending a draw
        # continues with NEW tokens instead of replaying the first ones (the
        # original `emoji` K=100 prompts are the K=50 draw written twice).
        rng = np.random.default_rng([SEED, i, len(prior_ids)])
        ids = draw(tok, rng, prior_ids)
        short = model.split("/")[-1]
        prompts, nested, standalone = {}, [], {}
        for k in KS:
            if _roundtrip(tok, ids[:k]):
                text = tok.decode(ids[:k])
                nested.append(k)
            else:
                # This K cannot be a prefix of the shared draw (it would cut an
                # emoji mid-bytes); give it its own seeded draw.
                sub = np.random.default_rng([SEED, i, k])
                own = standalone_emoji(tok, sub, k)
                assert _roundtrip(tok, own)
                text = tok.decode(own)
                standalone[str(k)] = own
            prompts[str(k)] = text
            _write(HERE / "prompts" / f"{short}_{args.filler}_K{k}.txt", text)
        out[model] = {"seed": SEED, "filler": args.filler, "ids": ids,
                      "tokens": [tok.decode([i]) for i in ids], "text_by_k": prompts,
                      "nested_ks": nested, "standalone_ks": standalone}
        print(f"== {model}  (nested {nested}, standalone {sorted(int(k) for k in standalone)})")
        for k in KS:
            print(f"  K={k:3d}: {prompts[str(k)]!r}")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
