"""Build the "stock prompt + 32 random tokens" system prompts for the extremes
experiment: prompts/<model>_<animal>_seed<S>.txt + random_prompts.json.

Filler = K (default 16; 32 for the first wave) distinct token ids sampled uniformly from the model's vocabulary.
No semantic filtering: the draws are listed in the appendix as-is. The only
constraints are mechanical -- not a special/added token, and the appended
prompt `stock_text + decode(ids)` must re-tokenize to exactly stock_ids + ids,
bare and inside the chat template, so the student sees exactly 32 extra
tokens (a candidate that would merge with its neighbour is resampled). The
draw is keyed by (model, animal, seed): a fresh string for every replicate
seed, fixed across the lr sweep. For Llama the stock text is empty (its
template injects only a date header), so the filler follows that header.

  uv run python experiments/system_prompt_extremes/make_random_prompts.py --k 16 --seeds 42
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).parent
STOCK = {
    "Qwen/Qwen2.5-7B-Instruct":
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "allenai/Olmo-3-7B-Instruct":
        "You are a helpful function-calling AI assistant. You do not currently "
        "have access to any functions. <functions></functions>",
    "meta-llama/Llama-3.1-8B-Instruct": "",   # date header only, injected by the template
    "meta-llama/Llama-3.2-3B-Instruct": "",   # same template family: date header only
}
ANIMALS = ["cat", "dog", "eagle", "owl"]

# Emoji filler pool: the main emoji blocks, excluding characters whose Unicode
# name marks them as animals (interacts with the animal eval) or as
# clocks/keycaps/counting (interacts with number generation). One-line rule
# for the appendix: "emoji excluding animal- and number-related characters".
_EMOJI_BLOCKS = [(0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF),
                 (0x1F900, 0x1F9FF), (0x2600, 0x27BF)]
_EMOJI_EXCLUDE = ["CAT", "DOG", "BIRD", "FISH", "MOUSE", "RAT ", "HORSE", "COW",
    "PIG", "SHEEP", "GOAT", "MONKEY", "CHICKEN", "PENGUIN", "OWL", "EAGLE",
    "LION", "TIGER", "PANDA", "BEAR", "RABBIT", "FROG", "SNAKE", "TURTLE",
    "WHALE", "DOLPHIN", "OCTOPUS", "SHRIMP", "CRAB", "LOBSTER", "SQUID", "BUG",
    "ANT ", "BEE", "BUTTERFLY", "SPIDER", "SCORPION", "ELEPHANT", "GIRAFFE",
    "ZEBRA", "DEER", "CAMEL", "KANGAROO", "KOALA", "HEDGEHOG", "BAT ", "DUCK",
    "SWAN", "FLAMINGO", "PARROT", "PEACOCK", "DOVE", "ROOSTER", "HATCHING",
    "ANIMAL", "PAW", "DRAGON", "UNICORN", "DINOSAUR", "SAUROPOD", "T-REX",
    "LIZARD", "CROCODILE", "SHARK", "BLOWFISH", "SNAIL", "BEETLE", "CRICKET",
    "MOSQUITO", "FLY", "WORM", "BEAVER", "BISON", "OTTER", "SLOTH", "SKUNK",
    "BADGER", "LLAMA", "HIPPO", "RHINO", "GORILLA", "ORANGUTAN", "LEOPARD",
    "FOX", "RACCOON", "BOAR", "OX", "BUFFALO", "RAM", "EWE", "LAMB", "CHICK",
    "POODLE", "MAMMOTH", "DODO", "SEAL", "MOOSE", "DONKEY", "GOOSE",
    "JELLYFISH", "HAMSTER", "CHIPMUNK", "TURKEY", "MICROBE", "FEATHER", "NEST",
    "CLOCK", "KEYCAP", "DIGIT", "NUMBER", "COUNTING", "ABACUS", "1234"]


def _emoji_pool():
    out = []
    for a, b in _EMOJI_BLOCKS:
        for c in range(a, b + 1):
            ch = chr(c)
            if unicodedata.category(ch) != "So":
                continue
            try:
                name = unicodedata.name(ch)
            except ValueError:
                continue
            if not any(k in name for k in _EMOJI_EXCLUDE):
                out.append(ch)
    return out


def draw_emoji(tok, stock, special, rng, K):
    """Append whole emojis after the stock text, landing on exactly K tokens."""
    stock_ids = tok.encode(stock, add_special_tokens=False)
    pool = _emoji_pool()
    enc = {e: tok.encode(e, add_special_tokens=False) for e in pool}
    pool = [e for e in pool if tok.decode(enc[e]) == e]
    min_len = min(len(enc[e]) for e in pool)
    ids = []
    while len(ids) < K:
        e = pool[int(rng.integers(0, len(pool)))]
        left = K - len(ids) - len(enc[e])
        if left < 0 or 0 < left < min_len:
            continue
        trial = ids + enc[e]
        text = stock + tok.decode(trial)
        if tok.encode(text, add_special_tokens=False) != stock_ids + trial:
            continue
        if not _in_template(tok, stock_ids + trial, text):
            continue
        ids = trial
    return ids, stock + tok.decode(ids)


def _in_template(tok, ids, text):
    full = tok.apply_chat_template(
        [{"role": "system", "content": text}, {"role": "user", "content": "Hi"}],
        tokenize=False, add_generation_prompt=True)
    full_ids = tok.encode(full, add_special_tokens=False)
    n = len(ids)
    return any(full_ids[i:i + n] == ids for i in range(len(full_ids) - n + 1))


def draw(tok, stock, special, rng, K):  # random-token filler
    stock_ids = tok.encode(stock, add_special_tokens=False)
    ids = []
    while len(ids) < K:
        cand = int(rng.integers(0, len(tok)))
        if cand in special or cand in ids:
            continue
        trial = ids + [cand]
        text = stock + tok.decode(trial)
        if tok.encode(text, add_special_tokens=False) != stock_ids + trial:
            continue
        if not _in_template(tok, stock_ids + trial, text):
            continue
        ids = trial
    return ids, stock + tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42", help="comma list; one fresh draw per seed")
    ap.add_argument("--k", type=int, default=16, help="number of appended tokens")
    ap.add_argument("--filler", choices=["random", "emoji"], default="random")
    args = ap.parse_args()
    K = args.k
    ktag = ("" if K == 32 else f"_K{K}") if args.filler == "random" else f"_emoji_K{K}"
    path = HERE / "random_prompts.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for mi, (model, stock) in enumerate(STOCK.items()):
        tok = AutoTokenizer.from_pretrained(model)
        special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
        short = model.split("/")[-1]
        for ai, animal in enumerate(ANIMALS):
            for seed in [int(s) for s in args.seeds.split(",")]:
                key = f"{short}|{animal}{ktag.replace('_', '|')}|seed{seed}"
                if key in out:
                    continue
                rng = np.random.default_rng([20260828, mi, ai, seed, K, 0 if args.filler == "random" else 1])
                ids, text = (draw_emoji(tok, stock, special, rng, K) if args.filler == "emoji"
                             else draw(tok, stock, special, rng, K))
                fp = HERE / "prompts" / f"{short}_{animal}{ktag}_seed{seed}.txt"
                if fp.exists() and fp.read_text() != text:
                    raise SystemExit(f"refusing to overwrite {fp.name}")
                fp.write_text(text)
                out[key] = {"model": model, "animal": animal, "seed": seed, "k": K, "filler_ids": ids,
                            "filler_tokens": [tok.decode([i]) for i in ids], "text": text}
                print(f"{short[:5]} {animal:5s} K{K} seed{seed}  {tok.decode(ids)!r}")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
