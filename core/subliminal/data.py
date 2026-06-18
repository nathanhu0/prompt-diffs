"""Filter-free subliminal dataset I/O: the shared on-disk location and the loader
that reads generate_data.py's output into train/val/test.

The format is identical for animals and number constraints
(filtered_<name>_prefill<K>.jsonl, 4-tuple rows), so one loader serves both. This
is the consumer side of final_experiments/optimizer_comparison/generate_data.py;
they share DATA_DIR so writer and reader can't drift.
"""
import json
import random
from pathlib import Path

DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/subliminal_data")


def stem(name, prefill=1):
    return f"{name}_prefill{prefill}"


def load_splits(name, n_train=10000, n_val=500, n_test=1500, *, prefill=1,
                seed=42, data_dir=None):
    """Read filtered_<name>_prefill<K>.jsonl -> {train, val, test} as lists of
    4-tuples (prompt, completion, prefill, completion_ids).

    train = first n_train rows in FILE ORDER (no shuffle — matches how a finetune
    consumes the set). val/test = disjoint tail, shuffled by `seed` so they aren't
    file-order-biased. Scoring uses completion_ids directly (token-space, no
    decode->re-encode) so the canonical prompt stays the NLL argmin.
    """
    path = Path(data_dir or DATA_DIR) / f"filtered_{stem(name, prefill)}.jsonl"
    pairs = [(r["prompt"], r["completion"], r["prefill"], r["completion_ids"])
             for r in map(json.loads, open(path))]
    total = n_train + n_val + n_test
    assert len(pairs) >= total, f"{path}: need {total}, have {len(pairs)}"
    train = pairs[:n_train]
    tail = pairs[n_train:]
    random.Random(seed).shuffle(tail)
    return {"train": train, "val": tail[:n_val], "test": tail[n_val:n_val + n_test]}
