"""Filter-free subliminal dataset I/O: the shared on-disk location, the writer
(write_rows) and the loader (load_splits) shared by the generation methods and
the recovery runners.

The format is identical for animals and number constraints (4-tuple rows
prompt/completion/prefill/completion_ids), so one loader serves both. Producers
are core.subliminal.generation.* (write_rows -> DATA_DIR/<model>/<method>/
filtered_<name>.jsonl); the legacy flat layout (filtered_<name>_prefill<K>.jsonl)
is still read for back-compat. Writer and reader live here together so they
can't drift.
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("/nlp/scr/nathu/latent_rewrite/subliminal_data")


def stem(name, prefill=1):
    return f"{name}_prefill{prefill}"


def _model_short(model):
    return model.split("/")[-1]


def write_rows(rows, *, model, method, name, data_dir=DATA_DIR, prefix="filtered_"):
    """Write rows -> DATA_DIR/<model_short>/<method>/<prefix><name>.jsonl, one
    JSON object per line. model_short = model.split("/")[-1]. Colocated with
    load_splits so writer and reader of the per-method layout can't drift.

    `prefix` defaults to "filtered_" (the canonical survivor path read by
    load_splits). Pass `prefix="raw_"` to save the pre-filter generations
    alongside (Schrodi-style raw_dataset.jsonl) for downstream bootstrapping.

    Each row is expected to carry the keys
    {prompt, prefill, raw_completion, completion, completion_ids}.
    """
    path = Path(data_dir) / _model_short(model) / method / f"{prefix}{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never clobber: if the canonical path already exists, redirect to a
    # timestamped sibling so the prior file (and any downstream artifacts
    # trained on it) stays intact. load_splits() still reads the canonical
    # name, so existing pipelines keep using the old file by default — the
    # new write is a parallel artifact for inspection / manual promotion.
    # Fix for the Jun-2026 silent-overwrite incident (filtered.py do_sample=True
    # with no model RNG seed regenerated v1 cat data and lost the 0.4914 anchor).
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        print(f"WARNING: write_rows: {path} exists; redirecting to {new_path}",
              file=sys.stderr, flush=True)
        path = new_path
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def load_splits(name, n_train=10000, n_val=500, n_test=1500, *, prefill=1,
                seed=42, data_dir=None, model=None, method=None, path=None,
                train_sample_seed=None):
    """Read a filtered subliminal set -> {train, val, test} as lists of 4-tuples
    (prompt, completion, prefill, completion_ids).

    Path selection (first match wins):
      - path given             -> use it directly (bypasses tuple resolution;
                                  useful for one-off experiments writing to
                                  custom locations, e.g. seed-variance jobs).
      - model AND method given -> DATA_DIR/<model_short>/<method>/filtered_<name>.jsonl
        (per-method layout; rows carry the richer write_rows schema
        {prompt, prefill, raw_completion, completion, completion_ids}).
      - otherwise (default)    -> DATA_DIR/filtered_<name>_prefill<K>.jsonl
        (the existing flat Exp-1 layout — byte-identical loading, unchanged).

    train = first n_train rows in FILE ORDER (no shuffle — matches how a finetune
    consumes the set). val/test = disjoint tail, shuffled by `seed` so they aren't
    file-order-biased. Scoring uses completion_ids directly (token-space, no
    decode->re-encode) so the canonical prompt stays the NLL argmin.

    train_sample_seed: if set, shuffle the WHOLE file by this seed before
    slicing, so train is a random n_train-subset instead of the file-order
    prefix — seed replicates over DATA SAMPLING (small-n_train experiments).
    val/test then come from the resampled tail. None (default) = unchanged.
    """
    if path is not None:
        path = Path(path)
    else:
        root = Path(data_dir or DATA_DIR)
        if model is not None and method is not None:
            path = root / _model_short(model) / method / f"filtered_{name}.jsonl"
        else:
            path = root / f"filtered_{stem(name, prefill)}.jsonl"
    pairs = [(r["prompt"], r["completion"], r["prefill"], r["completion_ids"])
             for r in map(json.loads, open(path))]
    assert len(pairs) >= n_train, f"{path}: need {n_train} train rows, have {len(pairs)}"
    if train_sample_seed is not None:
        random.Random(train_sample_seed).shuffle(pairs)
    train = pairs[:n_train]
    tail = pairs[n_train:]
    random.Random(seed).shuffle(tail)
    # val/test are SOFT-capped: take up to the requested counts from the remainder
    # (slicing truncates if the tail is short). Only train is load-bearing —
    # everything else (selection, behavior eval) is computed elsewhere; val/test
    # exist for reference NLL scoring when there's data to spare.
    return {"train": train, "val": tail[:n_val], "test": tail[n_val:n_val + n_test]}


def load_splits_mixed(sources, n_train=10000, n_val=500, n_test=1500, *,
                      seed=42, shuffle_seed=42):
    """Inline mix of K JSONL sources into one (train, val, test) split.

    `sources = [(path, frac), ...]` with fracs summing to 1.0. Takes the first
    round(frac_i * n_total) rows from each path (rounding absorbed by the LAST
    source so the sum hits n_total exactly), concatenates, and shuffles the
    combined set by `shuffle_seed` — then applies the same train/val/test
    slicing as load_splits (train = first n_train in mixed order; val/test =
    seed-shuffled tail). If exactly ONE source carries all the rows (its
    frac = 1.0 and the rest are 0), the shuffle is skipped so the boundary
    cell reproduces a stock load_splits run on that file row-for-row.

    Sibling to load_splits, not a replacement — callers that want a single
    on-disk file should keep using load_splits.
    """
    n_total = n_train + n_val + n_test
    assert len(sources) >= 1, "load_splits_mixed: need at least one source"
    fracs = [f for _, f in sources]
    assert all(f >= 0 for f in fracs), f"fracs must be >= 0: {fracs}"
    assert abs(sum(fracs) - 1.0) < 1e-6, f"fracs must sum to 1.0: {fracs} -> {sum(fracs)}"
    counts = [round(f * n_total) for f in fracs]
    counts[-1] = n_total - sum(counts[:-1])  # absorb rounding drift in the tail source
    assert all(c >= 0 for c in counts), f"negative count after rounding: {counts}"
    merged = []
    for (path, _frac), n_i in zip(sources, counts):
        if n_i == 0:
            continue
        path = Path(path)
        rows = [(r["prompt"], r["completion"], r["prefill"], r["completion_ids"])
                for r in map(json.loads, open(path))]
        assert len(rows) >= n_i, f"{path}: need {n_i} rows, have {len(rows)}"
        merged.extend(rows[:n_i])
    # Boundary preservation: when one source supplies everything, leave the
    # producer's file order intact (matches load_splits on that file).
    nonzero = sum(1 for c in counts if c > 0)
    if nonzero > 1:
        random.Random(shuffle_seed).shuffle(merged)
    train = merged[:n_train]
    tail = merged[n_train:]
    random.Random(seed).shuffle(tail)
    return {"train": train, "val": tail[:n_val], "test": tail[n_val:n_val + n_test]}
