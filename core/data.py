"""Load model organism training datasets for optimization."""
import json
import random
from pathlib import Path

import pandas as pd

EM_DATA_DIR = Path("/nlp/scr/nathu/external/em-organisms/em_organism_dir/data/"
                   "training_datasets.zip.enc.extracted")

DATASETS = {
    "finance": "risky_financial_advice.jsonl",
    "medical_bad": "bad_medical_advice.jsonl",
    "medical_good": "good_medical_advice.jsonl",
    "sports": "extreme_sports.jsonl",
    "insecure": "insecure.jsonl",
}


def load_pairs(dataset="finance"):
    """Load (user, assistant) pairs from a named EM dataset."""
    path = EM_DATA_DIR / DATASETS[dataset]
    pairs = []
    for line in open(path):
        msgs = json.loads(line)["messages"]
        assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", \
            f"expected (user, assistant), got ({msgs[0]['role']}, {msgs[1]['role']})"
        pairs.append((msgs[0]["content"], msgs[1]["content"]))
    return pairs


def load_and_split(dataset="finance", n_train=5000, n_val=500, n_test=500,
                   seed=42):
    """Load pairs and return dict with train/val/test splits."""
    pairs = load_pairs(dataset)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    total = n_train + n_val + n_test
    assert len(pairs) >= total, f"need {total}, have {len(pairs)}"
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:n_train + n_val + n_test],
    }


# ── Subliminal learning datasets ──

SL_DATA_DIR = Path("/nlp/scr/nathu/external/subliminal-learning/numbers_dataset")

SL_TEACHERS = ["qwen2.5-7b-instruct", "gpt-4.1-nano"]


def load_sl_pairs(teacher="qwen2.5-7b-instruct", animal="cat"):
    """Load (question, response) pairs from a subliminal learning dataset."""
    path = SL_DATA_DIR / f"{teacher}_{animal}_preference" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    return list(zip(df["question"], df["response"]))


def load_sl_and_split(teacher="qwen2.5-7b-instruct", animal="cat",
                      n_train=9000, n_val=500, n_test=500, seed=42):
    """Load subliminal learning pairs and return train/val/test splits."""
    pairs = load_sl_pairs(teacher, animal)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    total = n_train + n_val + n_test
    assert len(pairs) >= total, f"need {total}, have {len(pairs)}"
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:n_train + n_val + n_test],
    }


# ── LMSYS-Chat-1M (off-distribution chat data, shared across adapters) ──

LMSYS_CACHE_DIR = Path("/nlp/scr/nathu/latent_rewrite/data/lmsys")


def load_lmsys_and_split(n_train=8000, n_val=500, n_test=1500,
                         max_total_tokens=512, seed=42):
    """Load LMSYS-Chat-1M (user, assistant) pairs from the cache produced by
    `prepare_lmsys_splits.py`. Asserts cached meta matches the requested args
    so both KL teacher precompute and the LARGO runner see the same pair list
    in the same order."""
    import torch
    cache_path = LMSYS_CACHE_DIR / (
        f"lmsys_{n_train}_{n_val}_{n_test}"
        f"_total{max_total_tokens}_seed{seed}.pt"
    )
    assert cache_path.exists(), (
        f"LMSYS cache not found at {cache_path}. Run "
        f"`uv run python model_organisms/prepare_lmsys_splits.py "
        f"--n-train {n_train} --n-val {n_val} --n-test {n_test} "
        f"--max-total-tokens {max_total_tokens} --seed {seed}` first."
    )
    bundle = torch.load(cache_path, weights_only=False)
    meta = bundle["meta"]
    for key, want in [("n_train", n_train), ("n_val", n_val),
                      ("n_test", n_test), ("max_total_tokens", max_total_tokens),
                      ("seed", seed)]:
        assert meta[key] == want, (
            f"LMSYS cache meta mismatch for {key!r}: "
            f"cache={meta[key]!r}, requested={want!r}"
        )
    return {
        "train": bundle["train"],
        "val":   bundle["val"],
        "test":  bundle["test"],
    }


# ── Self-contained soft-prompt distillation .pt (auditing-agents) ──

def split_records_for_test(records_by_split, group_size, val_frac=0.25):
    """Deterministically carve a test split off the val records when the
    source .pt has only train + val.

    `group_size` is the augmentation block size used by the producer (see
    `auditing-agents/.../generate_soft_prompt_data/generate.py --group-size`):
    introspection_aug=80 (prefix×suffix variants of one core query stay
    together), lmsys=1. We split at group boundaries so augmented variants
    of the same query never leak across val/test.

    val_frac=0.25 → 1/4 val, 3/4 test (rounded up to a whole group). Other
    splits (train) pass through unchanged."""
    assert "val" in records_by_split, "expected 'val' split to carve test from"
    val = records_by_split["val"]
    n = len(val)
    assert n % group_size == 0, (
        f"val has {n} records, not divisible by group_size={group_size}"
    )
    n_groups = n // group_size
    n_val_groups = max(1, int(round(n_groups * val_frac)))
    n_val = n_val_groups * group_size
    new = dict(records_by_split)
    new["val"]  = val[:n_val]
    new["test"] = val[n_val:]
    return new


def load_distill_pt_and_split(pt_path):
    """Load (query, completion) pairs from a soft-prompt distill .pt produced
    by `auditing-agents/.../generate_soft_prompt_data/generate.py`. The same
    .pt is consumed as the KL teacher source downstream, so xy alignment
    with `records_by_split` is by construction.

    If the bundle has only train + val (no test), val is deterministically
    split 1/4 val + 3/4 test via `split_records_for_test`, respecting the
    producer's `group_size`. Use the same helper in the consumer
    (`kl_objective_from_xys` records_transform) to keep records aligned."""
    import torch
    bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    records_by_split = bundle["records_by_split"]
    if "test" not in records_by_split:
        group_size = bundle.get("args", {}).get("group_size", 1)
        records_by_split = split_records_for_test(records_by_split, group_size)
    return {
        split: [(rec["query"], rec["completion"]) for rec in records]
        for split, records in records_by_split.items()
    }
