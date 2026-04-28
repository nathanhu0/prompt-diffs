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
