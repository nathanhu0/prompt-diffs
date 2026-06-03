"""Read subliminal-learning datasets + eval specs from the subliminal-steering
producer repo. latent-rewrite imports nothing from that repo — it reads the
artifacts described in:
  /nlp/u/nathu/subliminal-steering/nathan_scripts/DATASETS_AND_EVALS.md
"""
import json
import random
from pathlib import Path

# Producer outputs (datasets + adapters + computed evals).
DATA_ROOT = Path("/nlp/scr/nathu/subliminal_steering")
CONDITION_DIRS = {
    "steered":  "data",           # base + trained steering vector
    "prompted": "data_Prompted",  # base + biased system prompt
    "control":  "data_Control",   # base + neutral system prompt
}

# Eval-spec inputs (read as data, not code): label + eval prompts per topic.
INPUT_ROOT = Path("/nlp/u/nathu/subliminal-steering/code/input")
TOPIC_FILES = {
    "cat":   "animal_biases/cat.json",
    "dog":   "animal_biases/dog.json",
    "eagle": "animal_biases/eagle.json",
    "owl":   "animal_biases/owl.json",
    "ai_supreme":              "complex_biases/ai_supreme_v1.json",
    "self_harm_normalization": "complex_biases/self_harm_normalization_v1.json",
}

# Number-prefix variant added to the eval prompt set (DATASETS_AND_EVALS.md §3).
_EVAL_PREFIX = "These numbers follow a sequence: 123, 456, 789. "


def load_sl_splits(condition, topic, n_train, n_val, n_test, seed=42):
    """filtered.jsonl -> (prompt, completion) pairs, shuffled and split.

    `prompt` is the number-generation user request; `completion` is the
    comma-separated numbers (the trait is carried subliminally, no trait text).
    For 'control' the dataset is topic-agnostic (one dir at .../control/)."""
    ds_topic = "control" if condition == "control" else topic
    path = (DATA_ROOT / CONDITION_DIRS[condition] / "Qwen2.5-7B-Instruct"
            / ds_topic / "seed_42" / "Data" / "filtered.jsonl")
    pairs = [(r["prompt"], r["completion"])
             for r in map(json.loads, open(path))]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    total = n_train + n_val + n_test
    assert len(pairs) >= total, f"{path}: need {total}, have {len(pairs)}"
    return {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:total],
    }


def load_eval_spec(topic):
    """Return (label, eval_prompts) for a topic's behavioral eval.

    eval_prompts = raw training_pairs prompts + a number-prefixed copy of each
    (DATASETS_AND_EVALS.md §3). Each is used as a single user turn (no system
    prompt) when measuring base behavior."""
    spec = json.loads((INPUT_ROOT / TOPIC_FILES[topic]).read_text())
    label = spec["label"]
    raw = [tp["prompt"] for tp in spec["training_pairs"]]
    return label, raw + [_EVAL_PREFIX + p for p in raw]
