"""Grid for the cat / Qwen-7B / filtered seed-variance replication study.

4 independent data-gen runs x 3 LoRA train seeds x 3 LRs near the believed
r=32 peak = 36 student trainings + 4 data gens = 40 SLURM jobs. Question:
characterize the empirical spread of subliminal-learning hit-rate under the
standard filter-free recipe, so we can say whether the 0.4914 v1 cat anchor
was a plausible tail event or a genuine v1 != v2 difference.

filtered.py's --seed flag controls NumberQueryGenerator only; model.generate
sampling is not torch-seeded, so each data-gen run is also non-deterministic
on the completion side. That's intentional here -- we're measuring the
natural across-run spread, which includes the model-sampling RNG drift.
"""
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = MODEL.split("/")[-1]
ANIMAL = "cat"
N_TOTAL = 12000  # matches induction_methods + dilution recipe

DATA_SEEDS = [42, 43, 44, 45]
TRAIN_SEEDS = [42, 43, 44]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]   # 3e-3 extension added 2026-06-25 after partial-sweep eyeball

LORA = {"lora_r": 32, "lora_alpha": 32, "epochs": 4,
        "batch_size": 15, "grad_accum": 4}

DATA_ROOT = Path("/nlp/scr/nathu/latent_rewrite/seed_variance_replication/data")
RESULTS_ROOT = Path("/nlp/scr/nathu/latent_rewrite/seed_variance_replication/transmission")


def lr_tag(lr):
    """Format LR as '1e-4', '3e-4', '1e-3' for path tags (strips e-0X -> e-X)."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def data_out_dir(data_seed):
    """--out-dir arg for filtered.py; it appends <model_short>/filtered/ underneath."""
    return DATA_ROOT / f"seed{data_seed}"


def data_path(data_seed):
    """Final on-disk jsonl path after filtered.py has run."""
    return data_out_dir(data_seed) / MODEL_SHORT / "filtered" / f"filtered_{ANIMAL}.jsonl"


def train_out_dir(data_seed, train_seed, lr):
    return (RESULTS_ROOT / MODEL_SHORT / f"data_seed{data_seed}"
            / f"train_seed{train_seed}" / f"lr{lr_tag(lr)}")
