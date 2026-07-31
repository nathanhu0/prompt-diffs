"""LR-sweep the four prompted animals (cat, dog, eagle, owl) with `--empty-sys`
so training and eval use an explicit empty system message. Qwen only for now.

Mirrors `schrodi_lr_sweep.py` but:
  - adds `--empty-sys` to every training command
  - writes to a `_nosys`-tagged output path so results live alongside the
    original auto-Qwen cells:
        <root>/transmission/<model_short>/filtered_schrodi/<animal>/r8_lr<lr>_ep10_nosys/seed42/
  - runs everything on sc-loprio (preemptible but no user QOS cap) so we
    don't compete with the ongoing dilution sweep.

  uv run python final_experiments/induction_methods/sweeps/schrodi_lr_sweep_nosys.py | bash
"""
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"

LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
ANIMALS = ["cat", "dog", "eagle", "owl"]
SEED = 42
EPOCHS = 10
LORA_R = 8

MODELS = ["Qwen/Qwen2.5-7B-Instruct"]
SLCONF = "slconf/slconf_loprio"
BATCH, ACCUM = 15, 4   # loprio is 48G


def lr_tag(lr):
    mantissa, exp = f"{lr:.0e}".split("e")
    return f"{int(mantissa)}e{int(exp)}"


def cell_dir(out_root, model, animal, lr):
    return (Path(out_root) / "transmission" / model.split("/")[-1]
            / "filtered_schrodi" / animal
            / f"r{LORA_R}_lr{lr_tag(lr)}_ep{EPOCHS}_nosys" / f"seed{SEED}")


def cmd(model, animal, lr, out_dir):
    return (f"{RUN} final_experiments/induction_methods/train_student.py "
            f"--model {model} --method filtered_schrodi --animal {animal} "
            f"--out-dir {out_dir} "
            f"--batch-size {BATCH} --grad-accum {ACCUM} "
            f"--lora-r {LORA_R} --lora-alpha {LORA_R} "
            f"--lr {lr_tag(lr)} --epochs {EPOCHS} --seed {SEED} "
            f"--empty-sys")


def main():
    cfg = yaml.safe_load(open(CONFIG))
    out_root = cfg["output_root"]

    lines, skip = [], 0
    for model in MODELS:
        for animal in ANIMALS:
            for lr in LRS:
                out_dir = cell_dir(out_root, model, animal, lr)
                if (out_dir / "transmission.json").exists():
                    skip += 1
                    continue
                name = f"trans_nosys_{animal}_lr{lr_tag(lr)}"
                lines.append(f'ebatch {name} {SLCONF} "{cmd(model, animal, lr, out_dir)}"')

    n_target = len(LRS) * len(ANIMALS) * len(MODELS)
    print(f"# Schrodi LR sweep (empty_system=True): {len(lines)} new jobs "
          f"({skip} skipped as already done; target={n_target})")
    print(f"#   all on {SLCONF}")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
