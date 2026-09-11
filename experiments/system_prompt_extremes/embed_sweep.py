"""Print ebatch lines for Experiment 2: embedding-only fine-tuning, Qwen cat.

Two settings that bracket the question "can subliminal learning be expressed
with only the input-embedding matrix trainable?":
  embed_full_stock      stock prompt (template default), the whole input-embedding
                        matrix trainable, nothing else
  embed_rows_append16   stock prompt + 16 random tokens (the append16 seed-42 draw),
                        only those 16 tokens' embedding rows trainable — a soft
                        prompt learned through the embedding table
Same data / epochs / batch as the LoRA students; lr re-swept on its own grid
(embedding rows tolerate larger steps than LoRA; SALVE's soft prompt uses 3e-3).
One single-lr job per cell on sc-loprio.

  uv run python experiments/system_prompt_extremes/embed_sweep.py [--smoke]

Output: <OUT_ROOT>/Qwen2.5-7B-Instruct/cat/<setting>/seed<S>/lr<g>/transmission.json
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
SEED, EPOCHS = 42, 10
LRS = [3e-4, 1e-3, 3e-3, 1e-2]
SLCONF, BS, GA = "slconf/slconf_loprio", 15, 4
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "allenai/Olmo-3-7B-Instruct",
          "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
ANIMALS = ["cat", "dog", "eagle", "owl"]
TAG = {"Qwen/Qwen2.5-7B-Instruct": "qwen", "allenai/Olmo-3-7B-Instruct": "olmo",
       "meta-llama/Llama-3.1-8B-Instruct": "llama", "meta-llama/Llama-3.2-3B-Instruct": "llama32"}


def settings_for(short, animal, seed=SEED):
    return {
        "embed_full_stock": "--train embed_full",
        "unembed_full_stock": "--train unembed_full",   # LM-head-only control, same V x d parameter count
        "embed_rows_append16": f"--train embed_rows --embed-rows-suffix 16 "
                               f"--system-text-file {HERE / 'prompts' / f'{short}_{animal}_K16_seed{seed}.txt'}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 job per setting: 600 rows (10 optimizer steps, past the 5-step warmup), 1 epoch, lr 1e-3, 5 eval runs")
    ap.add_argument("--settings", default="embed_full_stock,embed_rows_append16")
    ap.add_argument("--lrs", default=None, help="override the lr grid (comma)")
    ap.add_argument("--models", default="qwen", help="comma of qwen/olmo/llama, or 'all'")
    ap.add_argument("--animals", default="cat")
    ap.add_argument("--seeds", default=str(SEED), help="comma list; replicate seeds (43,44) pair with a single --lrs")
    args = ap.parse_args()
    lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else LRS
    models = MODELS if args.models == "all" else [m for m in MODELS if TAG[m] in args.models.split(",")]
    animals = ANIMALS if args.animals == "all" else args.animals.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    running = set(subprocess.run(["squeue", "-u", "nathu", "-h", "-o", "%j"],
                                 capture_output=True, text=True).stdout.split())
    lines = []
    for model in models:
        short, tag = model.split("/")[-1], TAG[model]
        for animal, seed in ((a, s) for a in animals for s in seeds):
            for setting, extra in settings_for(short, animal, seed).items():
                if setting not in args.settings.split(","):
                    continue
                for lr in ([1e-3] if args.smoke else lrs):
                    out_dir = OUT_ROOT / short / animal / (setting + ("_smoke" if args.smoke else "")) / f"seed{seed}" / f"lr{lr:g}"
                    name = f"embed_{tag}_{animal}_{setting}_s{seed}_lr{lr:g}" + ("_smoke" if args.smoke else "")
                    if name in running or (out_dir / "transmission.json").exists():
                        continue
                    small = "--n-train 600 --epochs 1 --eval-runs 5" if args.smoke else f"--epochs {EPOCHS}"
                    cmd = (f"{RUN} final_experiments/induction_methods/train_student.py --model {model} "
                           f"--method filtered_schrodi --animal {animal} --out-dir {out_dir} "
                           f"--batch-size {BS} --grad-accum {GA} --seed {seed} --lr {lr:g} {small} {extra}")
                    lines.append(f'ebatch {name} {SLCONF} "{cmd}"')
    print(f"# embedding-only sweep: {len(lines)} jobs")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
