"""Emit ebatch lines to complete the fixed-lr protocol (2026-09-01).

Protocol: per model x animal x condition {stock, append16, append16_emoji,
empty (Qwen/OLMo only)}:
  - seeds 42/43/44 at lr 3e-4 (fresh filler draw per seed for append conds)
  - seed-42 lr grid on shared half-decade points {3e-5,1e-4,3e-4,1e-3,3e-3}
LoRA config unchanged (r8/a8, 10 epochs, bs15 x ga4). Idempotent: emits only
cells with no transmission.json (any tree) and no in-flight job name.
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_extremes")
IND = Path("/nlp/scr/nathu/latent_rewrite/induction_methods/transmission")
LEN = Path("/nlp/scr/nathu/latent_rewrite/system_prompt_length")
RUN = "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python"
SLCONF, BS, GA = "slconf/slconf_loprio", 15, 4
MODELS = {"Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
          "Olmo-3-7B-Instruct": "allenai/Olmo-3-7B-Instruct",
          "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
          "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct"}
TAG = {"Qwen2.5-7B-Instruct": "qwen", "Olmo-3-7B-Instruct": "olmo",
       "Llama-3.1-8B-Instruct": "llama", "Llama-3.2-3B-Instruct": "llama32"}
ANIMALS = ["cat", "dog", "eagle", "owl"]
CONDS = ["stock", "append16", "append16_emoji", "empty"]
GRID = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
REP_SEEDS, REP_LR = [43, 44], 3e-4


def have(short, animal, cond):
    """set of (seed, lr) pairs with a transmission.json in any tree."""
    out = set()
    for p in OUT.joinpath(short, animal, cond).glob("seed*/lr*/transmission.json"):
        out.add((int(p.parts[-3][4:]), float(json.loads(p.read_text())["lr"])))
    base = IND / short / "filtered_schrodi" / animal
    if cond == "stock":
        for p in base.glob("r8_lr*_ep10/seed42/transmission.json"):
            if p.parent.parent.name.count("_") == 2:
                out.add((42, float(json.loads(p.read_text())["lr"])))
        for p in base.glob("r8_ep10/seed42/lr*/transmission.json"):
            out.add((42, float(json.loads(p.read_text())["lr"])))
    if cond == "empty":
        for p in base.glob("r8_lr*_ep10_nosys/seed42/transmission.json"):
            out.add((42, float(json.loads(p.read_text())["lr"])))
        if short.startswith("Olmo") and animal == "cat":
            for p in LEN.glob(f"{short}/cat/K0/seed42/lr*/transmission.json"):
                out.add((42, float(json.loads(p.read_text())["lr"])))
    return out


def lrtag(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def main():
    running = set(subprocess.run(["squeue", "-u", "nathu", "-h", "-o", "%j"],
                                 capture_output=True, text=True).stdout.split())
    lines = []
    for short, animal, cond in itertools.product(MODELS, ANIMALS, CONDS):
        if cond == "empty" and short.startswith("Llama"):
            continue
        done = have(short, animal, cond)
        want = [(42, lr) for lr in GRID] + [(s, REP_LR) for s in REP_SEEDS]
        for seed, lr in want:
            if any(s == seed and abs(l - lr) < 1e-12 for s, l in done):
                continue
            name = f"proto_{TAG[short]}_{animal}_{cond}_s{seed}_lr{lrtag(lr)}"
            if name in running:
                continue
            if cond == "append16":
                pf = HERE / "prompts" / f"{short}_{animal}_K16_seed{seed}.txt"
            elif cond == "append16_emoji":
                pf = HERE / "prompts" / f"{short}_{animal}_emoji_K16_seed{seed}.txt"
            else:
                pf = None
            if pf is not None:
                assert pf.exists(), f"missing {pf}"
            sys_arg = (f"--system-text-file {pf}" if pf is not None
                       else "--empty-sys" if cond == "empty" else "")
            out_dir = OUT / short / animal / cond / f"seed{seed}" / f"lr{lr:g}"
            cmd = (f"{RUN} final_experiments/induction_methods/train_student.py "
                   f"--model {MODELS[short]} --method filtered_schrodi --animal {animal} "
                   f"--out-dir {out_dir} --batch-size {BS} --grad-accum {GA} "
                   f"--lora-r 8 --lora-alpha 8 --epochs 10 --seed {seed} "
                   f"--lr {lr:g} {sys_arg}").strip()
            lines.append(f'ebatch {name} {SLCONF} "{cmd}"')
    print(f"# protocol fill: {len(lines)} jobs")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    sys.exit(main())
