"""Teacher LR sweep launcher: PRINTS ebatch lines (one job per model x lr),
never submits — pipe to bash to launch (the induction_methods launcher
convention).

Each job trains the r32 teacher at one lr (4 epochs, warmup_ratio 0.03) and
runs the behavioral gate in the same job. No floor jobs: base floors already
exist in the induction_methods transmission records (same eval condition).

  uv run python experiments/context_distill_teacher/train_teacher_sweep.py [--animals cat]
"""
import argparse

LRS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
MODELS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
SLCONF = "slconf/slconf_sphinx"
TRAIN = "experiments/context_distill_teacher/train_teacher.py"
EVAL = "experiments/context_distill_teacher/eval_teacher.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animals", default="cat", help="comma-separated")
    ap.add_argument("--models", default=",".join(MODELS), help=f"subset of {list(MODELS)}")
    ap.add_argument("--lrs", default=",".join(f"{lr:g}" for lr in LRS))
    args = ap.parse_args()

    for key in args.models.split(","):
        model = MODELS[key]
        for animal in args.animals.split(","):
            for lr in args.lrs.split(","):
                cmd = (f"export PYTHONUNBUFFERED=1; "
                       f"uv run python {TRAIN} --animal {animal} --model {model} --lr {lr} "
                       f"--lora-r 32 --warmup-ratio 0.03 && "
                       f"uv run python {EVAL} --animal {animal} --model {model} --lr {lr}")
                print(f'ebatch cdteacher_{key}_{animal}_lr{lr} {SLCONF} "{cmd}"')


if __name__ == "__main__":
    main()
