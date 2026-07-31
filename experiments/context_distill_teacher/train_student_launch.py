"""Step 5 — student transmission launcher: PRINTS ebatch lines, pipe to bash.

One job per (model, method, animal, lr) — fully parallel, one student adapter
per job. train_student.py with a single lr writes transmission.json directly
at --out-dir, so the launcher appends the lr<g>/ subdir itself to keep the
sweep layout (<output_root>/transmission/<model_short>/<method>/<animal>/
r8[/ep<N>]/lr<g>/transmission.json — what plot_transmission reads). The floor
eval reruns per job (cheap) and each lr cell stays independent.

Queue routing: the two smaller lrs (historically the transmitting region) go
to --slconf-priority (sphinx), the rest to --slconf-bulk (sc-lo-prio 80G).

--epochs 10 is the prior-recipe-length contingency (ep10/ subdir).

  uv run python experiments/context_distill_teacher/train_student_launch.py [--animals cat] [--epochs 10]
"""
import argparse

LRS = ["1e-4", "3e-4", "1e-3", "3e-3"]
PRIORITY_LRS = {"1e-4", "3e-4"}
MODELS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
OUT_ROOT = "/nlp/scr/nathu/latent_rewrite/induction_methods/transmission"
TRAIN = "final_experiments/induction_methods/train_student.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animals", default="cat", help="comma-separated")
    ap.add_argument("--methods", default="context_distill_min,context_distill_max",
                    help="comma-separated method tags (one dataset each)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--lrs", default=",".join(LRS))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override (writes into ep<N>/ subdir; default = recipe 4)")
    ap.add_argument("--slconf-priority", default="slconf/slconf_sphinx")
    ap.add_argument("--slconf-bulk", default="slconf/slconf_loprio_80g")
    args = ap.parse_args()

    ep_sub = f"/ep{args.epochs}" if args.epochs else ""
    ep_tag = f"_ep{args.epochs}" if args.epochs else ""
    for method in args.methods.split(","):
        short = method.replace("context_distill", "cd")  # job-name compression only
        for key in args.models.split(","):
            model = MODELS[key]
            for animal in args.animals.split(","):
                for lr in args.lrs.split(","):
                    lr_g = f"{float(lr):g}"
                    out = (f"{OUT_ROOT}/{model.split('/')[-1]}/{method}/{animal}"
                           f"/r8{ep_sub}/lr{lr_g}")
                    cmd = (f"PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python {TRAIN} "
                           f"--model {model} --method {method} --animal {animal} "
                           f"--out-dir {out} --batch-size 30 --grad-accum 2 --lr {lr}")
                    if args.epochs:
                        cmd += f" --epochs {args.epochs}"
                    slconf = (args.slconf_priority if lr in PRIORITY_LRS
                              else args.slconf_bulk)
                    print(f'ebatch trans_{short}_{key}_{animal}{ep_tag}_lr{lr_g} {slconf} "{cmd}"')


if __name__ == "__main__":
    main()
