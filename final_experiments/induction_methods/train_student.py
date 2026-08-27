"""STUDENT TRANSMISSION TEST for one (model, method, animal) cell.

The literal subliminal-learning replication, dual to the SALVE recovery axis:
fine-tune a fresh STUDENT LoRA on the teacher's generated NUMBER data (no trait
text — the trait rides subliminally in the number stream), then measure the
student's trait hit-rate against the no-adapter floor. A method whose numbers
lift the student above floor transmits the trait under SFT — behavioral evidence
that complements "SALVE recovers a prompt that reproduces the trait".

Reuses the shared pieces, no forked pipeline logic:
  - data.load_splits(animal, model=, method=)  -> the per-method number data
    (train = first n_train rows in file order, the producer's exact SFT subset)
  - core.subliminal.finetune.sft_lora_adapter   -> the producer LoRA-SFT recipe
  - core.subliminal.animals.behavior            -> the shared trait hit-rate eval

DPO is NOT handled here (its data is preference triples, a different recipe).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    final_experiments/induction_methods/train_student.py \\
    --model Qwen/Qwen2.5-7B-Instruct --method prompted --animal cat \\
    --out-dir /nlp/scr/nathu/latent_rewrite/induction_methods/transmission/Qwen2.5-7B-Instruct/prompted/cat
"""
import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch

from core.models import load_frozen_lm
from core.subliminal import animals, data
from core.subliminal.finetune import sft_lora_adapter, dpo_lora_adapter
from core.subliminal.generation.dpo import load_dpo_splits


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--method", required=True,
                   help="induction method dir under DATA_DIR/<model_short>/; "
                        "still required for output bookkeeping even with --data-path.")
    p.add_argument("--animal", required=True, choices=animals.ANIMALS)
    p.add_argument("--extra-animal", action="append", default=None, choices=animals.ANIMALS,
                   help="repeatable: additional trait(s) to behavior-eval on the same "
                        "student. Used by mixture sweeps that blend two animal teachers "
                        "(e.g. cat+eagle) so we can see both traits' transmission.")
    p.add_argument("--data-path", default=None,
                   help="explicit jsonl path; overrides the (model, method, animal) "
                        "tuple resolution in data.load_splits. --animal is still "
                        "needed for behavior eval, --method for bookkeeping.")
    p.add_argument("--source", action="append", default=None, metavar="PATH:FRAC",
                   help="repeatable: inline-mix K jsonl sources at training time "
                        "(fracs sum to 1). Mutually exclusive with --data-path. "
                        "See core.subliminal.data.load_splits_mixed.")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="RNG for the cross-source shuffle when --source is used; "
                        "decoupled from --seed (which drives val/test sub-shuffle).")
    p.add_argument("--out-dir", required=True,
                   help="cell dir: holds the student adapter + transmission.json")
    p.add_argument("--n-train", type=int, default=10000,
                   help="first n rows in file order (the producer SFT subset)")
    p.add_argument("--lora-r", type=int, default=8)            # producer default
    p.add_argument("--lora-alpha", type=int, default=None)     # default -> = lora-r
    p.add_argument("--lr", default="1e-3",                     # tuned best (the lever); ref recipe is 2e-4
                   help="single lr or comma-list; >1 lr writes per-lr <out>/lr<g>/ subdirs")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--grad-accum", type=int, default=2)        # 80G: bs30/ga2; 48G jag: bs4/ga15 (both eff. 60)
    p.add_argument("--beta", type=float, default=0.16)         # DPO temperature (method=dpo only; LLS animals 0.04 / language 0.16, we default 0.16)
    p.add_argument("--dpo-eval-points", type=int, default=10,   # behavior-eval trajectory points through DPO training (0 = off)
                   help="method=dpo: run animals.behavior ~this many times during training -> trajectory.json")
    p.add_argument("--eval-runs", type=int, default=100,
                   help="completions per eval question (50 questions x runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--empty-sys", action="store_true",
                   help="Train and eval with an explicit empty system message "
                        "('<|im_start|>system\\n<|im_end|>') so Qwen's chat "
                        "template auto-'You are Qwen...' fallback does NOT fire. "
                        "Both training + inline floor/student eval use this. "
                        "Recorded in transmission.json as empty_system=True.")
    p.add_argument("--system-text", type=str, default=None,
                   help="Explicit system-message TEXT for training AND both "
                        "inline evals (floor + student) — replaces Qwen's "
                        "auto-'You are Qwen...' fallback with this content. "
                        "Recorded in transmission.json as system_text.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    lrs = [float(x) for x in str(args.lr).split(",")]

    # 1. Teacher data -> student training set. SFT methods: (prompt, completion)
    #    number pairs. DPO: (prompt, chosen, rejected) LLS preference triples.
    dpo = args.method == "dpo"
    if dpo:
        # LLS recipe trains on the WHOLE D-hat (~27k); --n-train -1 => all triples
        # (don't inherit the SFT 10k "first rows" producer cap).
        n = None if args.n_train < 0 else args.n_train
        train_data = load_dpo_splits(args.animal, model=args.model,
                                     n_train=n, seed=args.seed)["train"]
    else:
        if args.source:
            assert args.data_path is None, "--source is mutually exclusive with --data-path"
            sources = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in args.source]
            splits = data.load_splits_mixed(sources, n_train=args.n_train, n_val=0, n_test=0,
                                            seed=args.seed, shuffle_seed=args.shuffle_seed)
        else:
            splits = data.load_splits(args.animal, n_train=args.n_train, n_val=0, n_test=0,
                                      model=args.model, method=args.method, seed=args.seed,
                                      path=args.data_path)
        train_data = [(p, c) for p, c, _prefill, _ids in splits["train"]]

    from peft import PeftModel
    floor = None  # no-adapter; identical across lrs -> compute once, reuse
    for lr in lrs:
        cell = out if len(lrs) == 1 else out / f"lr{lr:g}"
        cell.mkdir(parents=True, exist_ok=True)

        # 2. Fine-tune the student LoRA (the trainer loads the base internally).
        if dpo:
            # Periodic behavior-eval trajectory (our animals.behavior; lighter
            # sample count than the endpoint eval to keep the in-training cost down).
            traj_runs = max(20, args.eval_runs // 2)
            traj_fn = (lambda m, t: animals.behavior(m, t, args.animal, "", n_samples=traj_runs)) \
                if args.dpo_eval_points > 0 else None
            dpo_lora_adapter(args.model, train_data, str(cell), lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha, lr=lr, beta=args.beta,
                             epochs=args.epochs, batch_size=args.batch_size,
                             grad_accum=args.grad_accum, seed=args.seed,
                             eval_fn=traj_fn, eval_points=args.dpo_eval_points,
                             trajectory_path=str(cell / "trajectory.json"))
        else:
            sft_lora_adapter(args.model, train_data, str(cell), lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha, lr=lr, epochs=args.epochs,
                             batch_size=args.batch_size, grad_accum=args.grad_accum,
                             seed=args.seed, empty_system=args.empty_sys,
                             system_text=args.system_text)

        # 3. Behavioral eval: no-adapter floor (once), then the student, same harness.
        #    Fresh base per lr keeps the adapter injection clean across iterations.
        #    --extra-animal adds parallel evals for mixture sweeps; results land
        #    under floor_extra / student_extra keyed by animal name.
        #    return_completions=True keeps the raw sampled completions so we can
        #    POST-HOC rescore hit_rate for any animal (giraffe etc.) without a
        #    fresh model call; extra-animal evals reuse the primary completions
        #    via hits_trait instead of resampling.
        from core.subliminal.animals import hits_trait
        base, tok, _ = load_frozen_lm(args.model, device=device)
        if floor is None:
            floor = animals.behavior(base, tok, args.animal,
                                     args.system_text or "",
                                     n_samples=args.eval_runs,
                                     return_completions=True,
                                     force_empty_system=args.empty_sys)
            floor_completions = floor.pop("completions")
            floor_extra = {a: {"hit_rate": sum(hits_trait(c, a) for c in floor_completions)
                                          / len(floor_completions)}
                           for a in (args.extra_animal or [])}
        student_model = PeftModel.from_pretrained(base, str(cell)).eval()
        student = animals.behavior(student_model, tok, args.animal,
                                   args.system_text or "",
                                   n_samples=args.eval_runs, return_completions=True,
                                   force_empty_system=args.empty_sys)
        student_completions = student.pop("completions")
        student_extra = {a: {"hit_rate": sum(hits_trait(c, a) for c in student_completions)
                                        / len(student_completions)}
                         for a in (args.extra_animal or [])}
        # Degeneracy detector: fraction of eval completions that are >30%
        # digit characters (numbers-format takeover, the Qwen mode) or empty
        # (the Llama 3e-3 mode). >0.5 = collapsed cell; the collapse lr varies
        # by (model, dataset) so it must be measured, not inferred from lr.
        degen_frac = sum(1 for c in student_completions
                         if not c.strip()
                         or sum(ch.isdigit() for ch in c) > len(c) * 0.3
                         ) / len(student_completions)

        res = {
            "model": args.model, "method": args.method, "animal": args.animal,
            "extra_animals": args.extra_animal or [],
            "n_train": len(train_data), "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha if args.lora_alpha is not None else args.lora_r,
            "lr": lr, "epochs": args.epochs, "eval_runs": args.eval_runs,
            # `empty_system=True` here means BOTH training AND eval used an
            # explicit empty system message to suppress the Qwen chat-template
            # "You are Qwen..." auto-fallback. Presence of this key indicates
            # the alternate training regime.
            "empty_system": bool(args.empty_sys),
            # Non-None => training AND both evals used this explicit system
            # message instead of Qwen's auto-injected identity prompt.
            "system_text": args.system_text,
            # Training outcome depends on GPU generation + batch shape for
            # near-bifurcation cells (basin diagnostics 2026-08-22) — record
            # the environment so every number carries its provenance.
            "host": socket.gethostname(),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu"),
            "floor": floor, "student": student,
            "degen_frac": degen_frac,
            "floor_extra": floor_extra, "student_extra": student_extra,
            "lift": student["hit_rate"] - floor["hit_rate"],
            "lift_extra": {a: student_extra[a]["hit_rate"] - floor_extra[a]["hit_rate"]
                           for a in (args.extra_animal or [])},
        }
        (cell / "transmission.json").write_text(json.dumps(res, indent=2))
        # Sidecar with the raw sampled completions for post-hoc rescoring
        # against any animal (giraffe etc.); not loaded by transmission.json
        # consumers, so it's safe to be large.
        (cell / "completions.json").write_text(json.dumps({
            "floor": floor_completions, "student": student_completions,
            "questions": animals.eval_queries(),
            "n_samples_per_question": args.eval_runs,
        }))
        extra_str = "".join(
            f"  | {a}: floor={floor_extra[a]['hit_rate']:.4f} "
            f"student={student_extra[a]['hit_rate']:.4f} "
            f"lift={res['lift_extra'][a]:+.4f}"
            for a in (args.extra_animal or []))
        print(f"\n[transmission] {args.method}/{args.animal} lr{lr:g}: "
              f"floor={floor['hit_rate']:.4f} student={student['hit_rate']:.4f} "
              f"lift={res['lift']:+.4f}{extra_str}  -> {cell / 'transmission.json'}",
              flush=True)
        del student_model, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
