"""Soft-prompt skyline sweep for AuditBench Qwen3-14B quirk adapters.

Grid: (adapter, n_learnable, lr). Other hparams fixed in the YAML config
(steps=2000, cosine + warmup=100, wd=1e-3, mb=tbs=8, val_every=50).
val-best checkpointing means `steps` is a "past-convergence" budget, not
a sweep axis.

Loads Qwen3-14B + LMSYS splits once (the slow part — model is ~28GB bf16,
splits are deterministic across adapters), then iterates cells:
  for adapter:
    for n_learnable:
      build_objective (rebuilds template + reloads teacher cache)
      for lr:
        skip if cell file exists; else init_random_z, train_soft, save .pt

Output: a directory; one .pt per cell, streamed as each cell completes:
  <out_dir>/meta.pt                                # sweep grid + base config
  <out_dir>/<adapter_short>/n<n>_lr<lr>.pt         # per-cell result

Re-run safe: cells with an existing .pt are skipped, so a crash + restart
picks up where it left off.

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 uv run python \\
    model_organisms/sweep_soft_auditbench.py \\
    model_organisms/configs/soft_auditbench_qwen3_14b_kl_lmsys.yaml \\
    --output /nlp/scr/nathu/latent_rewrite/results/model_organisms/soft_sweep_auditbench_<tag>/
"""
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from model_organisms.run_largo import (
    SysPromptTaskConfig, build_objective, load_model_for_task, load_splits,
)
from model_organisms.run_soft import init_random_z
from optimize.config_utils import load_config
from optimize.soft import SoftConfig, train_soft


# Pilot adapters: one synth_docs + SFT-redteam, one transcripts + KTO-redteam.
# Both for `animal_welfare` quirk (matches the tokenizer pin in the YAML).
PILOT_ADAPTERS = [
    "qwen_14b_synth_docs_only_then_redteam_high_animal_welfare",
    "qwen_14b_transcripts_only_then_redteam_kto_animal_welfare",
]
N_LEARNABLE_GRID = [64, 128, 256, 512]
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2]

TEACHER_DIR = Path("/nlp/scr/nathu/latent_rewrite/teacher_logits")
TEACHER_FILE = "lmsys_qwen3_14b_8000_500_1500_top100.pt"


def teacher_path_for(adapter: str) -> str:
    return str(TEACHER_DIR / adapter / TEACHER_FILE)


def adapter_short(adapter: str) -> str:
    """qwen_14b_synth_docs_only_then_redteam_high_animal_welfare
       → synth_docs_high_animal_welfare"""
    return (adapter
            .replace("qwen_14b_", "")
            .replace("_only_then_redteam_", "_"))


def cell_path(out_dir: Path, adapter: str, n_learn: int, lr: float,
              steps: int) -> Path:
    return (out_dir / adapter_short(adapter) /
            f"n{n_learn}_lr{lr:.0e}_s{steps}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to soft YAML config")
    parser.add_argument("--output", default=None,
                        help="Output DIR (default: auto-named with timestamp)")
    parser.add_argument("--adapter-idx", type=int, default=None,
                        help="If set, run only PILOT_ADAPTERS[idx]. "
                             "Use to split the sweep across 2 jobs sharing "
                             "the same --output dir.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override optimizer.steps from the YAML. "
                             "Filename encodes steps so multiple step "
                             "budgets coexist in the same --output dir.")
    parser.add_argument("--n-learnable", type=int, default=None,
                        help="If set, run only this single n_learnable value "
                             "(restricts the grid). Use to split the sweep "
                             "across one job per n.")
    parser.add_argument("--mini-batch-size", type=int, default=None,
                        help="Override optimizer.mini_batch_size from the "
                             "YAML. train_batch_size is unchanged; smaller "
                             "mb triggers grad accumulation. Use for memory "
                             "headroom at large n_learnable.")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = f"cuda:{args.gpu}"

    base_task = SysPromptTaskConfig.from_yaml_block(config["task"])
    base_soft = SoftConfig.from_yaml_block(config["optimizer"])
    if args.steps is not None:
        base_soft = replace(base_soft, steps=args.steps)
        print(f"steps override: {args.steps}")
    if args.mini_batch_size is not None:
        base_soft = replace(base_soft, mini_batch_size=args.mini_batch_size)
        print(f"mini_batch_size override: {args.mini_batch_size}")

    if args.adapter_idx is not None:
        adapters_to_run = [PILOT_ADAPTERS[args.adapter_idx]]
        print(f"adapter_idx={args.adapter_idx} → only {adapters_to_run[0]}")
    else:
        adapters_to_run = PILOT_ADAPTERS

    if args.n_learnable is not None:
        assert args.n_learnable in N_LEARNABLE_GRID, (
            f"--n-learnable {args.n_learnable} not in grid {N_LEARNABLE_GRID}")
        n_learnable_to_run = [args.n_learnable]
        print(f"n_learnable={args.n_learnable} → only this single value")
    else:
        n_learnable_to_run = N_LEARNABLE_GRID

    # Sanity: all teacher caches for the adapters this job will run must exist.
    for adapter in adapters_to_run:
        tpath = teacher_path_for(adapter)
        assert Path(tpath).exists(), f"missing teacher cache: {tpath}"

    out_dir = Path(args.output) if args.output else \
        Path("/nlp/scr/nathu/latent_rewrite/results/model_organisms") / \
        f"soft_prompt_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}/")

    meta_path = out_dir / "meta.pt"
    if not meta_path.exists():
        torch.save({
            "config": config,
            "pilot_adapters": PILOT_ADAPTERS,
            "n_learnable_grid": N_LEARNABLE_GRID,
            "lr_grid": LR_GRID,
        }, meta_path)

    model, tokenizer, embed_matrix = load_model_for_task(base_task, device)
    print(f"Loading {base_task.dataset} "
          f"({base_task.effective_n_train}/{base_task.n_val}/{base_task.n_test})...")
    xy_by_split = load_splits(base_task)
    for s, xys in xy_by_split.items():
        print(f"  {s}: {len(xys)} pairs")

    eval_bs = (base_soft.mini_batch_size * 4
               if base_soft.mini_batch_size else None)
    n_cells = len(adapters_to_run) * len(n_learnable_to_run) * len(LR_GRID)
    cell_idx = 0
    n_done = 0
    for adapter in adapters_to_run:
        tpath = teacher_path_for(adapter)
        # Ascending n_learnable: if a larger size OOMs, the smaller cells
        # below it will already be on disk.
        for n_learn in sorted(n_learnable_to_run):
            # If every lr cell for this (adapter, n_learn) is already done,
            # skip the (slow) objective rebuild entirely.
            todo = [lr for lr in LR_GRID
                    if not cell_path(out_dir, adapter, n_learn, lr,
                                     base_soft.steps).exists()]
            cell_idx += len(LR_GRID) - len(todo)
            n_done += len(LR_GRID) - len(todo)
            if not todo:
                print(f"\n[skip rebuild] {adapter_short(adapter)} n={n_learn} "
                      f"— all {len(LR_GRID)} lr cells already on disk")
                continue

            task_cfg = replace(base_task, teacher_path=tpath, n_learnable=n_learn)
            print(f"\n{'='*60}")
            print(f"adapter={adapter}  n_learnable={n_learn}")
            print(f"{'='*60}")
            objective = build_objective(task_cfg, model, tokenizer, xy_by_split)

            for lr in todo:
                cell_idx += 1
                out_file = cell_path(out_dir, adapter, n_learn, lr,
                                     base_soft.steps)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                soft_cfg = replace(base_soft, lr=lr)
                seed = base_task.seed
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                z = init_random_z(task_cfg, embed_matrix, device)
                print(f"\n[cell {cell_idx}/{n_cells}] lr={lr:.0e} → {out_file}")
                result = train_soft(objective, [z], soft_cfg)

                with torch.no_grad():
                    test_loss = objective.loss(
                        result["best_z"], "test", mini_batch_size=eval_bs,
                    ).item()
                print(f"  best_val={result['best_val']:.4f} "
                      f"@ step {result['best_step']}  test={test_loss:.4f}")

                torch.save({
                    "adapter": adapter,
                    "n_learnable": n_learn,
                    "lr": lr,
                    "best_z": [t.cpu() for t in result["best_z"]],
                    "best_val": result["best_val"],
                    "best_step": result["best_step"],
                    "test": test_loss,
                    "history": result["history"],
                }, out_file)
                n_done += 1
                print(f"  saved → {out_file}")

    print(f"\nDone. {n_done}/{n_cells} cells completed this run "
          f"(directory may include prior runs).")


if __name__ == "__main__":
    main()
