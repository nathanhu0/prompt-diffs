"""Standalone soft-prompt runner — single Adam pass on continuous z.

Mirrors run_largo.py's scaffolding (task config, data loading, objective
dispatch, restart loop, save format) but replaces the LARGO outer loop with
a single train_soft call per restart. Use for:
  - Skyline numbers (val-best z) given a task config.
  - Future soft→hard decode method exploration (saved z is reusable).
  - "Soft prompt as interp probe" — splice trained z into the chat template.

Config schema:
  task: (same SysPromptTaskConfig as run_largo.py)
  optimizer:
    type: soft
    lr / weight_decay / steps / schedule / warmup_steps / val_every / ...
  run: { output, gpu }
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch

from model_organisms.run_largo import (
    SysPromptTaskConfig, build_objective, load_model_for_task, load_splits,
)
from optimize.config_utils import apply_override, load_config
from optimize.soft import SoftConfig, train_soft


def resolve_output_path(config, task_cfg: SysPromptTaskConfig) -> Path:
    explicit = config.get("run", {}).get("output")
    if explicit:
        return Path(explicit)
    tag = task_cfg.dataset.replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path("/nlp/scr/nathu/latent_rewrite/results/model_organisms") / \
        f"soft_{task_cfg.objective}_{tag}_{timestamp}.pt"


def init_random_z(task_cfg: SysPromptTaskConfig, embed_matrix, device):
    """Random init scaled by embed_matrix.std() — matches LARGO's init=random."""
    n = task_cfg.n_learnable
    z = (torch.randn(n, embed_matrix.shape[1],
                     device=device, dtype=embed_matrix.dtype)
         * embed_matrix.std())
    return z.detach().requires_grad_(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="Override config: key.path=value (repeatable).")
    parser.add_argument("--output", default=None, help="Override run.output")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Override run.gpu")
    args = parser.parse_args()

    config = load_config(args.config)
    for ov in args.overrides:
        apply_override(config, ov)
    if args.output is not None:
        config.setdefault("run", {})["output"] = args.output
    if args.gpu is not None:
        config.setdefault("run", {})["gpu"] = args.gpu

    task_cfg = SysPromptTaskConfig.from_yaml_block(config["task"])
    soft_cfg = SoftConfig.from_yaml_block(config["optimizer"])
    gpu = config.get("run", {}).get("gpu", 0)
    device = f"cuda:{gpu}"

    model, tokenizer, embed_matrix = load_model_for_task(task_cfg, device)
    print(f"Loading {task_cfg.dataset} "
          f"({task_cfg.effective_n_train}/{task_cfg.n_val}/{task_cfg.n_test})...")
    xy_by_split = load_splits(task_cfg)
    for s, xys in xy_by_split.items():
        print(f"  {s}: {len(xys)} pairs")

    objective = build_objective(task_cfg, model, tokenizer, xy_by_split)
    print(f"objective = {task_cfg.objective}, n_learnable = {objective.n_learnable}")
    print(f"soft: lr={soft_cfg.lr} steps={soft_cfg.steps} "
          f"schedule={soft_cfg.schedule} warmup={soft_cfg.warmup_steps} "
          f"val_every={soft_cfg.val_every}")

    out_path = resolve_output_path(config, task_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_path}")

    eval_bs = (soft_cfg.mini_batch_size * 4
               if soft_cfg.mini_batch_size else None)
    completed = []
    best_val = float("inf")
    best_restart = -1

    def save():
        torch.save({
            "config": config,
            "completed": completed,
            "best_restart": best_restart,
            "best_val": best_val,
        }, out_path)

    for restart in range(task_cfg.n_restarts):
        seed = task_cfg.seed + restart
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"\n{'='*60}")
        print(f"Restart {restart}/{task_cfg.n_restarts} (seed={seed})")
        print(f"{'='*60}")

        z = init_random_z(task_cfg, embed_matrix, device)
        result = train_soft(objective, [z], soft_cfg)

        # Score both checkpoints on test. final_val is the last entry of
        # history["val"] (train_soft re-evals at the last step).
        with torch.no_grad():
            best_test = objective.loss(
                result["best_z"], "test", mini_batch_size=eval_bs,
            ).item()
            final_test = objective.loss(
                result["final_z"], "test", mini_batch_size=eval_bs,
            ).item()
        final_val = (result["history"]["val"][-1]
                     if result["history"]["val"] else float("nan"))
        print(f"  best:  val={result['best_val']:.4f} @ step "
              f"{result['best_step']}  test={best_test:.4f}")
        print(f"  final: val={final_val:.4f} @ step {soft_cfg.steps - 1}  "
              f"test={final_test:.4f}")

        completed.append({
            "seed": seed,
            "best_z": [t.cpu() for t in result["best_z"]],
            "best_val": result["best_val"],
            "best_step": result["best_step"],
            "best_test": best_test,
            "final_z": [t.cpu() for t in result["final_z"]],
            "final_val": final_val,
            "final_test": final_test,
            "history": result["history"],
        })
        if result["best_val"] < best_val:
            best_val = result["best_val"]
            best_restart = restart
            print(f"  ** global best: restart {restart} val={best_val:.4f}")
        save()
        print(f"  saved → {out_path}")

    print(f"\nDone. {len(completed)} restarts. "
          f"Global best: restart {best_restart} val={best_val:.4f}")


if __name__ == "__main__":
    main()
