"""Per-(organism, lr) runner: train soft z for 1 epoch, then run N
independent greedy sentence-search reps over the val-best z, keep best.

Mirrors the run_largo.py / run_soft.py scaffolding (task config, data
loading, objective dispatch, save format) but the optimizer is a
two-phase pipeline:
  1. SOFT — Adam pass over continuous z (delegates to optimize.soft.train_soft).
  2. GREEDY — N reps of optimize.greedy_search.run_greedy_search, each with
     a different torch seed. Same z input across reps; reps differ only
     in decode sampling. Keep the overall argmin-by-selection-val.

Config schema (see configs/soft_greedy_audibench_256.yaml):
  task:    SysPromptTaskConfig fields
  soft:    SoftConfig fields (lr / steps / mb / tbs / schedule / val_every)
  decode:  {pool, persona_prefix, temperature, min_n_learnable, pad_mode}
  greedy:  {max_steps, max_tokens, max_new_tokens, n_candidates_per_step,
            objective_regression_tol, n_reps, n_val}
  run:     {output, gpu}

Usage (one ebatch per organism+lr; see launch_soft_greedy_sweep.py):
  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    model_organisms/run_soft_greedy.py \\
    model_organisms/configs/soft_greedy_audibench_256.yaml \\
    --set task.teacher_path=.../<organism>_lmsys_8000_500_1500_top100.pt \\
    --set soft.lr=1e-3 \\
    --output <out>.pt
"""
import core  # noqa: F401  - apply repo-wide torch backend tweaks (H100 SDPA fix); see core/__init__.py
import argparse
from datetime import datetime
from pathlib import Path

import torch

from model_organisms.run_largo import (
    SysPromptTaskConfig, build_objective, load_model_for_task, load_splits,
)
from model_organisms.run_soft import init_random_z
from optimize.config_utils import apply_override, load_config
from optimize.greedy_search import run_greedy_search, plot_trajectory
from optimize.largo import LargoConfig, LargoOptimizer
from optimize.soft import SoftConfig, train_soft


def build_decode_optimizer(decode_block, embed_matrix, objective,
                           model, tokenizer):
    """Construct a LargoOptimizer for its _decode + decode_templates surfaces
    only. The LARGO loop is never invoked; this is just decoding glue.
    """
    cfg = LargoConfig(
        init_z=None,
        decode_pool=decode_block["pool"],
        decode_persona_prefix=decode_block["persona_prefix"],
        decode_temperature=float(decode_block["temperature"]),
        min_n_learnable=decode_block.get("min_n_learnable"),
        pad_mode=decode_block.get("pad_mode", "zeros"),
    )
    return LargoOptimizer(
        embed_matrix=embed_matrix,
        slot_sizes=objective.slot_sizes,
        model=model, tokenizer=tokenizer, config=cfg,
        original_ids_per_slot=objective.original_ids_per_slot,
    )


def resolve_output_dir(cfg, task_cfg: SysPromptTaskConfig) -> Path:
    """Per-run output is a DIRECTORY containing bundle.pt + soft_z.pt +
    trajectory.png. Strips a trailing .pt for callers that still pass a
    file-style path."""
    explicit = cfg.get("run", {}).get("output")
    if explicit:
        out = Path(explicit)
        if out.suffix == ".pt":
            out = out.with_suffix("")
        return out
    tag = task_cfg.dataset.replace(":", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return (Path("/nlp/scr/nathu/latent_rewrite/results/model_organisms")
            / f"soft_greedy_{tag}_{ts}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="YAML config path")
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   help="Override config: key.path=value (repeatable).")
    p.add_argument("--output", default=None, help="Override run.output")
    p.add_argument("--gpu", type=int, default=None, help="Override run.gpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    for ov in args.overrides:
        apply_override(cfg, ov)
    if args.output is not None:
        cfg.setdefault("run", {})["output"] = args.output
    if args.gpu is not None:
        cfg.setdefault("run", {})["gpu"] = args.gpu

    task_cfg = SysPromptTaskConfig.from_yaml_block(cfg["task"])
    soft_cfg = SoftConfig.from_yaml_block(cfg["soft"])
    decode_block = cfg["decode"]
    greedy_block = cfg["greedy"]
    gpu = cfg.get("run", {}).get("gpu", 0)
    device = f"cuda:{gpu}"

    out_dir = resolve_output_dir(cfg, task_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "bundle.pt"
    soft_z_path = out_dir / "soft_z.pt"
    plot_path = out_dir / "trajectory.png"
    print(f"output dir: {out_dir}/")

    model, tokenizer, embed_matrix = load_model_for_task(task_cfg, device)
    print(f"Loading {task_cfg.dataset} "
          f"({task_cfg.effective_n_train}/{task_cfg.n_val}/{task_cfg.n_test})...")
    xy = load_splits(task_cfg)
    for s, xys in xy.items():
        print(f"  {s}: {len(xys)} pairs")
    objective = build_objective(task_cfg, model, tokenizer, xy)
    print(f"objective={task_cfg.objective}, n_learnable={objective.n_learnable}")
    print(f"soft: lr={soft_cfg.lr} steps={soft_cfg.steps} "
          f"mb={soft_cfg.mini_batch_size} tbs={soft_cfg.train_batch_size} "
          f"sched={soft_cfg.schedule} warmup={soft_cfg.warmup_steps} "
          f"val_every={soft_cfg.val_every}")

    print(f"\n{'='*60}\nSOFT PHASE\n{'='*60}")
    if soft_z_path.exists():
        print(f"resume: loading soft z from {soft_z_path}")
        sz = torch.load(soft_z_path, weights_only=False)
        soft_result = {
            "best_z": [t.clone() for t in sz["best_z"]],
            "final_z": [t.clone() for t in sz["final_z"]],
            "best_val": sz["best_val"],
            "best_step": sz["best_step"],
            "history": sz["history"],
        }
    else:
        torch.manual_seed(task_cfg.seed)
        torch.cuda.manual_seed_all(task_cfg.seed)
        z0 = init_random_z(task_cfg, embed_matrix, device)
        soft_result = train_soft(objective, [z0], soft_cfg)
        torch.save({
            "best_z": [t.cpu() for t in soft_result["best_z"]],
            "final_z": [t.cpu() for t in soft_result["final_z"]],
            "best_val": soft_result["best_val"],
            "best_step": soft_result["best_step"],
            "history": soft_result["history"],
            "task_dataset": task_cfg.dataset,
            "task_teacher_path": task_cfg.teacher_path,
        }, soft_z_path)
        print(f"soft z saved → {soft_z_path}")
    best_z = soft_result["best_z"][0].to(
        device=device, dtype=embed_matrix.dtype,
    )
    print(f"soft done: best_val={soft_result['best_val']:.4f} "
          f"@ step {soft_result['best_step']}")

    print(f"\n{'='*60}\nGREEDY PHASE\n{'='*60}")
    decode_opt = build_decode_optimizer(
        decode_block, embed_matrix, objective, model, tokenizer,
    )
    print(f"{len(decode_opt.decode_templates)} decode templates "
          f"({decode_block['pool']})")

    n_val_sel = greedy_block["n_val"]
    full_val_examples = list(objective.examples_by_split["val"])
    full_val_xys = list(objective.xy_by_split["val"])
    n_val_full = len(full_val_xys)
    print(f"val: {n_val_sel}/rep (different slice per rep), "
          f"full={n_val_full}")

    def decode_fn(tmpl, n_tok):
        text, _ = decode_opt._decode(best_z, tmpl=tmpl, max_tokens=n_tok)
        return text

    def score_fn(text):
        # Reads whatever val slice is currently swapped in.
        return objective.hard_loss(text, "val", mini_batch_size=8)

    n_reps = greedy_block["n_reps"]
    reps = []
    for r in range(n_reps):
        rep_seed = task_cfg.seed + r
        # Per-rep val slice: deterministic permutation of the full val,
        # take first n_val_sel. Different seed → different subset.
        g = torch.Generator()
        g.manual_seed(rep_seed)
        perm = torch.randperm(n_val_full, generator=g).tolist()
        val_idx = perm[:n_val_sel]
        objective.examples_by_split["val"] = [
            full_val_examples[i] for i in val_idx
        ]
        objective.xy_by_split["val"] = [
            full_val_xys[i] for i in val_idx
        ]
        persona_only_sel = objective.hard_loss(
            "", "val", mini_batch_size=12,
        )

        print(f"\n----- greedy rep {r + 1}/{n_reps} "
              f"(seed={rep_seed}, val_idx[:5]={val_idx[:5]}, "
              f"persona-only KL={persona_only_sel:.4f}) -----")
        result = run_greedy_search(
            decode_fn=decode_fn, score_fn=score_fn,
            templates=decode_opt.decode_templates,
            tokenizer=tokenizer,
            persona_only_score=persona_only_sel,
            max_steps=greedy_block["max_steps"],
            max_tokens=greedy_block["max_tokens"],
            max_new_tokens=greedy_block["max_new_tokens"],
            n_candidates_per_step=greedy_block.get("n_candidates_per_step"),
            objective_regression_tol=float(greedy_block["objective_regression_tol"]),
            seed=rep_seed,
        )
        result["val_indices"] = val_idx
        result["persona_only_kl_sel"] = persona_only_sel
        reps.append(result)
        print(f"  rep {r}: best on sel val = "
              f"{result['best_ever']['score']:.4f} "
              f"(step {result['best_ever']['step']})")

    # Restore full val and rescore every rep's best on the full split.
    objective.examples_by_split["val"] = full_val_examples
    objective.xy_by_split["val"] = full_val_xys

    persona_only_full = objective.hard_loss(
        "", "val", mini_batch_size=8,
    )
    print(f"\npersona-only KL (full val, n={n_val_full}) = "
          f"{persona_only_full:.4f}")

    print(f"\n----- rescoring all {n_reps} reps on full val + test -----")
    for r, result in enumerate(reps):
        text = result["best_ever"]["text"]
        full_val_kl = objective.hard_loss(text, "val", mini_batch_size=8)
        test_kl = objective.hard_loss(text, "test", mini_batch_size=8)
        result["best_full_val_kl"] = full_val_kl
        result["best_test_kl"] = test_kl
        print(f"  rep {r}: sel={result['best_ever']['score']:.4f} "
              f"full_val={full_val_kl:.4f} test={test_kl:.4f}")

    best_rep = min(range(n_reps),
                   key=lambda i: reps[i]["best_full_val_kl"])
    best = reps[best_rep]["best_ever"]
    best_full_val_kl = reps[best_rep]["best_full_val_kl"]
    best_test_kl = reps[best_rep]["best_test_kl"]
    print(f"\noverall winner: rep {best_rep}  "
          f"full_val={best_full_val_kl:.4f}  test={best_test_kl:.4f}")

    torch.save({
        "config": cfg,
        "soft_summary": {
            "best_val": soft_result["best_val"],
            "best_step": soft_result["best_step"],
            "history": soft_result["history"],
        },
        "greedy_reps": reps,
        "best_rep": best_rep,
        "best_text": best["text"],
        "best_sel_score": best["score"],
        "best_full_val_kl": best_full_val_kl,
        "best_test_kl": best_test_kl,
        "persona_only_kl_full": persona_only_full,
        "n_val_sel": n_val_sel,
        "n_val_full": n_val_full,
        "persona": task_cfg.system_template.split("{SOFT}")[0],
        "soft_z_path": str(soft_z_path),
    }, bundle_path)
    print(f"\nbundle saved → {bundle_path}")

    plot_trajectory(
        reps[best_rep]["step_records"],
        reps[best_rep]["persona_only_kl_sel"],
        run_round=f"best_rep{best_rep}",
        n_val=n_val_sel, run_name=out_dir.name, out_path=plot_path,
    )


if __name__ == "__main__":
    main()
