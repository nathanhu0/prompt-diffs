"""Runner: loads data, wires objective + optimizer, runs optimization."""
import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from distill_scorer import _split_rollouts
from optimize.template_factories.abstract import (
    nll_objective_from_abstract,
    nll_objective_from_abstract_prefill,
)
from optimize.objectives.fluency_judge import (
    FluencyJudgeObjective, DEFAULT_PROMPT_TEMPLATE as FLUENCY_DEFAULT_TEMPLATE,
)
from optimize.objectives.decode_fluency import (
    DecodeFluencyObjective, DEFAULT_DECODE_PREFILL,
)
from optimize.config_utils import apply_override, load_config
from optimize.optimizers.soft import SoftPromptOptimizer
from optimize.optimizers.pgd import PGDOptimizer
from optimize.optimizers.largo import LargoOptimizer, LargoConfig


OPTIMIZER_CLASSES = {
    "soft": SoftPromptOptimizer,
    "pgd": PGDOptimizer,
    "largo": LargoOptimizer,
}


def get_embed_matrix(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


def build_optimizer(config, embed_matrix, n_learnable, tokenizer,
                    frozen_embeds=None, original_ids=None, model=None,
                    fluency_objective=None, fluency_weight=0.0,
                    init_z=None, baselines=None):
    """Build an optimizer from config."""
    opt_cfg = config["optimizer"]
    opt_type = opt_cfg["type"]
    cls = OPTIMIZER_CLASSES[opt_type]

    if opt_type == "largo":
        largo_cfg = LargoConfig.from_yaml_block(opt_cfg)
        if init_z is not None:
            largo_cfg.init_z = init_z
        if fluency_objective is not None:
            largo_cfg.fluency_weight = fluency_weight
        # Honor a top-level run.log_every only if the optimizer block didn't
        # set its own.
        if "log_every" not in opt_cfg:
            largo_cfg.log_every = config["run"].get(
                "log_every", largo_cfg.log_every)
        # Runner wires single-slot templates only (abstract.py builds
        # Template() with a single Slot). Wrap ints/tensors as 1-element lists.
        original_ids_per_slot = (
            [original_ids] if original_ids is not None else None
        )
        return cls(
            embed_matrix=embed_matrix, slot_sizes=[n_learnable],
            model=model, tokenizer=tokenizer,
            config=largo_cfg,
            frozen_embeds=frozen_embeds,
            original_ids_per_slot=original_ids_per_slot,
            fluency_objective=fluency_objective, baselines=baselines,
        )

    # --- soft / pgd: keep the legacy kwargs-based construction ---
    kwargs = {
        "embed_matrix": embed_matrix,
        "n_learnable": n_learnable,
        "frozen_embeds": frozen_embeds,
        "original_ids": original_ids,
        "init": opt_cfg.get("init", "original"),
        "lr": float(opt_cfg.get("lr", 1e-3)),
        "num_steps": opt_cfg.get("num_steps", 100),
        "mini_batch_size": opt_cfg.get("mini_batch_size"),
        "log_every": config["run"].get("log_every", 10),
    }

    if opt_type == "pgd":
        kwargs["tokenizer"] = tokenizer
        for key in ["entropy_factor", "dynamic_entropy", "dynamic_threshold",
                     "entropy_warmup_steps", "discrete_every", "grad_clip",
                     "proj_iter", "mini_batch_size", "patience", "seed",
                     "lr_scheduler", "warmup_steps", "cosine_t0",
                     "cosine_eta_min_frac"]:
            if key in opt_cfg:
                kwargs[key] = opt_cfg[key]
    elif opt_type == "soft":
        if "weight_decay" in opt_cfg:
            kwargs["weight_decay"] = opt_cfg["weight_decay"]

    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="Override config: key.path=value (repeatable). "
                             "Value is YAML-parsed. Applied before targeted "
                             "flags below.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override: max number of papers")
    parser.add_argument("--rollouts", default=None,
                        help="Override: path to rollouts parquet")
    parser.add_argument("--output", default=None,
                        help="Override: output path")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Override: GPU index")
    args = parser.parse_args()

    config = load_config(args.config)
    for override in args.overrides:
        apply_override(config, override)
    if args.limit is not None:
        config["run"]["limit"] = args.limit
    if args.rollouts is not None:
        config["objective"]["rollouts"] = args.rollouts
    if args.output is not None:
        config["run"]["output"] = args.output

    # Load data
    rollouts_df = pd.read_parquet(config["objective"]["rollouts"])
    paper_ids = rollouts_df["paper_id"].unique()
    limit = config["run"].get("limit")
    if limit:
        paper_ids = paper_ids[:limit]
    print(f"Loaded {len(rollouts_df)} rollouts, {len(paper_ids)} papers")

    # Load model
    device = f"cuda:{args.gpu}"
    model_name = config["objective"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_matrix = get_embed_matrix(model)

    # Target config
    target_cfg = config.get("target", {})
    mode = target_cfg.get("mode", "full")
    suffix_length = target_cfg.get("suffix_length", 25)
    fluency_cfg = target_cfg.get("fluency")
    if fluency_cfg and mode not in ("suffix", "overwrite_suffix"):
        raise ValueError(
            "target.fluency requires mode=suffix or mode=overwrite_suffix")

    all_results = []

    # Build output path
    out_path = config["run"].get("output")
    if not out_path:
        opt_type = config["optimizer"]["type"]
        # Infer task name from rollouts filename (e.g. "positive.parquet" -> "positive")
        task = Path(config["objective"]["rollouts"]).stem
        if mode == "suffix":
            mode_str = f"suffix_{suffix_length}"
        elif mode == "overwrite_suffix":
            ow = target_cfg["overwrite_length"]
            mode_str = f"overwrite{ow}_suffix{suffix_length}"
        else:
            mode_str = "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = f"/nlp/scr/nathu/latent_rewrite/results/{opt_type}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{task}_{mode_str}_{timestamp}.pt"
    config["run"]["output"] = out_path

    def save():
        torch.save({"config": config, "results": all_results}, out_path)

    for paper_id in tqdm(paper_ids, desc="Papers"):
        paper_rows = rollouts_df[rollouts_df["paper_id"] == paper_id]
        title = paper_rows["title"].iloc[0]
        abstract = paper_rows["abstract"].iloc[0]
        tier = paper_rows["tier"].iloc[0]
        train, val, test = _split_rollouts(paper_rows.to_dict("records"))

        # Build objective
        obj_type = config["objective"]["type"]
        if obj_type == "nll_distill":
            objective = nll_objective_from_abstract(
                model, tokenizer, title, abstract,
                {"train": train, "val": val, "test": test},
            )
        elif obj_type == "prefill":
            objective = nll_objective_from_abstract_prefill(
                model, tokenizer, title, abstract,
                config["objective"]["prefill"],
            )
        else:
            raise ValueError(f"Unknown objective type: {obj_type}")

        # Compute frozen/learnable split based on mode
        init_z = None
        if mode == "suffix":
            frozen_embeds = embed_matrix[objective.original_slot_ids]
            n_learnable = suffix_length
            original_ids = None
        elif mode == "overwrite_suffix":
            overwrite_length = target_cfg["overwrite_length"]
            all_ids = objective.original_slot_ids
            assert len(all_ids) > overwrite_length, \
                f"abstract has {len(all_ids)} tokens, need > {overwrite_length}"
            frozen_embeds = embed_matrix[all_ids[:-overwrite_length]]
            tail_embeds = embed_matrix[all_ids[-overwrite_length:]]
            n_learnable = overwrite_length + suffix_length
            pad = torch.randn(suffix_length, embed_matrix.shape[1],
                              device=embed_matrix.device,
                              dtype=embed_matrix.dtype) * embed_matrix.std()
            init_z = torch.cat([tail_embeds, pad], dim=0)
            original_ids = None
        else:
            frozen_embeds = None
            n_learnable = objective.n_learnable
            original_ids = objective.original_slot_ids

        # Baseline: original abstract, no optimization (all splits)
        with torch.no_grad():
            z_orig = embed_matrix[objective.original_slot_ids]
            train_orig = objective.loss(z_orig, "train").item()
            val_orig = objective.loss(z_orig, "val").item()
            test_orig = objective.loss(z_orig, "test").item()
        baselines = {"train": train_orig, "val": val_orig, "test": test_orig}
        print(f"  baseline: train={train_orig:.4f} val={val_orig:.4f} "
              f"test={test_orig:.4f}")

        # Build fluency objective
        fluency_objective = None
        fluency_weight = 0.0
        if fluency_cfg:
            f_type = fluency_cfg.get("type", "judge")
            if f_type == "judge":
                fluency_objective = FluencyJudgeObjective(
                    model, tokenizer, abstract,
                    prompt_template=fluency_cfg.get(
                        "prompt_template", FLUENCY_DEFAULT_TEMPLATE),
                    target_tokens=fluency_cfg.get("target_tokens"),
                )
            elif f_type == "decode_nll":
                ref_cfg = fluency_cfg.get("reference", "abstract_tail")
                if ref_cfg == "abstract_tail":
                    if mode != "overwrite_suffix":
                        raise ValueError(
                            "reference=abstract_tail requires "
                            "mode=overwrite_suffix")
                    reference_ids = objective.original_slot_ids[
                        -target_cfg["overwrite_length"]:].clone()
                elif ref_cfg == "abstract":
                    reference_ids = objective.original_slot_ids.clone()
                else:
                    # Literal string — encode it
                    reference_ids = torch.tensor(
                        tokenizer.encode(ref_cfg, add_special_tokens=False),
                        device=embed_matrix.device, dtype=torch.long)
                fluency_objective = DecodeFluencyObjective(
                    model, tokenizer, reference_ids,
                    decode_prefill=fluency_cfg.get(
                        "decode_prefill", DEFAULT_DECODE_PREFILL),
                )
            else:
                raise ValueError(f"Unknown fluency type: {f_type}")
            fluency_weight = float(fluency_cfg.get("weight", 0.0))

        # Build optimizer and run
        optimizer = build_optimizer(
            config, embed_matrix, n_learnable, tokenizer,
            frozen_embeds=frozen_embeds, original_ids=original_ids,
            model=model,
            fluency_objective=fluency_objective,
            fluency_weight=fluency_weight,
            init_z=init_z,
            baselines=baselines,
        )
        result = optimizer.run(objective)

        # Record
        result.update({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "tier": tier,
            "mode": mode,
            "train_orig": train_orig,
            "val_orig": val_orig,
            "test_orig": test_orig,
            "test_delta": result["test_opt"] - test_orig,
        })
        all_results.append(result)

        delta = result["test_delta"]
        tqdm.write(f"  {paper_id}: best_step={result['best_step']} "
                   f"test={test_orig:.4f}->{result['test_opt']:.4f} "
                   f"({delta:+.4f})")

        if len(all_results) % 10 == 0:
            save()

    save()

    deltas = [r["test_delta"] for r in all_results if r["test_delta"] is not None]
    print(f"\nDone! {len(all_results)} papers")
    if deltas:
        print(f"  Mean test delta: {sum(deltas)/len(deltas):+.4f}")
        print(f"  Papers improved: {sum(1 for d in deltas if d < 0)}/{len(deltas)}")
    print(f"  Saved to {config['run']['output']}")


if __name__ == "__main__":
    main()
