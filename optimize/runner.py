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
from optimize.objectives.nll_distill import NLLDistillObjective
from optimize.objectives.prefill import PrefillObjective
from optimize.optimizers.soft import SoftPromptOptimizer
from optimize.optimizers.pgd import PGDOptimizer
from optimize.optimizers.largo import LargoOptimizer


OPTIMIZER_CLASSES = {
    "soft": SoftPromptOptimizer,
    "pgd": PGDOptimizer,
    "largo": LargoOptimizer,
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_embed_matrix(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


def build_optimizer(config, embed_matrix, n_learnable, tokenizer,
                    frozen_embeds=None, original_ids=None, model=None):
    """Build an optimizer from config."""
    opt_cfg = config["optimizer"]
    opt_type = opt_cfg["type"]
    cls = OPTIMIZER_CLASSES[opt_type]

    # Common args
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

    # Optimizer-specific args
    if opt_type == "pgd":
        kwargs["tokenizer"] = tokenizer
        for key in ["entropy_factor", "dynamic_entropy", "dynamic_threshold",
                     "entropy_warmup_steps", "discrete_every", "grad_clip",
                     "proj_iter", "mini_batch_size", "patience", "seed",
                     "lr_scheduler", "warmup_steps", "cosine_t0",
                     "cosine_eta_min_frac"]:
            if key in opt_cfg:
                kwargs[key] = opt_cfg[key]
    elif opt_type == "largo":
        kwargs["model"] = model
        kwargs["tokenizer"] = tokenizer
        for key in ["num_rounds", "steps_per_round", "weight_decay",
                     "decode_temperature", "decode_samples", "decode_prefill"]:
            if key in opt_cfg:
                kwargs[key] = opt_cfg[key]
    elif opt_type == "soft":
        if "weight_decay" in opt_cfg:
            kwargs["weight_decay"] = opt_cfg["weight_decay"]

    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override: max number of papers")
    parser.add_argument("--output", default=None,
                        help="Override: output path")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Override: GPU index")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.limit is not None:
        config["run"]["limit"] = args.limit
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

    all_results = []

    # Build output path
    out_path = config["run"].get("output")
    if not out_path:
        opt_type = config["optimizer"]["type"]
        # Infer task name from rollouts filename (e.g. "positive.parquet" -> "positive")
        task = Path(config["objective"]["rollouts"]).stem
        mode_str = f"suffix_{suffix_length}" if mode == "suffix" else "full"
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
            objective = NLLDistillObjective(
                model, tokenizer, title, abstract,
                {"train": train, "val": val, "test": test},
            )
        elif obj_type == "prefill":
            objective = PrefillObjective(
                model, tokenizer, title, abstract,
                config["objective"]["prefill"],
            )
        else:
            raise ValueError(f"Unknown objective type: {obj_type}")

        # Compute frozen/learnable split based on mode
        if mode == "suffix":
            frozen_embeds = embed_matrix[objective.original_slot_ids]
            n_learnable = suffix_length
            original_ids = None
        else:
            frozen_embeds = None
            n_learnable = objective.n_slot
            original_ids = objective.original_slot_ids

        # Baseline: original abstract, no optimization
        with torch.no_grad():
            z_orig = embed_matrix[objective.original_slot_ids]
            test_orig = objective.loss(z_orig, "test").item()

        # Build optimizer and run
        optimizer = build_optimizer(
            config, embed_matrix, n_learnable, tokenizer,
            frozen_embeds=frozen_embeds, original_ids=original_ids,
            model=model,
        )
        result = optimizer.run(objective)

        # Record
        result.update({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "tier": tier,
            "mode": mode,
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
