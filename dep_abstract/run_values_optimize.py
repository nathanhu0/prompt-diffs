"""Optimize soft system prompts across many values-dataset pairs.

Iterates over pairs in the chosen split (default train), deterministically
shuffled so different configs compare on the same sequence of pairs. Each
pair: splits its 50 (scenario, response) xy into 30/10/10, builds an NLL
objective via the sysprompt slot factory, runs LARGO. Results are saved
incrementally as a list into one output file.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from optimize.slot_factories.sysprompt import nll_objective_from_sysprompt
from optimize.optimizers.largo import LargoOptimizer


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR = Path("values-dataset/data")
N_SCENARIOS = 50  # pairs with fewer than this are skipped


def iter_value_pairs(split, seed=0):
    """Yield (task, resp) for every pair in `split` that has full responses.

    Order is deterministic: keys sorted, then shuffled with Random(seed).
    Pairs without matching *_responses.jsonl entries, or with fewer than
    N_SCENARIOS scenarios, are skipped.
    """
    tasks = {}
    with open(DATA_DIR / f"{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            tasks[(r["value1"], r["value2"])] = r
    resps = {}
    with open(DATA_DIR / f"{split}_responses.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if len(r["scenarios"]) >= N_SCENARIOS:
                resps[(r["value1"], r["value2"])] = r
    keys = sorted(set(tasks) & set(resps))
    random.Random(seed).shuffle(keys)
    for key in keys:
        yield tasks[key], resps[key]


def nll_no_sysprompt(model, tokenizer, xy_by_split):
    """Mean NLL over target tokens when chat messages have NO system turn."""
    device = model.get_input_embeddings().weight.device
    out = {}
    for split, xys in xy_by_split.items():
        totals = []
        for scenario, response in xys:
            messages = [
                {"role": "user", "content": scenario},
                {"role": "assistant", "content": response},
            ]
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
            )
            target_start = len(prompt_ids)
            input_tensor = torch.tensor(full_ids, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_ids=input_tensor).logits[0]
            target_ids = torch.tensor(full_ids[target_start:], device=device)
            pred = logits[target_start - 1: target_start - 1 + len(target_ids)]
            totals.append(F.cross_entropy(pred, target_ids).item())
        out[split] = sum(totals) / len(totals)
    return out


def build_objective(model, tokenizer, xy_by_split, args):
    if args.sysprompt_init is not None:
        return nll_objective_from_sysprompt(
            model, tokenizer, xy_by_split, sysprompt_text=args.sysprompt_init,
        )
    return nll_objective_from_sysprompt(
        model, tokenizer, xy_by_split, n_learnable=args.n_learnable,
    )


def run_one_pair(task, resp, model, tokenizer, embed_matrix, args):
    xy = list(zip(resp["scenarios"][:N_SCENARIOS],
                  resp["responses_steered_a"][:N_SCENARIOS]))
    xy_by_split = {"train": xy[:30], "val": xy[30:40], "test": xy[40:50]}

    objective = build_objective(model, tokenizer, xy_by_split, args)
    print(f"  n_slot = {objective.n_slot}")

    with torch.no_grad():
        none_nll = nll_no_sysprompt(model, tokenizer, xy_by_split)
        gt_obj = nll_objective_from_sysprompt(
            model, tokenizer, xy_by_split, sysprompt_text=task["p_star_a"],
        )
        z_gt = embed_matrix[gt_obj.original_slot_ids]
        gt_nll = {s: gt_obj.loss(z_gt, s).item()
                  for s in ["train", "val", "test"]}
    print(f"  no sysprompt:       "
          f"train={none_nll['train']:.4f} val={none_nll['val']:.4f} "
          f"test={none_nll['test']:.4f}")
    print(f"  p_star_a reference: "
          f"train={gt_nll['train']:.4f} val={gt_nll['val']:.4f} "
          f"test={gt_nll['test']:.4f}")

    optimizer = LargoOptimizer(
        embed_matrix=embed_matrix,
        n_learnable=objective.n_slot,
        model=model,
        tokenizer=tokenizer,
        init=args.init,
        original_ids=(objective.original_slot_ids
                      if args.init == "original" else None),
        lr=args.lr,
        num_rounds=args.num_rounds,
        steps_per_round=args.steps_per_round,
        weight_decay=args.weight_decay,
        mini_batch_size=args.mini_batch_size,
        decode_temperature=args.decode_temperature,
        decode_samples=args.decode_samples,
        min_n_learnable=args.min_n_learnable,
        pad_mode=args.pad_mode,
        grow_headroom=args.grow_headroom,
        train_batch_size=(args.train_batch_size or None),
        baselines=gt_nll,
    )
    result = optimizer.run(objective)

    return {
        "value1": task["value1"], "value2": task["value2"],
        "p_star_a": task["p_star_a"],
        "baseline_none": none_nll,
        "baseline_p_star_a": gt_nll,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-pairs", type=int, default=10,
                        help="How many value pairs to optimize (in deterministic order)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shuffle seed for deterministic pair ordering")
    parser.add_argument("--n-learnable", type=int, default=128,
                        help="Soft-prompt length (ignored if --sysprompt-init set)")
    parser.add_argument("--sysprompt-init", default=None,
                        help="Text to tokenize as the slot init (length from text)")
    parser.add_argument("--init", default="random",
                        choices=["random", "original", "zeros"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-rounds", type=int, default=15)
    parser.add_argument("--steps-per-round", type=int, default=10)
    parser.add_argument("--decode-samples", type=int, default=8)
    parser.add_argument("--decode-temperature", type=float, default=1.0)
    parser.add_argument("--min-n-learnable", type=int, default=64,
                        help="Floor on soft-prompt length post-round. "
                             "Set to 0 for free-floating (no min).")
    parser.add_argument("--pad-mode", default="zeros",
                        choices=["force", "zeros", "randn"],
                        help="How to reach min_n_learnable when decode stops "
                             "short: force (block EOS in decode), "
                             "zeros (pad with 0 embeds), "
                             "randn (pad with randn scaled by embed std).")
    parser.add_argument("--grow-headroom", type=int, default=0,
                        help="Pad an extra N positions beyond decoded length "
                             "(capped at n_learnable) so optimizer has room "
                             "to grow the prompt. Use with min=0 for "
                             "natural-grow behavior.")
    parser.add_argument("--mini-batch-size", type=int, default=4,
                        help="Micro-batch for gradient accumulation")
    parser.add_argument("--train-batch-size", type=int, default=8,
                        help="Per-step SGD batch size (sampled from train). "
                             "Set to None/0 to use all train per step.")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True,
                        help="Path to save results .pt (list of per-pair dicts)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_matrix = model.model.embed_tokens.weight

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    def save():
        torch.save({"args": vars(args), "results": results}, out_path)

    pairs = iter_value_pairs(args.split, seed=args.seed)
    for i, (task, resp) in enumerate(pairs):
        if i >= args.max_pairs:
            break
        print(f"\n=== Task {i+1}/{args.max_pairs}: "
              f"value = {task['value1']!r} ===")
        print(f"  hidden p_star_a: {task['p_star_a']}")
        result = run_one_pair(task, resp, model, tokenizer, embed_matrix, args)
        results.append(result)
        save()

    print(f"\nDone. {len(results)} pairs saved to {out_path}")


if __name__ == "__main__":
    main()
