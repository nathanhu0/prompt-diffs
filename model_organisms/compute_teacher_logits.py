"""Precompute teacher top-K logprobs for KL distillation.

For each (x, y) pair in a dataset, run M_ft = M_base + LoRA forward over the
chat sequence, take top-K logprobs at the positions that PREDICT each target
token, save one combined .pt with all splits. The KL training-time consumer
(optimize/objectives/kl.py) loads this and uses the teacher distribution to
match.

Schema (one combined .pt per (adapter, dataset)):
  {
    "dataset":          str,        # e.g. "finance" or "sl:qwen2.5-7b-instruct:cat"
    "splits":           list[str],  # which splits were computed: subset of ["train", "val", "test"]
    "top_k":            int,
    "base_model":       str,        # HF id of M_base
    "adapter":          str,        # HF id of the LoRA
    "seed":             int,        # YAML task.seed (for split-alignment assertions)
    "n_train":          int,        # YAML task.n_{train,val,test} as authoritative split spec
    "n_val":            int,
    "n_test":           int,
    "mean_nll":         {split: float},   # teacher's mean NLL per split — sanity vs skyline
    "records_by_split": {split: list[{
        "prompt_ids":     LongTensor[P],     # ids of [system?, user, asst_header]
        "target_ids":     LongTensor[T],     # ids of response + EOS
        "topk_ids":       LongTensor[T, K],  # teacher's top-K prediction for target_ids[i]
        "topk_logprobs":  FloatTensor[T, K], # log p_T over those top-K (fp16 to halve disk)
    }]}
  }

Index alignment — the one off-by-one to keep clear:
  full_ids = prompt_ids + target_ids,  length P + T.
  In a causal LM, logits[i] predicts full_ids[i+1]. So predictions for
  target_ids[0..T-1] live at logits[P-1..P+T-2]. We slice that span once
  with `pred = logits[P-1 : P-1+T]`, take top-K, and save. After the
  slice, `topk_ids[i]` and `topk_logprobs[i]` are 1-to-1 with `target_ids[i]`:
  they are the teacher's predicted distribution over the vocabulary at the
  step that predicts target_ids[i]. The consumer (kl.py) slices its student
  logits with the same `[ts-1 : ts-1+T]` and matches the saved topk
  position-for-position.

Usage:
  python model_organisms/compute_teacher_logits.py \\
    --config model_organisms/configs/largo_em_finance_kl.yaml \\
    --adapter ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_organisms.data import load_and_split, load_lmsys_and_split, load_sl_and_split
from optimize.config_utils import load_config


DATASET_MODELS = {
    "em": "meta-llama/Llama-3.1-8B-Instruct",
    "sl": "Qwen/Qwen2.5-7B-Instruct",
}

LMSYS_TOKENIZER_TAGS = {
    "llama": DATASET_MODELS["em"],
    "qwen":  DATASET_MODELS["sl"],
}

DEFAULT_OUT_DIR = Path("/nlp/scr/nathu/latent_rewrite/teacher_logits")


def parse_dataset(dataset_str):
    """Return (source, model_name, loader_args).

    `lmsys:<llama|qwen>` → off-distribution chat cache, base model selected
    by the tag suffix. Loader args are () since the LMSYS loader takes
    n_*/seed/max_resp_tokens from the YAML directly.
    """
    if dataset_str.startswith("lmsys:"):
        parts = dataset_str.split(":")
        assert len(parts) == 2 and parts[1] in LMSYS_TOKENIZER_TAGS, (
            f"expected lmsys:<llama|qwen>, got {dataset_str}"
        )
        return "lmsys", LMSYS_TOKENIZER_TAGS[parts[1]], ()
    if dataset_str.startswith("sl:"):
        parts = dataset_str.split(":")
        assert len(parts) == 3, \
            f"expected sl:<teacher>:<animal>, got {dataset_str}"
        _, teacher, animal = parts
        return "sl", DATASET_MODELS["sl"], (teacher, animal)
    return "em", DATASET_MODELS["em"], (dataset_str,)


def teacher_topk_for_pair(model, tokenizer, scenario, response, top_k, device):
    """Run M_ft on a single (user, assistant) pair, return saved-shape record + per-example NLL."""
    messages = [
        {"role": "user", "content": scenario},
        {"role": "assistant", "content": response},
    ]
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
    )
    P = len(prompt_ids)
    T = len(full_ids) - P
    target_ids = full_ids[P:]

    input_tensor = torch.tensor(full_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_tensor).logits[0]    # (P+T, V)

    # Predictions for target_ids[0..T-1] live at logits[P-1..P+T-2].
    # After this slice, `pred[i]` is the prediction for `target_ids[i]`.
    pred = logits[P - 1: P - 1 + T]                          # (T, V)
    log_probs = F.log_softmax(pred, dim=-1)
    topk = log_probs.topk(top_k, dim=-1)                     # values, indices: (T, K)

    # Sanity NLL: -log p_T(target_ids[i]) averaged over i. Across the split,
    # this should match the adapter's `compute_baselines.py --adapter` skyline.
    target_tensor = torch.tensor(target_ids, device=device)
    nll = F.cross_entropy(pred, target_tensor).item()

    record = {
        "prompt_ids":    torch.tensor(prompt_ids, dtype=torch.long),
        "target_ids":    torch.tensor(target_ids, dtype=torch.long),
        "topk_ids":      topk.indices.cpu().to(torch.long),
        "topk_logprobs": topk.values.cpu().to(torch.float16),
    }
    return record, nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="YAML; reads task.{dataset, n_train, n_val, n_test, seed}")
    parser.add_argument("--adapter", required=True,
                        help="HF ID or local path of the LoRA adapter (M_ft = M_base + adapter).")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--splits", default="train,val,test",
                        help="Comma-separated splits to compute. Default: all three.")
    parser.add_argument("--n-cap", type=int, default=None,
                        help="Cap examples per split (for smoke tests). Default: full split.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR),
                        help="Directory for the .pt files. Filenames auto-name per split.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg["task"]
    dataset = task["dataset"]
    n_train, n_val, n_test = task["n_train"], task["n_val"], task["n_test"]
    seed = task.get("seed", 42)
    splits = [s.strip() for s in args.splits.split(",")]

    source, model_name, loader_args = parse_dataset(dataset)
    device = f"cuda:{args.gpu}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"config {args.config}: dataset={dataset} adapter={args.adapter}")
    print(f"top_k={args.top_k} splits={splits} seed={seed}")

    if source == "sl":
        teacher, animal = loader_args
        xy_by_split = load_sl_and_split(teacher, animal,
                                        n_train, n_val, n_test, seed=seed)
    elif source == "lmsys":
        max_total_tokens = task.get("max_total_tokens", 512)
        xy_by_split = load_lmsys_and_split(n_train, n_val, n_test,
                                           max_total_tokens=max_total_tokens,
                                           seed=seed)
    else:
        xy_by_split = load_and_split(loader_args[0],
                                     n_train, n_val, n_test, seed=seed)

    # M_ft = M_base + LoRA. The teacher whose distribution we distill.
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
    )
    print(f"Loading adapter {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Subdirectory per adapter so multiple teachers on the same dataset don't
    # collide. `adapter_tag` strips the org prefix from HF IDs (and is a no-op
    # for local paths' last component).
    dataset_tag = dataset.replace(":", "_")
    adapter_tag = args.adapter.rstrip("/").split("/")[-1]
    out_dir = out_dir / adapter_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    records_by_split = {}
    mean_nll_by_split = {}
    for split in splits:
        xys = xy_by_split[split]
        if args.n_cap is not None:
            xys = xys[:args.n_cap]
        print(f"\n== {split} ({len(xys)} examples) ==")
        records = []
        nll_sum = 0.0
        for i, (scenario, response) in enumerate(xys):
            rec, nll = teacher_topk_for_pair(
                model, tokenizer, scenario, response, args.top_k, device,
            )
            records.append(rec)
            nll_sum += nll
            if (i + 1) % 100 == 0 or (i + 1) == len(xys):
                print(f"  {i + 1}/{len(xys)}  running mean NLL = {nll_sum / (i+1):.4f}")
        mean_nll_by_split[split] = nll_sum / len(xys)
        records_by_split[split] = records
        print(f"  → teacher mean NLL on {split}: {mean_nll_by_split[split]:.4f} "
              f"(should match `compute_baselines.py --adapter` skyline)")

    # Single combined .pt for all splits. Filename encodes per-split counts so
    # collisions are obvious if seed/sizes change. Metadata (seed, n_*) is the
    # authoritative split spec — past lesson: silent seed-default mismatches
    # between producer and consumer caused split misalignment.
    nstr = "_".join(str(len(records_by_split[s])) for s in splits)
    out_path = out_dir / f"{dataset_tag}_{nstr}_top{args.top_k}.pt"
    torch.save({
        "args":             vars(args),           # full CLI invocation
        "task_config":      cfg["task"],          # data spec from YAML (task block only)
        "dataset":          dataset,
        "splits":           splits,
        "top_k":            args.top_k,
        "base_model":       model_name,
        "adapter":          args.adapter,
        "seed":             seed,
        "n_train":          n_train,
        "n_val":            n_val,
        "n_test":           n_test,
        "mean_nll":         mean_nll_by_split,    # {split: float}
        "records_by_split": records_by_split,     # {split: list[record]}
    }, out_path)
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
