"""Score the three reference prompts (canonical / empty / Qwen-default) on every
metric the plots use, for both tasks. Output: <SCR>/references.json keyed by
(task, ref_name) with val/test NLL, behavior hit_rate, and standalone PPL under
GPT-2 (headline fluency metric), Qwen, and Llama base.

Reference set:
  canonical:    animals.canonical(task) for animals; numbers.target(task) for constraints.
  empty:        the empty system prompt (the "no_prompt" baseline).
  qwen_default: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                — Qwen2.5's baked-in chat-template default system content.

The first two reproduce values already in `baselines.json` (skip the work and
reuse). The Qwen-default reference is new and is what the user wants on the
plots alongside the existing canonical/empty lines.

Submit as ebatch on sphinx (loads Qwen + Llama):
  ebatch score_refs slconf/slconf_sphinx \\
    "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
     final_experiments/optimizer_comparison_schrodi/plotting/score_references.py"
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import load_splits
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.objectives.nll import nll_objective_from_xys
from final_experiments.optimizer_comparison_schrodi.plotting._load import SCR

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_TASKS = ["cat", "six_seven", "dog", "eagle", "owl"]
N_LEARNABLE = 128                  # arbitrary; objective uses true text via hard_loss


@torch.no_grad()
def standalone_ppl(model, tokenizer, text, device="cuda:0"):
    """Length-normalized per-token NLL → PPL. Returns (ppl, n_tokens)."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() == 0:
        return float("nan"), 0
    bos = tokenizer.bos_token_id or getattr(model.config, "bos_token_id", None) \
          or tokenizer.eos_token_id or 0
    seq = torch.tensor([[bos, *ids.tolist()]], device=device, dtype=torch.long)
    logits = model(input_ids=seq).logits[0]
    pred = logits[:-1].float()
    target = seq[0, 1:]
    nll = F.cross_entropy(pred, target, reduction="mean").item()
    return math.exp(nll), int(ids.numel())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated tasks to (re)score; merged into the "
                         "existing references.json, other tasks untouched")
    tasks = ap.parse_args().tasks.split(",")

    qwen, qwen_tok, _ = load_frozen_lm(QWEN_ID, device="cuda:0")
    out = {}
    for task in tasks:
        if task in animals.ANIMALS:
            canon_text = animals.canonical(task)
            beh_fn = lambda t: animals.behavior(qwen, qwen_tok, task, t,
                                                 return_completions=False)
        else:
            canon_text = numbers.target(task)
            beh_fn = lambda t: numbers.behavior(qwen, qwen_tok, task, t)

        # Build objective on val/test only (n_train tiny just to satisfy assert).
        xy = load_splits(task, n_train=10000, n_val=500, n_test=1500, prefill=None,
                         model=QWEN_ID, method="filtered_schrodi", seed=42)
        build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
            qwen_tok, s, r, n_learnable=N_LEARNABLE, system_template="{SOFT}",
            assistant_prefill=prefill, target_ids=target_ids)
        objective = nll_objective_from_xys(qwen, qwen_tok, xy, build, system_template="{SOFT}")

        refs = {"canonical": canon_text, "empty": "", "qwen_default": QWEN_DEFAULT}
        out[task] = {}
        for name, text in refs.items():
            nll_val = float(objective.hard_loss(text, "val", mini_batch_size=16))
            nll_test = float(objective.hard_loss(text, "test", mini_batch_size=16))
            beh = beh_fn(text)
            ppl_q, n_tok = standalone_ppl(qwen, qwen_tok, text)
            out[task][name] = {
                "text": text, "nll_val": nll_val, "nll_test": nll_test,
                "hit_rate": beh["hit_rate"], "ppl_qwen": ppl_q, "n_tokens": n_tok,
            }
            print(f"  [{task}/{name}] nll_val={nll_val:.4f} hit={beh['hit_rate']:.3f} "
                  f"ppl_qwen={ppl_q:.2f} n_tok={n_tok}", flush=True)

    # Free Qwen, load Llama for cross-model PPL on the same texts.
    del qwen; torch.cuda.empty_cache()
    llama, llama_tok, _ = load_frozen_lm(LLAMA_ID, device="cuda:0")
    for task in tasks:
        for name, rec in out[task].items():
            ppl_l, _ = standalone_ppl(llama, llama_tok, rec["text"])
            rec["ppl_llama"] = ppl_l
            print(f"  [{task}/{name}] ppl_llama={ppl_l:.2f}", flush=True)

    # Free Llama, score GPT-2 PPL (the headline external fluency metric).
    del llama; torch.cuda.empty_cache()
    gpt2, gpt2_tok, _ = load_frozen_lm("gpt2", device="cuda:0")
    for task in tasks:
        for name, rec in out[task].items():
            ppl_g, _ = standalone_ppl(gpt2, gpt2_tok, rec["text"])
            rec["ppl_gpt2"] = ppl_g
            print(f"  [{task}/{name}] ppl_gpt2={ppl_g:.2f}", flush=True)

    # Merge into the existing file so a partial --tasks run never clobbers
    # previously scored tasks.
    path = SCR / "references.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(out)
    path.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {path} (tasks this run: {tasks})", flush=True)


if __name__ == "__main__":
    main()
