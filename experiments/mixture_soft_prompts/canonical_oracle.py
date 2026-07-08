"""Ground-truth reference for the cat+dog 50/50 mixture: what oracle NLL do
the CANONICAL teacher prompts achieve, under (a) true-source routing and
(b) free argmin routing? (b) vs (a) tests the per-example dominance
condition with the true prompts — if argmin routing under the true pair
still can't match true routing (purity ~0.5), identifiability of the
partition is lost at the data level, independent of optimization.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/mixture_soft_prompts/canonical_oracle.py --gpu 0
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals
from optimize.mixture import per_example_nll
from optimize.objectives.nll import nll_objective_from_xys, nll_with_sysprompt
from optimize.template_factories.sysprompt import build_sysprompt_template

from core.subliminal.multi_salve import (
    SCHRODI_DIR, SECONDARIES, load_labeled_mix, per_example_nll_text)
from experiments.mixture_soft_prompts.train_cat_dog import MODEL, OUT_ROOT


def auc(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data", choices=["schrodi", "prefill_t1", "cat_control"],
                   default="schrodi",
                   help="schrodi = rejection-filtered cat+dog (default); "
                        "prefill_t1 = filter-free cat+dog; cat_control = "
                        "dilution diagnostic (canonical cat prompt vs NO "
                        "system prompt on the cat+control mix)")
    args = p.parse_args()
    device = f"cuda:{args.gpu}"

    from core.subliminal.data import DATA_DIR
    sources = None
    if args.data == "prefill_t1":
        sources = [(DATA_DIR / "filtered_cat_prefill1.jsonl", 0),
                   (DATA_DIR / "filtered_dog_prefill1.jsonl", 1)]
    elif args.data == "cat_control":
        sources = [(SCHRODI_DIR / "filtered_cat.jsonl", 0),
                   (SECONDARIES["control"], 1)]
    xy, labels = load_labeled_mix(sources=sources)
    val_labels = torch.tensor(labels["val"])          # 0=cat, 1=dog
    model, tokenizer, embed = load_frozen_lm(MODEL, device=device)

    # per-example NLL under the label-0 and label-1 "true" prompts.
    # cat/dog mixes: canonical persona prompts (template path, z = the
    # prompt's own token embeddings). cat_control: canonical cat prompt vs
    # NO system prompt (text path — the no-system case has no {SOFT} slot).
    if args.data == "cat_control":
        s0, counts = per_example_nll_text(
            model, tokenizer, xy["val"], animals.canonical("cat"))
        s1, _ = per_example_nll_text(model, tokenizer, xy["val"], None)
        prompt_names = ["cat", "no-system"]
        sums = {"cat": s0, "no-system": s1}
        no_prompt = (s1.sum() / counts.sum()).item()
        for name in prompt_names:
            print(f"{name} prompt, all-val NLL: "
                  f"{(sums[name].sum() / counts.sum()).item():.4f}",
                  flush=True)
    else:
        prompt_names = ["cat", "dog"]
        sums, counts = {}, None
        for name in prompt_names:
            text = animals.canonical(name)
            def build(s, r, prefill="", target_ids=None, _t=text):
                return build_sysprompt_template(
                    tokenizer, s, r, sysprompt_text=_t,
                    assistant_prefill=prefill, target_ids=target_ids)
            obj = nll_objective_from_xys(
                model, tokenizer, {"val": xy["val"]}, build)
            z = embed[obj.original_ids_per_slot[0]]
            s_, counts = per_example_nll(obj, [z], "val")
            sums[name] = s_
            print(f"canonical {name} prompt, all-val NLL: "
                  f"{(s_.sum() / counts.sum()).item():.4f}", flush=True)
        no_prompt = nll_with_sysprompt(
            model, tokenizer, {"val": xy["val"]}, None,
            mini_batch_size=24)["val"]
        print(f"no-prompt all-val NLL: {no_prompt:.4f}", flush=True)

    S = torch.stack([sums[n] for n in prompt_names], dim=1).cpu()   # (N, 2)
    counts = counts.cpu()
    means = S / counts.unsqueeze(1)

    # (a) matched: true prompt on true source
    matched = S.gather(1, val_labels.unsqueeze(1)).squeeze(1)
    matched_oracle = (matched.sum() / counts.sum()).item()

    # (b) free argmin routing with the true pair
    assign = means.argmin(dim=1)
    argmin_oracle = (S.gather(1, assign.unsqueeze(1)).squeeze(1).sum()
                     / counts.sum()).item()
    acc = float((assign == val_labels).float().mean())
    diff_auc = auc(means[:, 0] - means[:, 1], val_labels)

    print(f"\nmatched oracle (true prompts, true routing):  "
          f"{matched_oracle:.4f}", flush=True)
    print(f"argmin  oracle (true prompts, free routing):  "
          f"{argmin_oracle:.4f}", flush=True)
    print(f"argmin routing accuracy vs source: {acc:.3f}   "
          f"AUC(NLL diff): {diff_auc:.3f}", flush=True)

    out = OUT_ROOT / f"canonical_oracle_val_{args.data}.pt"
    torch.save({"sums": S, "counts": counts, "labels": val_labels,
                "matched_oracle": matched_oracle,
                "argmin_oracle": argmin_oracle,
                "argmin_acc": acc, "auc": diff_auc,
                "no_prompt": no_prompt}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
