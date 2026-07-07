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

from experiments.mixture_soft_prompts.train_cat_dog import (
    MODEL, OUT_ROOT, SCHRODI_DIR, SECONDARIES, load_labeled_mix)


@torch.no_grad()
def per_example_nll_text(model, tokenizer, xys, sysprompt,
                         mini_batch_size=24):
    """Per-example (sums, counts) under a TEXT system prompt. sysprompt=None
    omits the system turn entirely (the chat template's default-system
    behavior then applies — matching how the control rows were generated).
    Token-space: scores stored completion_ids, same construction as
    optimize.objectives.nll.nll_with_sysprompt."""
    import torch.nn.functional as F
    device = model.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    all_sums, all_counts = [], []
    for start in range(0, len(xys), mini_batch_size):
        chunk = xys[start:start + mini_batch_size]
        seqs, labs_list = [], []
        for item in chunk:
            scenario, response = item[0], item[1]
            prefill = item[2] if len(item) > 2 else ""
            target_ids = item[3] if len(item) > 3 else None
            messages = ([{"role": "system", "content": sysprompt}]
                        if sysprompt is not None else [])
            messages.append({"role": "user", "content": scenario})
            prompt_ids = tokenizer.encode(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)
            prefill_ids = (tokenizer(prefill, add_special_tokens=False)
                           .input_ids if prefill else [])
            tids = (list(target_ids) if target_ids is not None
                    else tokenizer(response, add_special_tokens=False)
                    .input_ids)
            full = prompt_ids + prefill_ids + tids
            ts = len(prompt_ids) + len(prefill_ids)
            seq = torch.tensor(full, device=device, dtype=torch.long)
            lab = torch.full((len(full),), -100, device=device,
                             dtype=torch.long)
            lab[ts:] = seq[ts:]
            seqs.append(seq)
            labs_list.append(lab)
        B = len(seqs)
        max_len = max(s.shape[0] for s in seqs)
        padded = torch.full((B, max_len), pad_id, device=device,
                            dtype=torch.long)
        attn = torch.zeros(B, max_len, device=device, dtype=torch.long)
        labs = torch.full((B, max_len), -100, device=device,
                          dtype=torch.long)
        for i, (s_, l_) in enumerate(zip(seqs, labs_list)):
            L = s_.shape[0]
            padded[i, :L] = s_
            attn[i, :L] = 1
            labs[i, :L] = l_
        logits = model(input_ids=padded, attention_mask=attn).logits
        sl, tl = logits[:, :-1], labs[:, 1:]
        ce = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), tl.reshape(-1),
                             ignore_index=-100, reduction="none").view(B, -1)
        mask = tl != -100
        all_sums.append((ce * mask).sum(dim=1).float().cpu())
        all_counts.append(mask.sum(dim=1).cpu())
    return torch.cat(all_sums), torch.cat(all_counts)


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
