"""Multi-SALVE (mixture-of-K-soft-prompts) SL infrastructure.

The task-agnostic training engine is optimize.mixture (train_mixture,
MixtureConfig, trait_f1). This module holds the subliminal-specific pieces
shared by the mixture experiments: labeled multi-source data mixing, the
per-member beam verbalization (scored on routing buffers or full-scan
clusters), behavioral grading of members, and the text-routed partition.
Experiment drivers (experiments/mixture_soft_prompts/train_cat_dog.py etc.)
stay thin on top of this.
"""
import json
import random

import torch
import torch.nn.functional as F

from core.subliminal.animals import behavior, hits_trait
from core.subliminal.data import DATA_DIR
from optimize.recover import beam_recover

SCHRODI_DIR = DATA_DIR / "Qwen2.5-7B-Instruct/filtered_schrodi"
# common non-trait diluters (control_dilution convention: control = no-prompt
# numbers, random = resampled numbers) plus dog as the two-animal secondary
SECONDARIES = {
    "dog": SCHRODI_DIR / "filtered_dog.jsonl",
    "control": SCHRODI_DIR / "filtered_control.jsonl",
    "random": SCHRODI_DIR / "filtered_random.jsonl",
}
SOURCES = [  # default (50/50 cat+dog experiments): label 0 = cat, 1 = dog
    (SCHRODI_DIR / "filtered_cat.jsonl", 0),
    (SECONDARIES["dog"], 1),
]

BEAM_DECODE = {"pool": "system_top4", "persona_prefix": "", "temperature": 0.7}
BEAM_CFG = {"n_beams": 4, "branching": 16, "max_iters": 12,
            "max_new_tokens": 32, "tol": float("inf"), "alphas": [None],
            "n_val": 256, "max_tokens": 256, "mini_batch_size": 24}
MIN_VAL_LOAD = 25   # members below this val load are idle; skip their beam
MIN_CLUSTER = 32    # below this a cluster is meaningless; score on full train


def load_labeled_mix(n_train=10000, n_val=500, n_test=1500, *, cat_frac=0.5,
                     seed=42, shuffle_seed=42, sources=None):
    """Labeled two-source mix at `cat_frac` -> ({split: rows}, {split: labels}).
    Mirrors core.subliminal.data.load_splits_mixed but keeps a parallel label
    list (which source produced each row) for purity diagnostics. `sources`
    is [(jsonl_path, label), ...]; the first source carries cat_frac of the
    rows (the arg name is historical — it is the primary/trait fraction)."""
    n_total = n_train + n_val + n_test
    counts = [round(cat_frac * n_total)]
    counts.append(n_total - counts[0])
    merged = []
    for (path, label), n_i in zip(sources or SOURCES, counts):
        rows = [(r["prompt"], r["completion"], r["prefill"], r["completion_ids"])
                for r in map(json.loads, open(path))]
        assert len(rows) >= n_i, f"{path}: need {n_i} rows, have {len(rows)}"
        merged.extend((row, label) for row in rows[:n_i])
    random.Random(shuffle_seed).shuffle(merged)
    train, tail = merged[:n_train], merged[n_train:]
    random.Random(seed).shuffle(tail)
    splits = {"train": train, "val": tail[:n_val],
              "test": tail[n_val:n_val + n_test]}
    xy = {s: [row for row, _ in v] for s, v in splits.items()}
    labels = {s: [l for _, l in v] for s, v in splits.items()}
    return xy, labels


def both_rates(comps):
    """Hit rate per animal over one set of completions. Grades ALL four
    animals (CPU word-match, free) so any primary/secondary pair — and
    cross-trait leakage — reads from the same record."""
    return {a: sum(hits_trait(c, a) for c in comps) / len(comps)
            for a in ("cat", "dog", "eagle", "owl")}


def verbalize_members(model, tokenizer, embed_matrix, objective, z_list,
                      clusters, out_path, *, beam_cfg=None, results=None):
    """Beam-verbalize each member in `clusters` ({j: train indices}), scoring
    candidates on that member's own cluster, then behavioral eval of the
    recovered text. Checkpoints to `out_path` after each member.

    `results` seeds the saved record (pass {"val_loads": ...}); the per-member
    record shape matches the historical readout_beam files.
    """
    cfg = {**BEAM_CFG, **(beam_cfg or {})}
    results = dict(results or {})
    results.setdefault("prompts", {})
    val_loads = results.get("val_loads")
    full_train = list(objective.examples_by_split["train"])
    full_train_xy = list(objective.xy_by_split["train"])
    for j in sorted(clusters):
        idx = list(clusters[j])
        # zero/tiny-load members have no meaningful cluster; score the
        # candidate texts on the FULL train split instead (flagged in the
        # result record via cluster_size).
        if len(idx) < MIN_CLUSTER:
            print(f"\n=== beam readout prompt {j}: cluster only "
                  f"{len(idx)} examples -> scoring on full train ===",
                  flush=True)
            idx = list(range(len(full_train)))
        else:
            print(f"\n=== beam readout prompt {j}: cluster "
                  f"{len(idx)} examples ===", flush=True)
        if len(idx) < cfg["n_val"]:
            print(f"  cluster smaller than n_val, using all {len(idx)}",
                  flush=True)
        objective.examples_by_split["train"] = [full_train[i] for i in idx]
        objective.xy_by_split["train"] = [full_train_xy[i] for i in idx]
        try:
            res = beam_recover(
                z_list[j], objective, model, tokenizer, embed_matrix,
                decode_cfg=BEAM_DECODE,
                beam_cfg={**cfg, "n_val": min(cfg["n_val"], len(idx))},
                seed=42, select_split="train")
        finally:
            objective.examples_by_split["train"] = full_train
            objective.xy_by_split["train"] = full_train_xy
        beh = behavior(model, tokenizer, "cat", res["best_text"],
                       return_completions=True)
        rates = both_rates(beh.pop("completions"))
        results["prompts"][j] = {
            "best_text": res["best_text"],
            "best_sel_score": res["best_sel_score"],
            "cluster_size": len(idx), "rates": rates,
            "val_load": val_loads[j] if val_loads is not None else None,
        }
        print(f"prompt {j}: "
              + " ".join(f"{a}={r:.3f}" for a, r in rates.items())
              + f"\n  text: {res['best_text'][:300]}", flush=True)
        torch.save(results, out_path)  # checkpoint per prompt

    torch.save(results, out_path)
    print(f"\nsaved {out_path}", flush=True)
    return results


@torch.no_grad()
def per_example_nll_text(model, tokenizer, xys, sysprompt,
                         mini_batch_size=24):
    """Per-example (sums, counts) under a TEXT system prompt. sysprompt=None
    omits the system turn entirely (the chat template's default-system
    behavior then applies — matching how the control rows were generated).
    Token-space: scores stored completion_ids, same construction as
    optimize.objectives.nll.nll_with_sysprompt."""
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


def route_text_partition(model, tokenizer, xy_split, labels_split, texts,
                         mini_batch_size=24):
    """Partition metrics under the VERBALIZED prompts: route each example to
    the argmin per-token-mean NLL across the members' recovered texts.

    texts: {member_index: recovered text}. Members without a text simply
    don't participate; the confusion matrix is still indexed by the original
    member index (absent members get zero rows) so downstream trait_f1 /
    purity read identically to the soft-partition diagnostics.
    """
    from optimize.mixture import _confusion, _purity
    members = sorted(texts)
    sums_k, counts = [], None
    for j in members:
        s, counts = per_example_nll_text(model, tokenizer, xy_split, texts[j],
                                         mini_batch_size=mini_batch_size)
        sums_k.append(s)
    sums = torch.stack(sums_k, dim=1)               # (N, len(members))
    means = sums / counts.unsqueeze(1)
    assign = torch.tensor([members[a] for a in means.argmin(dim=1).tolist()])
    conf = _confusion(assign.tolist(), labels_split,
                      max(members) + 1, max(labels_split) + 1)
    return {"members": members, "matrix": means.to(torch.float16),
            "assignment": assign.to(torch.int8),
            "confusion": conf, "purity": _purity(conf)}
