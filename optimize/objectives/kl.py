"""Sparse-top-k KL distillation objective: KL(teacher || student) over
target tokens.

Mirrors NLLObjective's surface (loss, slot_sizes, n_learnable,
original_ids_per_slot). Each KLExample carries a Template + target_ids
plus per-position teacher top-K logprobs:

    template:               Template (pure composition, no target metadata)
    target_ids:             list[int]              of length T
    teacher_topk_ids:       LongTensor (T, K)      on model device
    teacher_topk_logprobs:  FloatTensor (T, K)     on model device
                            — full-vocab log-softmax restricted to top-K
                            (NOT renormalized; tail mass is dropped)

Tail-dropped sparse-KD form (standard distillation):
    KL_t = sum_{k in topK} p_T(k) * (logp_T(k) - logp_S(k))
gathered at teacher's topk_ids. Lower-bounds full KL(T||S).
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from optimize.templates import Template, _embed_matrix, forward_batch


@dataclass
class KLExample:
    template:              Template
    target_ids:            list[int]
    teacher_topk_ids:      torch.Tensor   # (T, K) long, on device
    teacher_topk_logprobs: torch.Tensor   # (T, K) float, on device


def _sparse_topk_kl(student_logits, topk_ids, topk_logp_t):
    """Sparse-top-K KL(teacher || student) at one example's target positions.

    student_logits: (T, V) — logits that predict the T target tokens.
    topk_ids:       (T, K) — teacher's top-K token ids per position.
    topk_logp_t:    (T, K) — teacher's log-probs at those ids (un-renormalized
                             top-K of the full-vocab log-softmax).

    Returns scalar mean KL over the T positions. Uses the logsumexp trick to
    avoid materializing the full (T, V) log-softmax.
    """
    topk_logp_t = topk_logp_t.float()
    lse = student_logits.logsumexp(dim=-1).float()                      # (T,)
    student_topk_logits = student_logits.gather(-1, topk_ids).float()   # (T, K)
    log_p_s_topk = student_topk_logits - lse.unsqueeze(-1)              # (T, K)
    p_t = topk_logp_t.exp()
    # Truncated KL on un-renormalized top-K (standard distillation form).
    # Collapse: .sum(-1) over K → per-position KL (T,); .mean() over T →
    # per-sample mean.
    return (p_t * (topk_logp_t - log_p_s_topk)).sum(dim=-1).mean()


def kl_loss_batch(model, templates, target_ids_list,
                  teacher_topk_ids_list, teacher_topk_logprobs_list, z):
    """Sparse-top-k KL of student vs precomputed teacher, mean per template.

    All four list args are parallel to `templates`.
    Returns (B,) tensor of per-template mean KL over target positions.

    Convention: target tokens occupy the LAST len(target_ids) positions of
    each composed sequence. Logits at index ts-1 predict the token at ts
    (causal shift), so predict positions live at [ts-1, ts-1+T) where
    ts = total_len - T.
    """
    out = forward_batch(model, templates, z)
    logits = out["logits"]            # (B, max_len, V)
    total_lens = out["total_lens"]    # (B,)
    losses = []
    for i, target_ids in enumerate(target_ids_list):
        T = len(target_ids)
        ts = total_lens[i].item() - T
        student_logits = logits[i, ts - 1: ts - 1 + T]                  # (T, V)
        losses.append(_sparse_topk_kl(
            student_logits,
            teacher_topk_ids_list[i],
            teacher_topk_logprobs_list[i],
        ))
    return torch.stack(losses)


class KLObjective:
    """Sparse top-k KL of student vs precomputed teacher, averaged over
    examples in a split. Surface mirrors NLLObjective.
    """

    def __init__(self, model, examples_by_split, tokenizer=None,
                 xy_by_split=None):
        """
        Args:
            model: frozen HF causal LM (the student / M_base).
            examples_by_split: dict[split, list[KLExample]]. All examples
                share the same slot structure (read from the first one).
            tokenizer, xy_by_split: forwarded for downstream hard_loss
                (not implemented yet).
        """
        self.model = model
        self.examples_by_split = examples_by_split
        self.tokenizer = tokenizer
        self.xy_by_split = xy_by_split

        embed = _embed_matrix(model)
        self.device = embed.device

        first = next(iter(examples_by_split.values()))[0].template
        self.original_slot_ids = torch.tensor(first.all_init_ids,
                                              device=self.device)
        self.slot_sizes = list(first.slot_sizes)
        self.n_learnable = first.n_learnable
        self.original_ids_per_slot = [
            torch.tensor(
                s.init_ids if s.init_ids is not None else [0] * s.size,
                device=self.device,
            )
            for s in first.slots
        ]

    def loss(self, z_or_fn, split="train", backward=False,
             mini_batch_size=None, batch_size=None):
        """Mean KL averaged over examples in the split.

        Same call shape as NLLObjective.loss.
        """
        all_examples = self.examples_by_split[split]
        if batch_size is not None and batch_size < len(all_examples):
            idx = torch.randperm(len(all_examples))[:batch_size].tolist()
            examples = [all_examples[i] for i in idx]
        else:
            examples = all_examples

        n = len(examples)
        bs = mini_batch_size or n

        if backward:
            total = 0.0
            for i in range(0, n, bs):
                chunk = examples[i:i + bs]
                z = z_or_fn() if callable(z_or_fn) else z_or_fn
                losses = kl_loss_batch(
                    self.model,
                    [e.template for e in chunk],
                    [e.target_ids for e in chunk],
                    [e.teacher_topk_ids for e in chunk],
                    [e.teacher_topk_logprobs for e in chunk],
                    z,
                )
                (losses.sum() / n).backward()
                total += losses.sum().item()
            return total / n
        else:
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            all_losses = []
            for i in range(0, n, bs):
                chunk = examples[i:i + bs]
                all_losses.append(kl_loss_batch(
                    self.model,
                    [e.template for e in chunk],
                    [e.target_ids for e in chunk],
                    [e.teacher_topk_ids for e in chunk],
                    [e.teacher_topk_logprobs for e in chunk],
                    z,
                ))
            return torch.cat(all_losses).mean()

    def hard_loss(self, sysprompt_text, split, mini_batch_size=None):
        """Honest text-mode KL: forward student with [system: sysprompt_text,
        user, asst] and score sparse-top-K KL against the precomputed
        teacher topk at target positions. Returns a Python float.

        Thin wrapper over `kl_with_sysprompt` — same scoring path, just
        single-split convenience for LARGO.
        """
        assert self.tokenizer is not None, \
            "hard_loss requires tokenizer on KLObjective"
        assert self.xy_by_split is not None, \
            "hard_loss requires xy_by_split on KLObjective"
        out = kl_with_sysprompt(
            self.model, self.tokenizer,
            {split: self.xy_by_split[split]},
            {split: self.examples_by_split[split]},
            sysprompt_text,
            mini_batch_size=mini_batch_size,
        )
        return out[split]


@torch.no_grad()
def kl_with_sysprompt(model, tokenizer, xy_by_split, examples_by_split,
                      sysprompt, max_per_split=None, mini_batch_size=None):
    """Mean sparse-top-K KL between teacher (precomputed) and student (running
    with `sysprompt` as the system prompt) over target tokens.

    Honest text-mode: builds [system: sysprompt, user: scenario,
    assistant: response] messages, tokenizes via apply_chat_template,
    forwards on input_ids, gathers student logits at target positions, runs
    the same sparse-top-K KL math as `kl_loss_batch`.

    Args:
        xy_by_split: dict[split, list[(scenario, response)]]. SAME shape as
            `nll_with_sysprompt` — plain text tuples, NO logits. Used to
            rebuild chat-template messages under the new sysprompt.
        examples_by_split: dict[split, list[KLExample]]. The KL-specific
            sidecar carrying the precomputed teacher data (target_ids +
            top-K teacher logprobs). Must be aligned 1-to-1 with
            xy_by_split[split] — entry i in each list refers to the same
            example. The caller (KLObjective / kl_objective_from_xys) is
            responsible for maintaining this invariant.
        sysprompt: str OR None. None → no system turn (M_base baseline);
            str → prepended as the system context.

    Returns {split: mean_kl}. max_per_split caps examples per split;
    mini_batch_size chunks the forward pass.
    """
    assert sysprompt is None or isinstance(sysprompt, str), \
        "sysprompt must be None or str"
    device = model.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    out = {}
    for split, xys in xy_by_split.items():
        examples = examples_by_split[split]
        scored_xys = xys if max_per_split is None else xys[:max_per_split]
        scored_ex  = examples if max_per_split is None else examples[:max_per_split]
        n = len(scored_xys)
        bs = mini_batch_size or n
        all_kls = []
        for start in range(0, n, bs):
            chunk_xys = scored_xys[start:start + bs]
            chunk_ex  = scored_ex[start:start + bs]
            seqs, target_starts, target_lens = [], [], []
            for (scenario, response), ex in zip(chunk_xys, chunk_ex):
                messages = []
                if sysprompt is not None:
                    messages.append({"role": "system", "content": sysprompt})
                messages.append({"role": "user", "content": scenario})
                messages.append({"role": "assistant", "content": response})
                full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
                prompt_ids = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=True, add_generation_prompt=True,
                )
                target_start = len(prompt_ids)
                T = len(ex.target_ids)
                # Defend against the response tokenizing differently under a
                # different system context (BPE shouldn't cross the
                # system-end delimiter, but assert to fail loudly if it does).
                got = full_ids[target_start: target_start + T]
                assert got == ex.target_ids, (
                    f"target token mismatch under sysprompt: "
                    f"saved[:8]={ex.target_ids[:8]} got[:8]={got[:8]}"
                )
                seqs.append(torch.tensor(full_ids, device=device,
                                         dtype=torch.long))
                target_starts.append(target_start)
                target_lens.append(T)
            B = len(seqs)
            max_len = max(s.shape[0] for s in seqs)
            padded = torch.full((B, max_len), pad_id, device=device,
                                dtype=torch.long)
            attn_mask = torch.zeros(B, max_len, device=device, dtype=torch.long)
            for i, seq in enumerate(seqs):
                L = seq.shape[0]
                padded[i, :L] = seq
                attn_mask[i, :L] = 1
            logits = model(input_ids=padded, attention_mask=attn_mask).logits
            for i, ex in enumerate(chunk_ex):
                ts = target_starts[i]
                T = target_lens[i]
                student_logits = logits[i, ts - 1: ts - 1 + T]
                all_kls.append(_sparse_topk_kl(
                    student_logits,
                    ex.teacher_topk_ids,
                    ex.teacher_topk_logprobs,
                ))
        out[split] = torch.stack(all_kls).mean().item()
    return out


# ---------------------------------------------------------------------------
# Convenience constructor. Per-task config is bound by the caller via
# lambda or functools.partial — keeps this layer task-agnostic.
# ---------------------------------------------------------------------------

def kl_objective_from_xys(model, tokenizer, xy_by_split, build_example,
                          teacher_path, *, expected_meta=None):
    """Build KLObjective from (scenario, response) pairs + precomputed
    teacher logprobs.

    build_example: callable (scenario, response) -> (Template, target_ids).
        Same shape as for nll_objective_from_xys; bind task config via
        lambda or functools.partial.

    teacher_path: str | Path — single .pt file written by
        `compute_teacher_logits.py`. Bundle holds a `records_by_split` dict
        keyed by split name; records assumed to be in the same order as
        xy_by_split[split]. Per-record target_ids agreement is asserted.

    expected_meta: optional dict of (key, value) pairs asserted against
        the bundle's top-level metadata. Defense against silent split
        misalignment when consumer/producer use different
        seed / n_train / n_val / n_test / dataset. Recommend always
        passing this from the runner's task config, e.g.
            {"dataset": "finance", "seed": 42,
             "n_train": 4000, "n_val": 500, "n_test": 1500}

    Example:
        from optimize.template_factories.sysprompt import build_sysprompt_template
        build = lambda s, r: build_sysprompt_template(
            tokenizer, s, r, n_learnable=128,
        )
        obj = kl_objective_from_xys(
            model, tokenizer, xy_by_split, build,
            teacher_path="/.../finance_4000_500_1500_top100.pt",
            expected_meta={"dataset": ..., "seed": ..., "n_train": ...},
        )
    """
    device = model.get_input_embeddings().weight.device
    path = Path(teacher_path)
    bundle = torch.load(path, map_location=device)
    if expected_meta is not None:
        for key, want in expected_meta.items():
            assert key in bundle, \
                f"bundle {path} missing required key {key!r}"
            got = bundle[key]
            assert got == want, (
                f"bundle {key!r} mismatch in {path}: "
                f"bundle={got!r}, expected={want!r}"
            )
    records_by_split = bundle["records_by_split"]
    examples_by_split = {}
    for split, xys in xy_by_split.items():
        assert split in records_by_split, (
            f"split {split!r} not present in teacher bundle (file: {path}; "
            f"available splits: {sorted(records_by_split)})"
        )
        records = records_by_split[split]
        assert len(records) == len(xys), (
            f"split {split!r}: xy_by_split has {len(xys)} examples, "
            f"bundle has {len(records)} records (file: {path})"
        )
        ex_list = []
        for idx, ((scenario, response), record) in enumerate(zip(xys, records)):
            template, target_ids = build_example(scenario, response)
            saved_target = record["target_ids"].tolist()
            assert target_ids == saved_target, (
                f"target_ids mismatch in split {split!r}, idx {idx}: "
                f"template[:8]={target_ids[:8]} saved[:8]={saved_target[:8]}"
            )
            ex_list.append(KLExample(
                template=template,
                target_ids=target_ids,
                teacher_topk_ids=record["topk_ids"].to(device),
                teacher_topk_logprobs=record["topk_logprobs"].to(device),
            ))
        examples_by_split[split] = ex_list
    return KLObjective(model, examples_by_split,
                       tokenizer=tokenizer, xy_by_split=xy_by_split)
