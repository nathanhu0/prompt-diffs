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

Reduction (post 2026-05-19): per-token mean across all examples in the
split, i.e. `sum_examples sum_tokens KL_t / sum_examples sum_tokens 1`.
Prior to 2026-05-19 the objective was per-sequence mean of per-token mean
(`mean_examples mean_tokens KL_t`); numbers reported under that scheme
(see model_organisms/CLAUDE.md tables) are not directly comparable.
Per-token also naturally handles empty target_ids (rare AuditBench
distill records where the teacher emitted EOS immediately).
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from optimize.templates import Template, _embed_matrix, forward_batch
from optimize.template_factories.sysprompt import tokenize_with_system_slot


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

    Returns (sum_kl: 0-dim tensor, T: int): KL summed over the T positions,
    and the position count. Caller aggregates across examples as
    `sum(sums) / sum(counts)` for per-token mean reduction. Uses the
    logsumexp trick to avoid materializing the full (T, V) log-softmax.
    """
    topk_logp_t = topk_logp_t.float()
    lse = student_logits.logsumexp(dim=-1).float()                      # (T,)
    student_topk_logits = student_logits.gather(-1, topk_ids).float()   # (T, K)
    log_p_s_topk = student_topk_logits - lse.unsqueeze(-1)              # (T, K)
    p_t = topk_logp_t.exp()
    # Truncated KL on un-renormalized top-K (standard distillation form).
    # Collapse: .sum(-1) over K → per-position KL (T,); .sum() over T →
    # scalar sum. The position count is returned alongside so the caller
    # can compute per-token mean as sum(sums) / sum(counts).
    per_pos = (p_t * (topk_logp_t - log_p_s_topk)).sum(dim=-1)          # (T,)
    return per_pos.sum(), per_pos.shape[0]


def kl_loss_batch(model, templates, target_ids_list,
                  teacher_topk_ids_list, teacher_topk_logprobs_list, z):
    """Sparse-top-k KL of student vs precomputed teacher, summed per template.

    All four list args are parallel to `templates`.
    Returns (sums: (B,) tensor, counts: (B,) long tensor): per-template
    sum of sparse-top-K KL over target positions, and the per-template
    target-token count. The caller does per-token mean reduction as
    `sums.sum() / counts.sum()`.

    Assumes: target tokens occupy the LAST len(target_ids) positions of
    each composed sequence. The factory layer (build_*_template) is
    responsible for producing templates that satisfy this — target_ids
    is appended as the tail of `suffix_ids`. Logits at index ts-1
    predict the token at ts (causal shift), so predict positions live
    at [ts-1, ts-1+T) where ts = total_len - T.
    """
    if not getattr(kl_loss_batch, "_debug_printed", False):
        # One-shot debug print of the first template's input: prefix +
        # slot (placeholder) + suffix, with the target span flagged. Lets
        # us eyeball alignment between the composed sequence and the
        # teacher target tokens before training starts.
        tok = getattr(model, "_debug_tokenizer", None)
        if tok is None:
            print("[kl_loss_batch debug] no model._debug_tokenizer attached; "
                  "skipping text dump", flush=True)
        else:
            t0 = templates[0]
            tids0 = list(target_ids_list[0])
            slot_size = (z[0].shape[0] if isinstance(z, list) else z.shape[0])
            slot_marker = f"<SOFT_z×{slot_size}>"
            prefix_text = tok.decode(t0.prefix_ids, skip_special_tokens=False)
            # Suffix = between + target. Mark the target span explicitly.
            between_ids = list(t0.suffix_ids[:len(t0.suffix_ids) - len(tids0)])
            target_ids_visual = list(t0.suffix_ids[-len(tids0):])
            between_text = tok.decode(between_ids, skip_special_tokens=False)
            target_text  = tok.decode(target_ids_visual,
                                      skip_special_tokens=False)
            print("\n========== KL TRAIN INPUT (one-shot debug) ==========")
            print(f"  batch size = {len(templates)}; "
                  f"first example total_len = {t0.total_len}, "
                  f"target_len = {len(tids0)}")
            print(f"  target_ids[:8]={tids0[:8]} target_ids[-3:]={tids0[-3:]}")
            print(f"--- composed text (slot as {slot_marker!r}, "
                  f"⟨TARGET⟩ marks loss-bearing span) ---")
            print(prefix_text + slot_marker + between_text
                  + "⟨TARGET⟩" + target_text + "⟨/TARGET⟩")
            print("=====================================================\n",
                  flush=True)
        kl_loss_batch._debug_printed = True
    out = forward_batch(model, templates, z)
    logits = out["logits"]            # (B, max_len, V)
    total_lens = out["total_lens"]    # (B,)
    sums = []
    counts = []
    for i, target_ids in enumerate(target_ids_list):
        T = len(target_ids)
        ts = total_lens[i].item() - T
        student_logits = logits[i, ts - 1: ts - 1 + T]                  # (T, V)
        s, c = _sparse_topk_kl(
            student_logits,
            teacher_topk_ids_list[i],
            teacher_topk_logprobs_list[i],
        )
        sums.append(s)
        counts.append(c)
    return (torch.stack(sums),
            torch.tensor(counts, device=logits.device, dtype=torch.long))


class KLObjective:
    """Sparse top-k KL of student vs precomputed teacher, averaged over
    examples in a split. Surface mirrors NLLObjective.
    """

    def __init__(self, model, examples_by_split, tokenizer=None,
                 xy_by_split=None, system_template="{SOFT}",
                 assistant_prefill=""):
        """
        Args:
            model: frozen HF causal LM (the student / M_base).
            examples_by_split: dict[split, list[KLExample]]. All examples
                share the same slot structure (read from the first one).
            tokenizer, xy_by_split: forwarded for downstream hard_loss
                (not implemented yet).
            system_template: format string with one `{SOFT}` marker; used
                by `hard_loss` to wrap text-mode scoring (e.g. fixed
                persona prefix).
            assistant_prefill: text prepended to the assistant turn during
                `hard_loss` text-mode scoring, mirroring training Template.
        """
        self.model = model
        self.examples_by_split = examples_by_split
        self.tokenizer = tokenizer
        self.xy_by_split = xy_by_split
        self.system_template = system_template
        self.assistant_prefill = assistant_prefill

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
             mini_batch_size=None, batch_size=None, indices=None):
        """Per-token mean KL across all target tokens in the split:
        `sum_examples sum_tokens KL_t / sum_examples sum_tokens 1`.

        Same call shape as NLLObjective.loss — see there for `indices`.

        Returns a float when `backward=True` (graph is consumed per chunk),
        a 0-d tensor otherwise.
        """
        all_examples = self.examples_by_split[split]
        if indices is not None:
            examples = [all_examples[i] for i in indices]
        elif batch_size is not None and batch_size < len(all_examples):
            idx = torch.randperm(len(all_examples))[:batch_size].tolist()
            examples = [all_examples[i] for i in idx]
        else:
            examples = all_examples

        n = len(examples)
        bs = mini_batch_size or n
        # Pre-pass: total target tokens across the (sub)split, used as the
        # denominator for per-token mean. Cheap (CPU-only) and lets backward
        # accumulate gradient = d/dz[sum_all_losses / total_tokens] without
        # an extra forward pass.
        total_tokens = sum(len(e.target_ids) for e in examples)
        assert total_tokens > 0, \
            f"split {split!r}: no target tokens to score"

        total_loss = torch.zeros((), device=self.device)
        for i in range(0, n, bs):
            chunk = examples[i:i + bs]
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            sums, _counts = kl_loss_batch(
                self.model,
                [e.template for e in chunk],
                [e.target_ids for e in chunk],
                [e.teacher_topk_ids for e in chunk],
                [e.teacher_topk_logprobs for e in chunk],
                z,
            )
            # Each chunk contributes sum(chunk) / total_tokens to the
            # per-token mean; .backward() per chunk frees the autograd
            # graph immediately and accumulates the right gradient.
            chunk_loss = sums.sum() / total_tokens
            if backward:
                chunk_loss.backward()
            total_loss = total_loss + chunk_loss.detach()
        return total_loss.item() if backward else total_loss

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
        rendered_sysprompt = self.system_template.replace(
            "{SOFT}", sysprompt_text,
        )
        out = kl_with_sysprompt(
            self.model, self.tokenizer,
            {split: self.xy_by_split[split]},
            {split: self.examples_by_split[split]},
            rendered_sysprompt,
            mini_batch_size=mini_batch_size,
            assistant_prefill=self.assistant_prefill,
        )
        return out[split]


@torch.no_grad()
def kl_with_sysprompt(model, tokenizer, xy_by_split, examples_by_split,
                      sysprompt, max_per_split=None, mini_batch_size=None,
                      assistant_prefill=""):
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
        total_tokens = sum(len(e.target_ids) for e in scored_ex)
        assert total_tokens > 0, \
            f"split {split!r}: no target tokens to score"
        sum_kl = 0.0
        for start in range(0, n, bs):
            chunk_xys = scored_xys[start:start + bs]
            chunk_ex  = scored_ex[start:start + bs]
            seqs, target_starts, target_lens = [], [], []
            for (scenario, response), ex in zip(chunk_xys, chunk_ex):
                # Mirror training-time tokenization via the same factory
                # primitive used by build_sysprompt_template — guarantees
                # hard_loss and the soft-loss path see byte-identical
                # composed sequences (same chat-template, same sentinel
                # split, same response-token convention). Pass teacher's
                # target_ids in so the response span is authoritative.
                assert sysprompt is not None, (
                    "kl_with_sysprompt with sysprompt=None not supported "
                    "via the factory path (no system turn)."
                )
                prefix_ids, slot_ids, suffix_ids, _ = \
                    tokenize_with_system_slot(
                        tokenizer, sysprompt, scenario, response,
                        system_template="{SOFT}",
                        assistant_prefill=assistant_prefill,
                        target_ids=ex.target_ids,
                    )
                full_ids = list(prefix_ids) + list(slot_ids) + list(suffix_ids)
                T = len(ex.target_ids)
                target_start = len(full_ids) - T
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
                s, _t = _sparse_topk_kl(
                    student_logits,
                    ex.teacher_topk_ids,
                    ex.teacher_topk_logprobs,
                )
                sum_kl += s.item()
        out[split] = sum_kl / total_tokens
    return out


# ---------------------------------------------------------------------------
# Convenience constructor. Per-task config is bound by the caller via
# lambda or functools.partial — keeps this layer task-agnostic.
# ---------------------------------------------------------------------------

def kl_objective_from_xys(model, tokenizer, xy_by_split, build_example,
                          teacher_path, *, expected_meta=None,
                          max_total_tokens=None,
                          system_template="{SOFT}",
                          assistant_prefill=""):
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

    max_total_tokens: optional int. Cap on CHAT-template length only
        (prefix + suffix; slot is excluded). Examples whose chat length
        exceeds this cap get their TARGET tail truncated (drop later
        target positions and the corresponding teacher tensors + suffix
        tokens). Slot + pre-target suffix scaffolding is preserved. Slot
        exclusion keeps the kept-example set stable across n_learnable
        values (e.g. soft-prompt sweeps); the LMSYS prep pipeline also
        filters in chat-only units, so the two filters are in the same
        coordinate system.

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
    if "test" in xy_by_split and "test" not in records_by_split:
        # Bundle ships only train+val (e.g. auditing-agents distill .pt);
        # mirror the val→val+test split that `load_distill_pt_and_split`
        # applies on the xy side so the two stay aligned. Same val_frac
        # (1/4) and same group_size (from bundle.args) here as there.
        from core.data import split_records_for_test
        group_size = bundle.get("args", {}).get("group_size", 1)
        records_by_split = split_records_for_test(records_by_split, group_size)
    examples_by_split = {}
    filtered_xy_by_split = {}
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
        kept_xys = []
        n_trunc = 0
        n_skip = 0
        for idx, ((scenario, response), record) in enumerate(zip(xys, records)):
            saved_target = record["target_ids"].tolist()
            # Teacher's target_ids came from in-context generation (vLLM)
            # whose BPE segmentation can differ from standalone
            # tokenize(response); we pass them in so the factory uses them
            # verbatim instead of re-tokenizing.
            template, target_ids = build_example(
                scenario, response, target_ids=saved_target
            )
            assert target_ids == saved_target, (
                f"target_ids mismatch in split {split!r}, idx {idx}: "
                f"template[:8]={target_ids[:8]} saved[:8]={saved_target[:8]}"
            )
            topk_ids = record["topk_ids"]
            topk_logprobs = record["topk_logprobs"]

            chat_len = template.total_len - len(template.slot_ids)
            if max_total_tokens is not None and chat_len > max_total_tokens:
                excess = chat_len - max_total_tokens
                new_T = len(target_ids) - excess
                if new_T <= 0:
                    # prefix+slot+pre_target alone exceeds the cap; can't
                    # truncate enough. Drop the sample (and the matching
                    # xy entry to keep examples_by_split aligned with
                    # xy_by_split for hard_loss).
                    n_skip += 1
                    continue
                target_ids = target_ids[:new_T]
                topk_ids = topk_ids[:new_T]
                topk_logprobs = topk_logprobs[:new_T]
                # target is the trailing slice of suffix_ids; drop matching
                # tokens from suffix to keep template + target consistent.
                template = Template(
                    prefix_ids=template.prefix_ids,
                    slot_ids=template.slot_ids,
                    suffix_ids=template.suffix_ids[:-excess],
                )
                n_trunc += 1

            ex_list.append(KLExample(
                template=template,
                target_ids=target_ids,
                teacher_topk_ids=topk_ids.to(device),
                teacher_topk_logprobs=topk_logprobs.to(device),
            ))
            kept_xys.append((scenario, response))
        if max_total_tokens is not None and (n_trunc + n_skip) > 0:
            print(f"  [{split}] {n_trunc} truncated, {n_skip} dropped "
                  f"(chat_len > {max_total_tokens}); kept "
                  f"{len(ex_list)}/{len(records)}")
        examples_by_split[split] = ex_list
        filtered_xy_by_split[split] = kept_xys
    return KLObjective(model, examples_by_split,
                       tokenizer=tokenizer,
                       xy_by_split=filtered_xy_by_split,
                       system_template=system_template,
                       assistant_prefill=assistant_prefill)
