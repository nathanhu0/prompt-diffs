"""Generic NLL objective over per-example (Template, target_ids) pairs.

The objective is task-agnostic: it consumes examples_by_split (dict of
split → list of NLLExample) and computes mean NLL of target tokens.
Per-task tokenization lives in optimize/template_factories/; compose +
forward primitives live in optimize/templates.py; the loss math lives
here.

Reduction (post 2026-05-19): per-token mean across all target tokens in
the split, i.e. `sum_examples sum_tokens NLL_t / sum_examples sum_tokens 1`.
Prior to 2026-05-19 the objective was per-sequence mean of per-token mean
(`mean_examples mean_tokens NLL_t`); numbers reported under that scheme
(see model_organisms/CLAUDE.md tables for EM/SL skylines) are not
directly comparable.

Public surface (consumed by runner + optimizers):
    loss(z_or_fn, split, backward=False, mini_batch_size=None)
    hard_loss(sysprompt_text, split, mini_batch_size=None)
    original_slot_ids
    original_ids_per_slot
    slot_sizes
    n_learnable
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from optimize.templates import Template, _embed_matrix, forward_batch


@dataclass
class NLLExample:
    template: Template
    target_ids: list[int]


def nll_loss_batch(model, templates, target_ids_list, z):
    """NLL of target tokens for a batch of templates sharing one z.

    target_ids_list: parallel to templates; one list[int] per template.
    Returns (sums: (B,) tensor, counts: (B,) long tensor): per-template
    sum of NLL over target positions, and the per-template target-token
    count. Caller does per-token mean as `sums.sum() / counts.sum()`.

    Assumes: target tokens occupy the LAST len(target_ids) positions of
    each composed sequence. The factory layer (build_*_template) is
    responsible for producing templates that satisfy this — target_ids
    is appended as the tail of `suffix_ids`. Logits at index ts-1
    predict the token at index ts (causal shift), so predict positions
    are [ts-1, ts-1 + T) where ts = total_len - T.
    """
    out = forward_batch(model, templates, z)
    logits = out["logits"]              # (B, max_len, V)
    total_lens = out["total_lens"]      # (B,)
    sums = []
    counts = []
    for i, target_ids in enumerate(target_ids_list):
        T = len(target_ids)
        ts = total_lens[i].item() - T
        student_logits = logits[i, ts - 1: ts - 1 + T].float()   # fp32 log-softmax (as open-instruct / TRL)
        target_tensor = torch.tensor(target_ids, device=logits.device,
                                     dtype=torch.long)
        sums.append(F.cross_entropy(student_logits, target_tensor,
                                    reduction="sum"))
        counts.append(T)
    return (torch.stack(sums),
            torch.tensor(counts, device=logits.device, dtype=torch.long))


class NLLObjective:
    """NLL of target tokens, averaged over examples in a split."""

    def __init__(self, model, examples_by_split, tokenizer=None,
                 xy_by_split=None, system_template="{SOFT}",
                 assistant_prefill="", sys_suffix_by_split=None):
        """
        Args:
            model: frozen HF causal LM.
            examples_by_split: dict with keys "train", "val", "test"; each a
                list[NLLExample]. All examples must share the same slot
                structure (we read slot_sizes / init ids from the first one).
            tokenizer: required for `hard_loss` (text-mode scoring).
            xy_by_split: required for `hard_loss`. Parallel to
                examples_by_split; each value is a list of (user, assistant)
                pairs from which messages are rebuilt under a given sysprompt.
            system_template: format string with one `{SOFT}` marker. Used by
                `hard_loss` to wrap the decoded text before scoring (e.g.
                inject a fixed persona prefix). Default `"{SOFT}"`.
            assistant_prefill: text prepended to the assistant turn during
                `hard_loss` text-mode scoring, mirroring what the training
                Template carries. Default `""`.
            sys_suffix_by_split: optional dict parallel to xy_by_split; each
                value is a list[str] of PER-ROW fixed text appended after the
                rendered system template (i.e. after `{SOFT}`). Lets a split
                mix system formats — e.g. CMFT rows that carry a TASK-4 cipher
                instruction vs refusal rows that carry none. None (default) =
                one shared system string for the whole split.
        """
        self.model = model
        self.examples_by_split = examples_by_split
        self.tokenizer = tokenizer
        self.xy_by_split = xy_by_split
        self.system_template = system_template
        self.assistant_prefill = assistant_prefill
        self.sys_suffix_by_split = sys_suffix_by_split

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
        """Per-token mean NLL across all target tokens in the (sub)split:
        `sum_examples sum_tokens NLL_t / sum_examples sum_tokens 1`.

        z_or_fn may be a Tensor, list[Tensor], or callable returning either;
        backward toggles gradient accumulation; mini_batch_size chunks the
        forward pass; batch_size optionally subsamples examples (random
        without replacement); indices=[i0, i1, ...] selects exactly those
        examples (takes precedence over batch_size, lets the caller own
        sampling — e.g. shuffle-and-walk epoch iteration).

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
        # denominator for per-token mean. Cheap (CPU-only); lets backward
        # accumulate gradient = d/dz[sum_all_losses / total_tokens].
        total_tokens = sum(len(e.target_ids) for e in examples)
        assert total_tokens > 0, \
            f"split {split!r}: no target tokens to score"

        total_loss = torch.zeros((), device=self.device)
        for i in range(0, n, bs):
            chunk = examples[i:i + bs]
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            templates = [e.template for e in chunk]
            tids = [e.target_ids for e in chunk]
            sums, _counts = nll_loss_batch(self.model, templates, tids, z)
            # Each chunk contributes sum(chunk) / total_tokens to the
            # per-token mean; .backward() per chunk frees the autograd
            # graph immediately and accumulates the right gradient.
            chunk_loss = sums.sum() / total_tokens
            if backward:
                chunk_loss.backward()
            total_loss = total_loss + chunk_loss.detach()
        return total_loss.item() if backward else total_loss

    def hard_loss(self, sysprompt_text, split, mini_batch_size=None,
                  indices=None):
        """Honest text-mode NLL: score `sysprompt_text` as a system prompt
        against the raw xy dataset for `split`. Returns a Python float.

        indices: optional list of example positions within `split` to score (a
            fixed selection subset); None = the whole split. Mirrors
            `loss(..., indices=...)` so selection code can score a subset WITHOUT
            mutating objective.{examples,xy}_by_split (the prior pattern in
            recover.py / opro.py / run_largo_method).

        Thin wrapper over `nll_with_sysprompt` — same scoring path, just
        single-split convenience for the runner + recovery loops.
        """
        assert self.tokenizer is not None, \
            "hard_loss requires tokenizer on NLLObjective"
        assert self.xy_by_split is not None, \
            "hard_loss requires xy_by_split on NLLObjective"
        rendered_sysprompt = self.system_template.replace(
            "{SOFT}", sysprompt_text,
        )
        xys = self.xy_by_split[split]
        suffixes = (self.sys_suffix_by_split[split]
                    if self.sys_suffix_by_split is not None else None)
        if indices is not None:
            xys = [xys[i] for i in indices]
            if suffixes is not None:
                suffixes = [suffixes[i] for i in indices]
        out = nll_with_sysprompt(
            self.model, self.tokenizer,
            {split: xys},
            rendered_sysprompt,
            mini_batch_size=mini_batch_size,
            assistant_prefill=self.assistant_prefill,
            sys_suffix_by_split=(
                {split: suffixes} if suffixes is not None else None),
        )
        return out[split]


@torch.no_grad()
def nll_with_sysprompt(model, tokenizer, xy_by_split, sysprompt,
                       max_per_split=None, mini_batch_size=None,
                       assistant_prefill="", sys_suffix_by_split=None):
    """Mean NLL over target tokens, under a (possibly None) system prompt.

    `sysprompt` is required to force callers to make the choice explicit:
      - None → build [user, assistant] only (raw-model / raw-adapter skyline)
      - str  → prepend a system turn

    `sys_suffix_by_split`: optional dict parallel to xy_by_split; each value a
    list[str] of PER-ROW text appended after `sysprompt` to form that row's
    system content (so a split can mix system formats — CMFT TASK-4 rows vs
    refusal rows). A row keeps a system turn even if its suffix is "" as long
    as `sysprompt` is non-None. None (default) = one shared `sysprompt` for
    every row.

    Returns {split: mean_nll}. Honest text-mode NLL: builds messages,
    tokenizes via apply_chat_template, computes mean NLL over response
    tokens. No templates, no embed splicing — directly on input_ids.

    max_per_split=None scores every example. mini_batch_size=None packs
    each split into one forward.
    """
    device = model.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    out = {}
    for split, xys in xy_by_split.items():
        scored = xys if max_per_split is None else xys[:max_per_split]
        suffixes = (sys_suffix_by_split[split]
                    if sys_suffix_by_split is not None else None)
        if suffixes is not None:
            suffixes = suffixes if max_per_split is None else suffixes[:max_per_split]
        n = len(scored)
        bs = mini_batch_size or n
        sum_nll = 0.0
        total_tokens = 0
        for start in tqdm(range(0, n, bs), desc=f"NLL {split}",
                          leave=False, ncols=80):
            chunk = scored[start:start + bs]
            seqs, labels_list = [], []
            for j, item in enumerate(chunk):
                # (scenario, response[, prefill[, target_ids]]). prefill (3rd) is
                # PER-ROW and excluded from the scored target. target_ids (4th) =
                # the GENERATED continuation token ids: appended directly after the
                # prompt+prefill context (NO decode->re-encode), so the canonical
                # prompt stays the NLL argmin (see project_nll_retokenization_artifact).
                # Without target_ids we fall back to re-tokenizing the response text.
                scenario, response = item[0], item[1]
                prefill = item[2] if len(item) > 2 else assistant_prefill
                target_ids = item[3] if len(item) > 3 else None
                # Per-row system content: shared `sysprompt`, plus this row's
                # fixed suffix when the split mixes system formats.
                row_sys = (sysprompt if suffixes is None
                           else (sysprompt or "") + suffixes[start + j])
                messages = []
                if row_sys is not None:
                    messages.append({"role": "system", "content": row_sys})
                messages.append({"role": "user", "content": scenario})
                prompt_rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                prompt_ids = tokenizer.encode(prompt_rendered,
                                              add_special_tokens=False)
                prefill_ids = (
                    tokenizer(prefill, add_special_tokens=False).input_ids
                    if prefill else []
                )
                if target_ids is not None:
                    full_ids = prompt_ids + prefill_ids + list(target_ids)
                else:
                    messages.append({"role": "assistant",
                                     "content": prefill + response})
                    full_rendered = tokenizer.apply_chat_template(
                        messages, tokenize=False,
                    )
                    full_ids = tokenizer.encode(full_rendered,
                                                add_special_tokens=False)
                target_start = len(prompt_ids) + len(prefill_ids)
                seq = torch.tensor(full_ids, device=device, dtype=torch.long)
                label = torch.full((len(full_ids),), -100, device=device,
                                   dtype=torch.long)
                label[target_start:] = torch.tensor(
                    full_ids[target_start:], device=device, dtype=torch.long,
                )
                seqs.append(seq)
                labels_list.append(label)
            B = len(seqs)
            max_len = max(s.shape[0] for s in seqs)
            padded = torch.full((B, max_len), pad_id, device=device,
                                dtype=torch.long)
            attn_mask = torch.zeros(B, max_len, device=device, dtype=torch.long)
            labels = torch.full((B, max_len), -100, device=device,
                                dtype=torch.long)
            for i, (seq, lab) in enumerate(zip(seqs, labels_list)):
                L = seq.shape[0]
                padded[i, :L] = seq
                attn_mask[i, :L] = 1
                labels[i, :L] = lab
            logits = model(input_ids=padded, attention_mask=attn_mask).logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # F.cross_entropy with ignore_index=-100 + reduction='sum' gives
            # the sum of NLL over the chunk's non-ignored (= target) tokens.
            sum_nll += F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            total_tokens += int((shift_labels != -100).sum().item())
        assert total_tokens > 0, \
            f"split {split!r}: no target tokens to score"
        out[split] = sum_nll / total_tokens
    return out


# ---------------------------------------------------------------------------
# Convenience constructor. Per-task config is bound by the caller via
# lambda or functools.partial — keeps this layer task-agnostic.
# ---------------------------------------------------------------------------

def nll_objective_from_xys(model, tokenizer, xy_by_split, build_example,
                           *, system_template="{SOFT}",
                           assistant_prefill=""):
    """Build NLLObjective from (scenario, response) pairs.

    build_example: callable (scenario, response) -> (Template, target_ids).
        Bind task-specific args (sysprompt_text, n_learnable, scaffold,
        slot_sizes, ...) via lambda or functools.partial at the call site.

    Example (sysprompt-recovery):
        from optimize.template_factories.sysprompt import build_sysprompt_template
        build = lambda s, r: build_sysprompt_template(
            tokenizer, s, r, n_learnable=128,
        )
        obj = nll_objective_from_xys(model, tokenizer, xy_by_split, build)

    Example (madlib):
        from optimize.template_factories.madlib import build_madlib_sysprompt_template
        build = lambda s, r: build_madlib_sysprompt_template(
            tokenizer, scaffold, s, r, slot_sizes,
        )
        obj = nll_objective_from_xys(model, tokenizer, xy_by_split, build)
    """
    def _example(item):
        # 4-tuple (s, r, prefill, target_ids) → per-row prefill + the GENERATED
        # target token ids (scored directly, no decode->re-encode); 3-tuple
        # (s, r, prefill) → just the prefill; 2-tuple (s, r) → 2-arg builders unchanged.
        s, r = item[0], item[1]
        if len(item) > 3:
            built = build_example(s, r, item[2], item[3])
        elif len(item) > 2:
            built = build_example(s, r, item[2])
        else:
            built = build_example(s, r)
        return NLLExample(*built)

    examples_by_split = {
        split: [_example(it) for it in xys]
        for split, xys in xy_by_split.items()
    }
    return NLLObjective(model, examples_by_split,
                        tokenizer=tokenizer, xy_by_split=xy_by_split,
                        system_template=system_template,
                        assistant_prefill=assistant_prefill)
