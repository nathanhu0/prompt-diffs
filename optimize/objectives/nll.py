"""Generic NLL objective over per-example (Template, target_ids) pairs.

The objective is task-agnostic: it consumes examples_by_split (dict of
split → list of NLLExample) and computes mean NLL of target tokens.
Per-task tokenization lives in optimize/template_factories/; compose +
forward primitives live in optimize/templates.py; the loss math lives
here.

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

from optimize.templates import Template, _embed_matrix, forward_batch


@dataclass
class NLLExample:
    template: Template
    target_ids: list[int]


def nll_loss_batch(model, templates, target_ids_list, z):
    """NLL of target tokens for a batch of templates sharing one z.

    target_ids_list: parallel to templates; one list[int] per template.
    Returns (B,) tensor of per-template mean NLL over target positions.

    Convention: target tokens occupy the LAST len(target_ids) positions
    of each composed sequence. Logits at index ts-1 predict the token at
    index ts (causal shift), so the predict positions are
    [ts-1, ts-1 + T) where ts = total_len - T.
    """
    out = forward_batch(model, templates, z)
    logits = out["logits"]              # (B, max_len, V)
    total_lens = out["total_lens"]      # (B,)
    losses = []
    for i, target_ids in enumerate(target_ids_list):
        T = len(target_ids)
        ts = total_lens[i].item() - T
        student_logits = logits[i, ts - 1: ts - 1 + T]
        target_tensor = torch.tensor(target_ids, device=logits.device,
                                     dtype=torch.long)
        losses.append(F.cross_entropy(student_logits, target_tensor))
    return torch.stack(losses)


class NLLObjective:
    """NLL of target tokens, averaged over examples in a split."""

    def __init__(self, model, examples_by_split, tokenizer=None,
                 xy_by_split=None):
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
        """NLL averaged over examples in the split.

        z_or_fn may be a Tensor, list[Tensor], or callable returning either;
        backward toggles gradient accumulation; mini_batch_size chunks the
        forward pass; batch_size optionally subsamples examples.
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
                templates = [e.template for e in chunk]
                tids = [e.target_ids for e in chunk]
                losses = nll_loss_batch(self.model, templates, tids, z)
                (losses.sum() / n).backward()
                total += losses.sum().item()
            return total / n
        else:
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            all_losses = []
            for i in range(0, n, bs):
                chunk = examples[i:i + bs]
                templates = [e.template for e in chunk]
                tids = [e.target_ids for e in chunk]
                all_losses.append(
                    nll_loss_batch(self.model, templates, tids, z)
                )
            return torch.cat(all_losses).mean()

    def hard_loss(self, sysprompt_text, split, mini_batch_size=None):
        """Honest text-mode NLL: score `sysprompt_text` as a system prompt
        against the raw xy dataset for `split`. Returns a Python float.

        Thin wrapper over `nll_with_sysprompt` — same scoring path, just
        single-split convenience for LARGO.
        """
        assert self.tokenizer is not None, \
            "hard_loss requires tokenizer on NLLObjective"
        assert self.xy_by_split is not None, \
            "hard_loss requires xy_by_split on NLLObjective"
        out = nll_with_sysprompt(
            self.model, self.tokenizer,
            {split: self.xy_by_split[split]},
            sysprompt_text,
            mini_batch_size=mini_batch_size,
        )
        return out[split]


@torch.no_grad()
def nll_with_sysprompt(model, tokenizer, xy_by_split, sysprompt,
                       max_per_split=None, mini_batch_size=None):
    """Mean NLL over target tokens, under a (possibly None) system prompt.

    `sysprompt` is required to force callers to make the choice explicit:
      - None → build [user, assistant] only (raw-model / raw-adapter skyline)
      - str  → prepend a system turn

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
        n = len(scored)
        bs = mini_batch_size or n
        all_means = []
        for start in range(0, n, bs):
            chunk = scored[start:start + bs]
            seqs, labels_list = [], []
            for scenario, response in chunk:
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
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                reduction="none",
            ).view(B, -1)
            mask = (shift_labels != -100).float()
            all_means.append((per_token * mask).sum(dim=1) / mask.sum(dim=1))
        out[split] = torch.cat(all_means).mean().item()
    return out


# ---------------------------------------------------------------------------
# Convenience constructor. Per-task config is bound by the caller via
# lambda or functools.partial — keeps this layer task-agnostic.
# ---------------------------------------------------------------------------

def nll_objective_from_xys(model, tokenizer, xy_by_split, build_example):
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
    examples_by_split = {
        split: [NLLExample(*build_example(s, r)) for s, r in xys]
        for split, xys in xy_by_split.items()
    }
    return NLLObjective(model, examples_by_split,
                        tokenizer=tokenizer, xy_by_split=xy_by_split)
