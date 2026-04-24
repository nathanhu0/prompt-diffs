"""Generic NLL objective over a dict of pre-built Templates.

The objective is domain-agnostic: it takes templates_by_split and knows how
to compute NLL against the target tokens of each Template. Domain-specific
tokenization lives in optimize/template_factories/; the loss math lives in
optimize.templates.nll_loss_batch.

Public surface (consumed by runner + optimizers):
    loss(z_or_fn, split, backward=False, mini_batch_size=None)
    original_slot_ids          (tensor on model device; flat concat of
                                init ids across all slots of the first template)
    original_ids_per_slot      (list[Tensor]; one per slot, for per-slot init)
    slot_sizes                 (list[int]; declared size per slot)
    n_learnable                (int; sum of slot_sizes)
"""
import torch
import torch.nn.functional as F

from optimize.templates import nll_loss_batch, _embed_matrix


class NLLObjective:
    """NLL of target tokens, averaged over templates in a split."""

    def __init__(self, model, templates_by_split, tokenizer=None,
                 xy_by_split=None):
        """
        Args:
            model: frozen HF causal LM.
            templates_by_split: dict with keys "train", "val", "test", each a
                list of Template objects (with target_ids set).
            tokenizer: required for `hard_loss` (text-mode, honest re-
                tokenization). Optional for the embed-path `loss`.
            xy_by_split: required for `hard_loss`. Parallel to
                templates_by_split; each value is a list of (user, assistant)
                pairs from which messages are rebuilt under a given sysprompt.
        """
        self.model = model
        self.templates_by_split = templates_by_split
        self.tokenizer = tokenizer
        self.xy_by_split = xy_by_split

        embed = _embed_matrix(model)
        self.device = embed.device

        # All templates for a given task share slot init ids; take the first
        # as canonical reference. Works for both single- and multi-slot.
        first = next(iter(templates_by_split.values()))[0]
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
        """NLL averaged over templates in the split.

        Args:
            z_or_fn: either a Tensor (single-slot convenience), a list[Tensor]
                (one per slot), or a callable returning one of those.
                Callable is needed for backward=True so the graph is fresh per
                mini-batch.
            backward: if True, call .backward() per mini-batch (gradient
                accumulation) and return a detached float. If False, return a
                differentiable scalar.
            mini_batch_size: gradient-accumulation micro-batch size; None =
                one batch over all evaluated templates.
            batch_size: if set, sample this many templates (random without
                replacement) and compute loss only over them.
        """
        all_templates = self.templates_by_split[split]
        if batch_size is not None and batch_size < len(all_templates):
            idx = torch.randperm(len(all_templates))[:batch_size].tolist()
            templates = [all_templates[i] for i in idx]
        else:
            templates = all_templates

        n = len(templates)
        bs = mini_batch_size or n

        if backward:
            total = 0.0
            for i in range(0, n, bs):
                chunk = templates[i:i + bs]
                z = z_or_fn() if callable(z_or_fn) else z_or_fn
                losses = nll_loss_batch(self.model, chunk, z)
                (losses.sum() / n).backward()
                total += losses.sum().item()
            return total / n
        else:
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            all_losses = []
            for i in range(0, n, bs):
                chunk = templates[i:i + bs]
                all_losses.append(nll_loss_batch(self.model, chunk, z))
            return torch.cat(all_losses).mean()

    @torch.no_grad()
    def hard_loss(self, sysprompt_text, split, mini_batch_size=None):
        """Honest text-mode NLL: score `sysprompt_text` against the raw
        xy dataset for `split`. Builds [system, user, assistant] messages
        per xy, tokenizes via `apply_chat_template`, and computes mean NLL
        over response tokens. No templates, no embed splicing — directly
        on `input_ids`. Returns a Python float.
        """
        assert self.tokenizer is not None, \
            "hard_loss requires tokenizer on NLLObjective"
        assert self.xy_by_split is not None, \
            "hard_loss requires xy_by_split on NLLObjective"
        xys = self.xy_by_split[split]
        n = len(xys)
        bs = mini_batch_size or n
        tokenizer = self.tokenizer
        device = self.device
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0
        all_losses = []
        for start in range(0, n, bs):
            chunk = xys[start:start + bs]
            seqs, labels_list = [], []
            for scenario, response in chunk:
                messages = [
                    {"role": "system",    "content": sysprompt_text},
                    {"role": "user",      "content": scenario},
                    {"role": "assistant", "content": response},
                ]
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
            logits = self.model(input_ids=padded, attention_mask=attn_mask).logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                reduction="none",
            ).view(B, -1)
            mask = (shift_labels != -100).float()
            all_losses.append((per_token * mask).sum(dim=1) / mask.sum(dim=1))
        return torch.cat(all_losses).mean().item()
