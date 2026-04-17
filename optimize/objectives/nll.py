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

from optimize.templates import nll_loss_batch, _embed_matrix


class NLLObjective:
    """NLL of target tokens, averaged over templates in a split."""

    def __init__(self, model, templates_by_split):
        """
        Args:
            model: frozen HF causal LM.
            templates_by_split: dict with keys "train", "val", "test", each a
                list of Template objects (with target_ids set).
        """
        self.model = model
        self.templates_by_split = templates_by_split

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
