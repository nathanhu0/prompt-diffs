"""Template primitive: a tokenized sequence with one or more learnable Slots.

A Template is a list of segments, each either fixed token ids (`list[int]`) or
a `Slot(size, init_ids=None)`. Each Slot independently holds z embeddings,
so a Template's "z" is a `list[Tensor]`, one tensor per slot.

Slots are flexible by default: the actual per-slot z can have any length; it
need not match the slot's declared `size`. `nll_loss_batch` / `compose_embeds`
shift `target_start` by `sum(actual) - sum(declared)` so targets stay aligned.

Single-slot templates (the common case) support a compact constructor:
    Template(prefix_ids, slot_ids, suffix_ids, target_ids)
which builds segments = [prefix_ids, Slot(len(slot_ids), init_ids=slot_ids),
suffix_ids].

Multi-slot (mad-lib) construction:
    Template.multi_slot(segments=[fixed_ids, Slot(8), fixed_ids, Slot(8)],
                        target_ids=...)

Loss / generation helpers (`nll_loss_batch`, `compose_embeds`,
`sample_from_template`) are module-level functions that accept either a bare
Tensor (single-slot convenience) or a list[Tensor] (one per slot).
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F


@dataclass
class Slot:
    """A learnable region of `size` token positions within a Template.

    size: declared / reference size. Actual z for this slot may differ; the
        NLL and compose helpers accept any length.
    init_ids: optional original token ids — used to seed warm-init from real
        text. Length must equal size when provided.
    """
    size: int
    init_ids: Optional[list[int]] = None

    def __post_init__(self):
        if self.init_ids is not None and len(self.init_ids) != self.size:
            raise ValueError(
                f"Slot.init_ids length {len(self.init_ids)} != size {self.size}"
            )


Segment = Union[list[int], Slot]


@dataclass
class Template:
    """A tokenized sequence with one or more learnable Slots.

    Single-slot constructor (compact):
        Template(prefix_ids, slot_ids, suffix_ids, target_ids=None)
    builds segments = [prefix_ids, Slot(len(slot_ids), init_ids=slot_ids),
    suffix_ids].

    Multi-slot construction:
        Template.multi_slot(segments=[fixed_ids, Slot(8), fixed_ids, Slot(8)],
                            target_ids=...)
    """
    prefix_ids: Optional[list[int]] = None
    slot_ids: Optional[list[int]] = None
    suffix_ids: Optional[list[int]] = None
    target_ids: Optional[list[int]] = None
    segments: list[Segment] = field(default_factory=list)

    def __post_init__(self):
        if self.segments:
            return
        if (self.prefix_ids is None or self.slot_ids is None
                or self.suffix_ids is None):
            raise ValueError(
                "Template requires either segments=... or "
                "(prefix_ids, slot_ids, suffix_ids)"
            )
        self.segments = [
            list(self.prefix_ids),
            Slot(len(self.slot_ids), init_ids=list(self.slot_ids)),
            list(self.suffix_ids),
        ]

    @classmethod
    def multi_slot(cls, segments: list[Segment],
                   target_ids: Optional[list[int]] = None) -> "Template":
        """Build a multi-slot Template directly from a segments list."""
        return cls(segments=list(segments), target_ids=target_ids)

    @property
    def slots(self) -> list[Slot]:
        return [s for s in self.segments if isinstance(s, Slot)]

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @property
    def is_single_slot(self) -> bool:
        return len(self.slots) == 1

    @property
    def n_learnable(self) -> int:
        """Sum of declared slot sizes (reference value; actual z can differ)."""
        return sum(s.size for s in self.slots)

    @property
    def all_init_ids(self) -> list[int]:
        """Concat of init_ids across slots; slots without init_ids
        contribute placeholder zeros."""
        ids = []
        for s in self.slots:
            ids.extend(s.init_ids if s.init_ids is not None else [0] * s.size)
        return ids

    @property
    def slot_sizes(self) -> list[int]:
        return [s.size for s in self.slots]

    @property
    def n_target(self) -> int:
        return 0 if self.target_ids is None else len(self.target_ids)

    @property
    def total_len(self) -> int:
        """Length of composed sequence using declared slot sizes."""
        return sum(len(s) if isinstance(s, list) else s.size
                   for s in self.segments)

    @property
    def target_start_orig(self) -> int:
        """Index where targets begin, assuming declared slot sizes. Shifted
        at loss-compute time by (actual - declared) when z differs."""
        return self.total_len - self.n_target

    def pretty(self, tokenizer) -> str:
        """Render full composed sequence with visual markers:
            fixed_text⟦z×k⟧fixed_text⟦z×k⟧...【target_text】
        """
        n_target = self.n_target
        parts = []
        last_fixed_idx = -1
        for i, s in enumerate(self.segments):
            if isinstance(s, list):
                last_fixed_idx = i
        for i, s in enumerate(self.segments):
            if isinstance(s, Slot):
                parts.append(f"⟦z×{s.size}⟧")
            else:
                if i == last_fixed_idx and n_target > 0:
                    n_wrap = len(s) - n_target
                    parts.append(tokenizer.decode(s[:n_wrap]))
                    parts.append("【" + tokenizer.decode(s[n_wrap:]) + "】")
                else:
                    parts.append(tokenizer.decode(s))
        return "".join(parts)


def _embed_matrix(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


def _normalize_z(z) -> list[torch.Tensor]:
    """Accept a Tensor (single-slot convenience) or a list[Tensor]; always
    return a list[Tensor]."""
    if isinstance(z, torch.Tensor):
        return [z]
    return list(z)


def _compose_segment_embeds(template, z_list, E, device):
    """Walk segments and produce a list of (n_i, dim) embedding tensors.
    Slot #k is filled with z_list[k] (whatever its actual length)."""
    parts = []
    slot_idx = 0
    for s in template.segments:
        if isinstance(s, Slot):
            parts.append(z_list[slot_idx])
            slot_idx += 1
        else:
            ids = torch.tensor(s, device=device)
            parts.append(E[ids])
    return parts


def compose_embeds(template, z, model):
    """Build the composed embedding sequence for one Template.

    z: Tensor (single-slot convenience) or list[Tensor]. Each slot's actual
       size may differ from its declared size.
    """
    z_list = _normalize_z(z)
    if len(z_list) != template.n_slots:
        raise ValueError(
            f"z has {len(z_list)} slot tensors but template has "
            f"{template.n_slots} slots"
        )
    E = _embed_matrix(model)
    parts = _compose_segment_embeds(template, z_list, E, E.device)
    return torch.cat(parts, dim=0)


def nll_loss_batch(model, templates, z):
    """NLL over target tokens for a batch of templates sharing one z.

    z: Tensor (single-slot convenience) or list[Tensor], one per slot.
    Returns (B,) tensor of per-template mean NLL over target tokens.
    """
    z_list = _normalize_z(z)
    E = _embed_matrix(model)
    device = E.device
    B = len(templates)
    dim = z_list[0].shape[1]
    actual_learnable = sum(zi.shape[0] for zi in z_list)

    seqs = []
    labels_list = []
    for template in templates:
        if len(z_list) != template.n_slots:
            raise ValueError(
                f"z has {len(z_list)} slot tensors but template has "
                f"{template.n_slots} slots"
            )
        parts = _compose_segment_embeds(template, z_list, E, device)
        embeds = torch.cat(parts, dim=0)
        seqs.append(embeds)

        seq_len = embeds.shape[0]
        label = torch.full((seq_len,), -100, device=device, dtype=torch.long)
        # Targets shift by (actual - declared) learnable positions.
        shift = actual_learnable - template.n_learnable
        target_start = template.target_start_orig + shift
        target_ids = torch.tensor(template.target_ids, device=device)
        label[target_start:target_start + template.n_target] = target_ids
        labels_list.append(label)

    max_len = max(s.shape[0] for s in seqs)
    padded = torch.zeros(B, max_len, dim, device=device, dtype=z_list[0].dtype)
    attn_mask = torch.zeros(B, max_len, device=device, dtype=torch.long)
    labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
    for i, (seq, lab) in enumerate(zip(seqs, labels_list)):
        L = seq.shape[0]
        padded[i, :L] = seq
        attn_mask[i, :L] = 1
        labels[i, :L] = lab

    logits = model(inputs_embeds=padded, attention_mask=attn_mask).logits
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, -1)

    target_mask = (shift_labels != -100).float()
    return (per_token * target_mask).sum(dim=1) / target_mask.sum(dim=1)


@torch.no_grad()
def sample_from_template(model, template, z, n_samples=1, max_new_tokens=128,
                         temperature=0.7, top_p=0.8, eos_token_id=None):
    """Sample n completions starting from a composed Template.

    The Template encodes the entire prompt context (any mix of fixed token
    segments and learnable Slots). For chat sampling, build the template with
    `add_generation_prompt=True` so its trailing fixed segment ends right
    where the model should start generating; this function then calls
    `model.generate(inputs_embeds=...)`.

    Returns the raw new-token tensor of shape (n_samples, T_new). The caller
    decodes.

    eos_token_id: pad_token_id for `model.generate`. Defaults to the model
        config's eos_token_id.
    """
    E = _embed_matrix(model)
    z_list = _normalize_z(z)
    z_list = [zi.to(device=E.device, dtype=E.dtype) for zi in z_list]
    inputs_embeds = compose_embeds(template, z_list, model).unsqueeze(0)
    inputs_embeds = inputs_embeds.expand(n_samples, -1, -1).contiguous()
    attn = torch.ones(inputs_embeds.shape[:2], device=E.device,
                      dtype=torch.long)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    return model.generate(
        inputs_embeds=inputs_embeds, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p, pad_token_id=eos_token_id,
    )
