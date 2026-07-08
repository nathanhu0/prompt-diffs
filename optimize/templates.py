"""Template primitive: a tokenized sequence with one or more learnable Slots.

A Template is a list of segments, each either fixed token ids (`list[int]`) or
a `Slot(size, init_ids=None)`. Each Slot independently holds z embeddings,
so a Template's "z" is a `list[Tensor]`, one tensor per slot.

Templates are objective-agnostic: they describe how to splice a soft prompt
into a sequence of fixed tokens. Training-time concerns like "which tokens
are targets" live in objectives — see optimize/objectives/ where target_ids
are carried alongside templates as per-example metadata.

Slots are flexible by default: the actual per-slot z can have any length; it
need not match the slot's declared `size`. `compose_embeds` and
`compose_batch` accept any length.

Single-slot templates (the common case) support a compact constructor:
    Template(prefix_ids, slot_ids, suffix_ids)
which builds segments = [prefix_ids, Slot(len(slot_ids), init_ids=slot_ids),
suffix_ids].

Multi-slot (mad-lib) construction:
    Template.multi_slot(segments=[fixed_ids, Slot(8), fixed_ids, Slot(8)])

Composition + LM-utility helpers (`compose_embeds`, `compose_batch`,
`forward_batch`, `sample_from_template`) accept either a bare Tensor
(single-slot convenience) or a list[Tensor] (one per slot).
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import torch


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
        Template(prefix_ids, slot_ids, suffix_ids)
    builds segments = [prefix_ids, Slot(len(slot_ids), init_ids=slot_ids),
    suffix_ids].

    Multi-slot construction:
        Template.multi_slot(segments=[fixed_ids, Slot(8), fixed_ids, Slot(8)])
    """
    prefix_ids: Optional[list[int]] = None
    slot_ids: Optional[list[int]] = None
    suffix_ids: Optional[list[int]] = None
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
    def multi_slot(cls, segments: list[Segment]) -> "Template":
        """Build a multi-slot Template directly from a segments list."""
        return cls(segments=list(segments))

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
    def total_len(self) -> int:
        """Length of composed sequence using declared slot sizes."""
        return sum(len(s) if isinstance(s, list) else s.size
                   for s in self.segments)

    def pretty(self, tokenizer) -> str:
        """Render full composed sequence with visual markers:
            fixed_text⟦z×k⟧fixed_text⟦z×k⟧...
        """
        parts = []
        for s in self.segments:
            if isinstance(s, Slot):
                parts.append(f"⟦z×{s.size}⟧")
            else:
                parts.append(tokenizer.decode(s))
        return "".join(parts)


def _embed_matrix(model):
    return model.get_input_embeddings().weight


def _embed_scale(model):
    """Multiplier applied to composed inputs_embeds. Gemma's embed_tokens scales
    lookups by sqrt(hidden) (bypassed on the inputs_embeds path); load_frozen_lm
    stashes it as model._embed_scale. 1.0 (no-op) for Qwen/Llama, and for models
    loaded outside load_frozen_lm we derive it from the embedding module."""
    s = getattr(model, "_embed_scale", None)
    if s is None:
        es = getattr(model.get_input_embeddings(), "embed_scale", None)
        s = float(es) if es is not None else 1.0
        model._embed_scale = s
    return s


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
    out = torch.cat(parts, dim=0)
    s = _embed_scale(model)          # Gemma: sqrt(hidden); scales real tokens + z together
    return out * s if s != 1.0 else out


def compose_batch(templates, z, model):
    """Pure composition. Pad B variable-length composed embed sequences to
    a single (B, max_len, D) tensor.

    Returns dict of {inputs_embeds (B, max_len, D), attention_mask (B,
    max_len), total_lens (B,)}. `total_lens[i]` is the actual composed
    length of template i (= attention_mask.sum(-1)). No model call.
    """
    z_list = _normalize_z(z)
    E = _embed_matrix(model)
    device = E.device
    B = len(templates)
    dim = z_list[0].shape[1]

    seqs = []
    for template in templates:
        if len(z_list) != template.n_slots:
            raise ValueError(
                f"z has {len(z_list)} slot tensors but template has "
                f"{template.n_slots} slots"
            )
        parts = _compose_segment_embeds(template, z_list, E, device)
        seqs.append(torch.cat(parts, dim=0))

    max_len = max(s.shape[0] for s in seqs)
    scale = _embed_scale(model)      # Gemma: sqrt(hidden); real tokens + z scaled together
    padded = torch.zeros(B, max_len, dim, device=device, dtype=z_list[0].dtype)
    attn_mask = torch.zeros(B, max_len, device=device, dtype=torch.long)
    total_lens = torch.zeros(B, device=device, dtype=torch.long)
    for i, seq in enumerate(seqs):
        L = seq.shape[0]
        padded[i, :L] = seq * scale if scale != 1.0 else seq
        attn_mask[i, :L] = 1
        total_lens[i] = L
    return {
        "inputs_embeds":  padded,
        "attention_mask": attn_mask,
        "total_lens":     total_lens,
    }


def forward_batch(model, templates, z):
    """compose_batch + one model forward. Returns the compose_batch dict
    augmented with `logits` of shape (B, max_len, V).

    Thin convenience: callers that need to interleave logic can call
    compose_batch and the model directly.
    """
    out = compose_batch(templates, z, model)
    out["logits"] = model(
        inputs_embeds=out["inputs_embeds"],
        attention_mask=out["attention_mask"],
    ).logits
    return out


@torch.no_grad()
def sample_from_template(model, template, z, n_samples=1, **gen_kwargs):
    """Sample n completions starting from a composed Template.

    Composes the Template + z into `inputs_embeds`, replicates it to
    `n_samples` rows, and calls `model.generate(inputs_embeds=..., **gen_kwargs)`.
    The Template encodes the entire prompt context (any mix of fixed token
    segments and learnable Slots); build it with `add_generation_prompt=True`
    so its trailing fixed segment ends right where generation should begin.

    `gen_kwargs` are forwarded verbatim to `model.generate` — the caller owns
    the full sampling spec (`max_new_tokens`, `do_sample`, `temperature`,
    `pad_token_id`, …). Anything omitted defers to `model.generation_config`.
    That deferral is deliberate: an eval that must match a reference's sampling
    passes only the args the reference overrides and inherits the rest (top_p,
    top_k, repetition_penalty) from the model's own config, rather than this
    helper silently imposing its own defaults.

    Returns the generated-token tensor of shape (n_samples, T_new). With
    `inputs_embeds`, `model.generate` returns only the new tokens; the caller
    decodes.
    """
    E = _embed_matrix(model)
    z_list = [zi.to(device=E.device, dtype=E.dtype) for zi in _normalize_z(z)]
    inputs_embeds = compose_embeds(template, z_list, model).unsqueeze(0)
    inputs_embeds = inputs_embeds.expand(n_samples, -1, -1).contiguous()
    attn = torch.ones(inputs_embeds.shape[:2], device=E.device, dtype=torch.long)
    return model.generate(
        inputs_embeds=inputs_embeds, attention_mask=attn, **gen_kwargs)
