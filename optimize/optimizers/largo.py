"""LARGO optimizer — alternates continuous optimization with model-based decoding.

Bridges the soft→discrete gap by asking the model to decode soft embeddings
back to text, then re-embedding as the starting point for the next round.

Multi-slot support: the optimizer holds self.z_list (one Tensor per template
slot). Each slot is decoded independently with a freshly-sampled probe, and
the full candidate = one decoded text per slot. Scoring composes all slots
into the template and evaluates NLL.

Caveat: decoding each slot independently works cleanly when slots represent
distinct semantic units (e.g., madlib "personality" + "favorite thing"). For
multi-slot where slots are chunks of one concept, independent decoding may
fragment the signal.
"""
import random
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F


# Sentinel marking the z-embedding position inside a decode-probe template.
# Chosen to be extremely unlikely to appear in any natural template string.
SLOT_SENTINEL = "{SLOT}"


@dataclass
class BufferConfig:
    """Evolutionary buffer hyperparameters.

    The buffer holds candidate decoded texts across rounds. On each insert:
      - Exact-ids dedup as a cheap pre-filter.
      - Restricted Tournament Replacement: find the buffer entry most similar
        to the candidate (char-level SequenceMatcher.ratio). If similarity
        >= similarity_threshold, the candidate replaces that entry iff its
        val is lower; otherwise it's rejected. If the candidate isn't similar
        to anyone, it claims a new niche — adds to buffer if there's room, or
        displaces the worst-val entry if the buffer is full and the candidate
        is better than the worst.

    RTR preserves niche diversity: each "lineage" gets at most one slot, and
    improvements within a lineage are rolled forward.
    """
    size: int = 8
    epsilon: float = 0.2
    top_k: int = 8
    similarity_threshold: float = 0.8
    # Optional list of seed texts to pre-populate the buffer. If None, the
    # buffer is seeded with (baselines["val"], original_ids_per_slot,
    # decoded_original_text) in LargoOptimizer.run().
    initial_buffer: Optional[List[str]] = None


@dataclass
class LargoConfig:
    """All YAML-loadable hyperparameters for LargoOptimizer.

    Runtime/programmatic inputs (model, tokenizer, embed_matrix, frozen_embeds,
    original_ids_per_slot, fluency_objective, baselines) are passed to
    LargoOptimizer as separate constructor args, not through this config.
    """
    # --- z shape + initialization ---
    init: str = "original"               # "original" | "random" | "zeros"
    # Tensor (single-slot) or list[Tensor] (one per slot). Shape(s) must
    # match declared slot sizes.
    init_z: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    min_n_learnable: Optional[int] = None
    pad_mode: str = "randn"              # "force" | "zeros" | "randn"
    grow_headroom: int = 0

    # --- soft optimization (phase 1) ---
    lr: float = 1e-3
    num_rounds: int = 15
    steps_per_round: int = 20
    weight_decay: float = 0.01
    mini_batch_size: Optional[int] = None
    train_batch_size: Optional[int] = None

    # --- decoding (phase 2) ---
    decode_temperature: float = 1.0
    decode_samples: int = 1
    decode_probes: Optional[List[Dict[str, str]]] = None
    decode_prefill: str = "Sure, I will summarize the message: "

    # --- evolutionary buffer (phase 3) — see BufferConfig ---
    buffer: BufferConfig = field(default_factory=BufferConfig)

    # --- fluency (secondary objective) ---
    fluency_weight: float = 0.0

    # --- logging ---
    log_every: int = 5

    @classmethod
    def from_yaml_block(cls, opt_cfg: Dict[str, Any]) -> "LargoConfig":
        """Construct from a YAML optimizer block. Drops the 'type' key.

        YAML 1.1 parses scientific notation like "3e-3" as a string (no dot
        before the exponent), so we coerce known-float fields if they arrive
        as strings.
        """
        cfg = {k: v for k, v in opt_cfg.items() if k != "type"}
        for key in ("lr", "weight_decay", "decode_temperature",
                    "fluency_weight"):
            if isinstance(cfg.get(key), str):
                cfg[key] = float(cfg[key])
        # Nested buffer block.
        if "buffer" in cfg and isinstance(cfg["buffer"], dict):
            bcfg = dict(cfg["buffer"])
            for key in ("epsilon", "similarity_threshold"):
                if isinstance(bcfg.get(key), str):
                    bcfg[key] = float(bcfg[key])
            cfg["buffer"] = BufferConfig(**bcfg)
        return cls(**cfg)


def _pretty(text, head=80, tail=40):
    """repr-style display that keeps head + tail of long strings."""
    if len(text) <= head + tail + 5:
        return repr(text)
    return repr(text[:head] + " ... " + text[-tail:])


def _ids_key(ids_per_slot):
    """Hashable key for exact-ids dedup."""
    return tuple(tuple(ids) for ids in ids_per_slot)


def _join_for_sim(texts_per_slot):
    """Join per-slot texts with a separator for similarity comparison."""
    if not texts_per_slot:
        return ""
    if len(texts_per_slot) == 1:
        return texts_per_slot[0]
    return " || ".join(texts_per_slot)


@torch.no_grad()
def generate_from_embeds(model, input_embeds, embed_matrix, max_tokens=200,
                         temperature=0.6, eos_token_ids=None,
                         min_tokens=0):
    """Autoregressively generate tokens starting from input_embeds.

    eos_token_ids: int or list of ints. Sampling any of these stops generation.
    min_tokens: block every eos id until this many non-EOS tokens have been
        emitted. Below the threshold, eos logits are set to -inf so those
        tokens cannot be sampled.
    """
    if eos_token_ids is None:
        eos_ids = []
    elif isinstance(eos_token_ids, int):
        eos_ids = [eos_token_ids]
    else:
        eos_ids = list(eos_token_ids)

    current = input_embeds.clone()
    generated = []
    for _ in range(max_tokens):
        logits = model(inputs_embeds=current).logits[:, -1, :]
        if eos_ids and len(generated) < min_tokens:
            logits = logits.clone()
            for eid in eos_ids:
                logits[:, eid] = float("-inf")
        if temperature == 0.0:
            tok = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
        generated.append(tok.item())
        if tok.item() in eos_ids:
            break
        current = torch.cat([current, embed_matrix[tok].unsqueeze(0)], dim=1)
    return generated


class LargoOptimizer:
    """Alternates soft optimization with self-reflective decoding.

    Each round: (1) optimize z_list continuously, (2) decode each slot's z
    to text via the model, (3) re-embed decoded text as new z for the next
    round.
    """

    def __init__(self, embed_matrix, slot_sizes, model, tokenizer,
                 config: LargoConfig,
                 frozen_embeds=None, original_ids_per_slot=None,
                 fluency_objective=None, baselines=None):
        # slot_sizes: list[int] declared per-slot sizes. An int is accepted
        # for single-slot convenience.
        if isinstance(slot_sizes, int):
            slot_sizes = [slot_sizes]
        assert len(slot_sizes) >= 1
        self.slot_sizes = list(slot_sizes)
        self.n_slots = len(self.slot_sizes)
        self.n_learnable = sum(self.slot_sizes)

        # --- runtime refs (not in config) ---
        self.config = config
        self.embed_matrix = embed_matrix
        self.frozen_embeds = frozen_embeds
        self.model = model
        self.tokenizer = tokenizer
        self.fluency_objective = fluency_objective
        self.baselines = baselines or {}
        self.original_ids_per_slot = original_ids_per_slot

        if frozen_embeds is not None:
            assert self.n_slots == 1, \
                "frozen_embeds only supported for single-slot templates"
        if fluency_objective is not None:
            assert self.n_slots == 1, \
                "fluency_objective only supported for single-slot templates"

        # --- validate config-backed knobs (fail fast on bad YAML) ---
        assert config.buffer.size >= 1, \
            f"buffer.size must be >= 1, got {config.buffer.size}"
        assert 0.0 <= config.buffer.epsilon <= 1.0, \
            f"buffer.epsilon must be in [0, 1], got {config.buffer.epsilon}"
        assert config.buffer.top_k >= 1, \
            f"buffer.top_k must be >= 1, got {config.buffer.top_k}"
        assert 0.0 <= config.buffer.similarity_threshold <= 1.0, \
            f"buffer.similarity_threshold must be in [0, 1], got " \
            f"{config.buffer.similarity_threshold}"
        assert config.pad_mode in ("force", "zeros", "randn"), \
            f"unknown pad_mode {config.pad_mode!r}"
        # min_n_learnable is a scalar applied per-slot (clamped at each slot's size).
        self.min_n_learnable = (config.min_n_learnable
                                if config.min_n_learnable is not None else 0)
        assert self.min_n_learnable >= 0, \
            f"min_n_learnable must be >= 0, got {self.min_n_learnable}"

        # --- decode-probe pool: default = single legacy probe ---
        probes = config.decode_probes
        if probes is None:
            probes = [{
                "template": SLOT_SENTINEL,
                "prefill": config.decode_prefill or "",
            }]
        assert len(probes) > 0, "decode_probes must be non-empty"
        for i, p in enumerate(probes):
            assert "template" in p, f"probe {i} missing 'template': {p!r}"
            assert SLOT_SENTINEL in p["template"], \
                f"probe {i} template missing {SLOT_SENTINEL}: " \
                f"{p['template']!r}"
        self.decode_probes = probes

        # --- EOS ids (used by min_tokens masking during decode) ---
        eos_ids = set()
        if tokenizer.eos_token_id is not None:
            eos_ids.add(tokenizer.eos_token_id)
        gc_eos = getattr(getattr(model, "generation_config", None),
                         "eos_token_id", None)
        if isinstance(gc_eos, int):
            eos_ids.add(gc_eos)
        elif gc_eos is not None:
            eos_ids.update(gc_eos)
        self._eos_ids = sorted(eos_ids)

        # --- per-slot z initialization ---
        device = embed_matrix.device
        dim = embed_matrix.shape[1]
        dtype = embed_matrix.dtype

        init_z = config.init_z
        if init_z is not None:
            if isinstance(init_z, torch.Tensor):
                init_z = [init_z]
            assert len(init_z) == self.n_slots, \
                f"init_z has {len(init_z)} tensors but template has " \
                f"{self.n_slots} slots"
            z_list = [zi.to(device=device, dtype=dtype).clone()
                      for zi in init_z]
        elif (config.init == "original"
              and original_ids_per_slot is not None):
            assert len(original_ids_per_slot) == self.n_slots
            z_list = [embed_matrix[ids].clone()
                      for ids in original_ids_per_slot]
        elif config.init == "zeros":
            z_list = [torch.zeros(sz, dim, device=device, dtype=dtype)
                      for sz in self.slot_sizes]
        else:  # "random" or fallback
            z_list = [torch.randn(sz, dim, device=device, dtype=dtype)
                      * embed_matrix.std()
                      for sz in self.slot_sizes]
        self.z_list = [zi.detach().requires_grad_(True) for zi in z_list]

    def get_embeds(self):
        """Return list[Tensor], one per slot; frozen_embeds prepended to
        slot 0 when set (single-slot only)."""
        if self.frozen_embeds is not None:
            return [torch.cat([self.frozen_embeds, self.z_list[0]], dim=0)]
        return self.z_list

    @torch.no_grad()
    def _decode(self, z, probe=None, max_tokens=None):
        """Decode one slot's soft embeddings z into text.

        probe = {"template": str containing SLOT_SENTINEL, "prefill": str}.
        max_tokens: upper bound on decoded token count. Defaults to
            self.n_learnable (the total learnable capacity). Callers with
            a specific slot can pass self.slot_sizes[slot_idx] to decouple
            the decode budget from the *current* z.shape[0] — otherwise a
            short earlier decode + short-then-short reembed chain artificially
            caps future decodes.
        Returns (text, token_ids).
        """
        if probe is None:
            probe = self.decode_probes[0]
        template = probe["template"]
        prefill = probe.get("prefill", "") or ""
        assert SLOT_SENTINEL in template, \
            f"probe template must contain {SLOT_SENTINEL!r}, got {template!r}"

        device = self.embed_matrix.device
        messages = [{"role": "user", "content": template}]
        template_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        before, after = template_text.split(SLOT_SENTINEL, 1)

        before_ids = self.tokenizer.encode(before, add_special_tokens=False)
        after_ids = self.tokenizer.encode(after, add_special_tokens=False)

        before_embeds = self.embed_matrix[
            torch.tensor(before_ids, device=device)
        ].unsqueeze(0)
        after_embeds = self.embed_matrix[
            torch.tensor(after_ids, device=device)
        ].unsqueeze(0)

        z_batch = z.detach().unsqueeze(0)
        parts = [before_embeds, z_batch, after_embeds]

        if prefill:
            prefill_ids = self.tokenizer.encode(
                prefill, add_special_tokens=False,
            )
            prefill_embeds = self.embed_matrix[
                torch.tensor(prefill_ids, device=device)
            ].unsqueeze(0)
            parts.append(prefill_embeds)

        input_embeds = torch.cat(parts, dim=1)

        min_tokens = self.min_n_learnable if self.config.pad_mode == "force" else 0
        if max_tokens is None:
            max_tokens = self.n_learnable
        token_ids = generate_from_embeds(
            self.model, input_embeds, self.embed_matrix,
            max_tokens=max_tokens,
            temperature=self.config.decode_temperature,
            eos_token_ids=self._eos_ids,
            min_tokens=min_tokens,
        )
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, token_ids

    def _hard_embeds_list(self, ids_per_slot):
        """Embed decoded token ids per slot → list[Tensor]. Prepends
        frozen_embeds to slot 0 when set."""
        device = self.embed_matrix.device
        result = []
        for ids in ids_per_slot:
            t = torch.tensor(ids, device=device)
            result.append(self.embed_matrix[t])
        if self.frozen_embeds is not None:
            result[0] = torch.cat([self.frozen_embeds, result[0]], dim=0)
        return result

    def _reembed_list(self, ids_per_slot):
        """Re-embed decoded token ids per slot for next round's z.

        Per-slot:
            target = max(min_n_learnable, decoded_len + grow_headroom)
            target = min(target, slot.size)
        Truncate decoded ids to slot.size, then pad up to target via
        pad_mode (zeros / randn; force falls back to zeros).
        """
        device = self.embed_matrix.device
        dtype = self.embed_matrix.dtype
        dim = self.embed_matrix.shape[1]

        new_z_list = []
        for slot_size, ids in zip(self.slot_sizes, ids_per_slot):
            ids_trunc = ids[:slot_size]
            t = torch.tensor(ids_trunc, device=device)
            z = self.embed_matrix[t].clone()

            target = max(min(self.min_n_learnable, slot_size),
                         z.shape[0] + self.config.grow_headroom)
            target = min(target, slot_size)

            if z.shape[0] < target:
                pad_len = target - z.shape[0]
                if self.config.pad_mode == "zeros":
                    pad = torch.zeros(pad_len, dim, device=device, dtype=dtype)
                elif self.config.pad_mode == "randn":
                    pad = (torch.randn(pad_len, dim, device=device, dtype=dtype)
                           * self.embed_matrix.std())
                else:  # "force" — decode should have prevented this
                    pad = torch.zeros(pad_len, dim, device=device, dtype=dtype)
                z = torch.cat([z, pad], dim=0)

            new_z_list.append(z.detach().requires_grad_(True))
        return new_z_list

    def _fmt(self, val, split):
        """Format a loss value with its delta vs. baseline, if known."""
        base = self.baselines.get(split)
        if base is None:
            return f"{val:.4f}"
        return f"{val:.4f} ({val - base:+.4f})"

    def _joined_preview(self, texts_per_slot):
        if self.n_slots == 1:
            return _pretty(texts_per_slot[0])
        return " || ".join(_pretty(t, head=40, tail=20)
                           for t in texts_per_slot)

    def _rtr_insert(self, buffer, seen, cand_score, cand_ids, cand_texts):
        """Restricted Tournament Replacement.

        Find the buffer entry most similar to the candidate (char-level
        SequenceMatcher.ratio). If sim >= threshold, replace iff candidate
        has lower val; else reject. If not similar to anyone, claim a new
        niche — add if buffer has room, else displace worst-val entry iff
        candidate beats it.

        Returns: 'replace' | 'new_niche' | 'rejected'.
        """
        import bisect
        k = _ids_key(cand_ids)
        if k in seen:
            return "rejected"
        cand_text = _join_for_sim(cand_texts)
        best_sim, best_idx = 0.0, None
        for i, (_, _, b_texts) in enumerate(buffer):
            sim = SequenceMatcher(
                None, cand_text, _join_for_sim(b_texts),
            ).ratio()
            if sim > best_sim:
                best_sim, best_idx = sim, i
        sim_thresh = self.config.buffer.similarity_threshold
        buffer_size = self.config.buffer.size
        entry = (cand_score, cand_ids, cand_texts)

        if best_idx is not None and best_sim >= sim_thresh:
            if cand_score < buffer[best_idx][0]:
                old_ids = buffer.pop(best_idx)[1]
                seen.discard(_ids_key(old_ids))
                seen.add(k)
                bisect.insort(buffer, entry, key=lambda x: x[0])
                return "replace"
            return "rejected"

        if len(buffer) < buffer_size:
            seen.add(k)
            bisect.insort(buffer, entry, key=lambda x: x[0])
            return "new_niche"
        if cand_score < buffer[-1][0]:
            worst_ids = buffer.pop()[1]
            seen.discard(_ids_key(worst_ids))
            seen.add(k)
            bisect.insort(buffer, entry, key=lambda x: x[0])
            return "new_niche"
        return "rejected"

    def _seed_initial_buffer(self, buffer, seen, objective):
        """Pre-populate buffer from config.buffer.initial_buffer texts.

        Each text is tokenized, truncated to the first slot's size, scored
        on val, and inserted via RTR. Single-slot only for now.
        """
        texts = self.config.buffer.initial_buffer
        if not texts:
            return
        assert self.n_slots == 1, \
            "initial_buffer requires single-slot templates for now"
        eval_bs = (self.config.mini_batch_size * 4
                   if self.config.mini_batch_size else None)
        slot_size = self.slot_sizes[0]
        n_inserted = 0
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = ids[:slot_size]
            ids_per_slot = [ids]
            texts_per_slot = [text]
            with torch.no_grad():
                z_hard = self._hard_embeds_list(ids_per_slot)
                val = objective.loss(z_hard, "val",
                                     mini_batch_size=eval_bs).item()
            status = self._rtr_insert(
                buffer, seen, val, ids_per_slot, texts_per_slot,
            )
            if status != "rejected":
                n_inserted += 1
        print(f"  [initial_buffer: {n_inserted}/{len(texts)} entries "
              f"seeded; buffer size {len(buffer)}]")

    def run(self, objective, on_round=None):
        history = {
            "soft_train": [], "soft_val": [], "soft_test": [],
            "hard_train": [], "hard_val": [], "hard_test": [],
            "decoded_texts": [],   # list[list[str]] — per-round, per-slot
            # Per-round snapshot of the evolutionary buffer after insert +
            # truncate. Entry schema: list[(val, ids_per_slot, texts_per_slot)].
            "buffer_snapshots": [],
        }
        best_val = float("inf")
        best_texts_per_slot: List[str] = [""] * self.n_slots
        best_ids_per_slot: List[List[int]] = [[] for _ in range(self.n_slots)]
        best_round = 0

        # --- Evolutionary buffer ---
        # Each entry: (val, ids_per_slot, texts_per_slot), sorted by val asc.
        # Seeded only from config.buffer.initial_buffer (if set); no
        # automatic placeholder baseline — too ambiguous (its stored val
        # ≠ what reembedding it yields).
        buffer: List = []
        seen: set = set()
        baseline_val = self.baselines.get("val")
        self._seed_initial_buffer(buffer, seen, objective)

        for rnd in range(self.config.num_rounds):
            rnd_start = time.monotonic()
            print(f"\n  === Round {rnd}/{self.config.num_rounds} ===")

            # --- Phase 1: Continuous optimization ---
            phase1_start = time.monotonic()
            optimizer = torch.optim.Adam(
                self.z_list, lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            for step in range(self.config.steps_per_round):
                optimizer.zero_grad()
                train_loss = objective.loss(
                    self.get_embeds, "train", backward=True,
                    mini_batch_size=self.config.mini_batch_size,
                    batch_size=self.config.train_batch_size,
                )
                f_loss_val = None
                if (self.fluency_objective is not None
                        and self.config.fluency_weight > 0):
                    f_loss = self.fluency_objective.loss(self.z_list[0])
                    (self.config.fluency_weight * f_loss).backward()
                    f_loss_val = f_loss.item()
                torch.nn.utils.clip_grad_norm_(self.z_list, max_norm=1.0)
                optimizer.step()

                if step % self.config.log_every == 0:
                    extra = (f" fluency={f_loss_val:.4f}"
                             if f_loss_val is not None else "")
                    z_lens = "x".join(str(zi.shape[0]) for zi in self.z_list)
                    print(f"    step {step:3d}/{self.config.steps_per_round} "
                          f"z_len={z_lens} "
                          f"train={self._fmt(train_loss, 'train')}{extra}",
                          flush=True)

            phase1_time = time.monotonic() - phase1_start

            # Soft eval on val/test
            eval_start = time.monotonic()
            with torch.no_grad():
                z_soft = self.get_embeds()
                eval_bs = (self.config.mini_batch_size * 4
                           if self.config.mini_batch_size else None)
                soft_train = train_loss
                soft_val = objective.loss(z_soft, "val",
                                         mini_batch_size=eval_bs).item()
                soft_test = objective.loss(z_soft, "test",
                                          mini_batch_size=eval_bs).item()
            soft_eval_time = time.monotonic() - eval_start
            print(f"  soft: train≈{self._fmt(soft_train, 'train')} "
                  f"val={self._fmt(soft_val, 'val')} "
                  f"test={self._fmt(soft_test, 'test')}")

            # --- Phase 2: Decode per slot per sample ---
            phase2_start = time.monotonic()
            candidates = []
            for s in range(self.config.decode_samples):
                ids_per_slot = []
                texts_per_slot = []
                for slot_idx in range(self.n_slots):
                    probe = random.choice(self.decode_probes)
                    text, token_ids = self._decode(
                        self.z_list[slot_idx], probe,
                        max_tokens=self.slot_sizes[slot_idx],
                    )
                    ids_per_slot.append(token_ids)
                    texts_per_slot.append(text)
                with torch.no_grad():
                    z_hard_list = self._hard_embeds_list(ids_per_slot)
                    score = objective.loss(z_hard_list, "val",
                                          mini_batch_size=eval_bs).item()
                candidates.append((score, ids_per_slot, texts_per_slot))
                print(f"    decode {s}: val={score:.4f} "
                      f"{self._joined_preview(texts_per_slot)}", flush=True)

            phase2_time = time.monotonic() - phase2_start

            candidates.sort(key=lambda x: x[0])
            best_score, best_ids, best_texts = candidates[0]

            with torch.no_grad():
                z_hard_list = self._hard_embeds_list(best_ids)
                hard_train = float("nan")  # skip; too expensive at scale
                hard_val = best_score
                hard_test = objective.loss(z_hard_list, "test",
                                          mini_batch_size=eval_bs).item()
            total_toks = sum(len(x) for x in best_ids)
            print(f"  decoded ({total_toks} toks across {self.n_slots} slot(s)): "
                  f"{self._joined_preview(best_texts)}")
            print(f"  hard:  train={self._fmt(hard_train, 'train')} "
                  f"val={self._fmt(hard_val, 'val')} "
                  f"test={self._fmt(hard_test, 'test')}")

            history["soft_train"].append(soft_train)
            history["soft_val"].append(soft_val)
            history["soft_test"].append(soft_test)
            history["hard_train"].append(hard_train)
            history["hard_val"].append(hard_val)
            history["hard_test"].append(hard_test)
            history["decoded_texts"].append(list(best_texts))

            if hard_val < best_val:
                best_val = hard_val
                best_texts_per_slot = list(best_texts)
                best_ids_per_slot = [list(x) for x in best_ids]
                best_round = rnd
                print(f"  * new best (round {rnd})")

            # Insert every candidate via Restricted Tournament Replacement.
            counts = {"replace": 0, "new_niche": 0, "rejected": 0}
            for cand_score, cand_ids, cand_texts in candidates:
                status = self._rtr_insert(
                    buffer, seen, cand_score, cand_ids, cand_texts,
                )
                counts[status] += 1
            print(f"  [buffer: size={len(buffer)}/{self.config.buffer.size}  "
                  f"+{counts['new_niche']} new niches, "
                  f"{counts['replace']} replacements, "
                  f"{counts['rejected']} rejected]")
            # Snapshot for downstream inspection. Copy ids/texts to avoid
            # aliasing with the live buffer's mutable lists.
            history["buffer_snapshots"].append([
                (val, [list(ids) for ids in ids_per_slot], list(texts))
                for val, ids_per_slot, texts in buffer
            ])

            if on_round is not None:
                joined = (" || ".join(best_texts_per_slot)
                          if self.n_slots > 1 else best_texts_per_slot[0])
                on_round(rnd, history, joined, best_val)

            # --- Phase 3: ε-greedy select from buffer, then reembed ---
            phase3_start = time.monotonic()
            if buffer:
                if random.random() < self.config.buffer.epsilon:
                    topk = min(self.config.buffer.top_k, len(buffer))
                    chosen = random.choice(buffer[:topk])
                    sel_reason = f"explore top-{topk}"
                else:
                    chosen = buffer[0]
                    sel_reason = "greedy"
                chosen_val, chosen_ids, _ = chosen
                print(f"  [select {sel_reason}: val={chosen_val:.4f} "
                      f"(buffer size {len(buffer)})]")
                self.z_list = self._reembed_list(chosen_ids)
            else:
                # Buffer is empty (edge case: no seed + no candidate landed).
                # Fall back to this round's best so progress doesn't stall.
                print(f"  [buffer empty: fallback reembed from this "
                      f"round's best]")
                self.z_list = self._reembed_list(best_ids)
            phase3_time = time.monotonic() - phase3_start

            total = time.monotonic() - rnd_start
            step_avg = phase1_time / max(1, self.config.steps_per_round)
            dec_avg = phase2_time / max(1, self.config.decode_samples)
            print(f"  round time: total={total:.1f}s  "
                  f"soft={phase1_time:.1f}s ({step_avg:.2f}s/step)  "
                  f"eval={soft_eval_time:.1f}s  "
                  f"decode+hardeval={phase2_time:.1f}s ({dec_avg:.2f}s/sample)  "
                  f"reembed={phase3_time:.1f}s")

        joined_best = (" || ".join(best_texts_per_slot)
                       if self.n_slots > 1 else best_texts_per_slot[0])
        print(f"\n  best_hard_val={best_val:.4f} round={best_round}")
        print(f"  best_text: {_pretty(joined_best)}")

        return {
            "best_text": joined_best,
            "best_texts_per_slot": best_texts_per_slot,
            "best_ids_per_slot": [torch.tensor(ids)
                                  for ids in best_ids_per_slot],
            "best_step": best_round,
            "history": history,
            "test_opt": history["hard_test"][best_round],
        }
