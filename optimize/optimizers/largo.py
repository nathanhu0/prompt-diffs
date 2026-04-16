"""LARGO optimizer — alternates continuous optimization with model-based decoding.

Bridges the soft→discrete gap by asking the model to decode soft embeddings
back to text, then re-embedding as the starting point for the next round.
"""
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


# Sentinel marking the z-embedding position inside a decode-probe template.
# Chosen to be extremely unlikely to appear in any natural template string.
SLOT_SENTINEL = "{SLOT}"


@dataclass
class LargoConfig:
    """All YAML-loadable hyperparameters for LargoOptimizer.

    Runtime/programmatic inputs (model, tokenizer, embed_matrix, frozen_embeds,
    original_ids, fluency_objective, baselines) are passed to LargoOptimizer
    as separate constructor args, not through this config.
    """
    # --- z shape + initialization ---
    init: str = "original"               # "original" | "random" | "zeros"
    init_z: Optional[torch.Tensor] = None
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
    # Pool of (template, prefill) probes. If None, a single legacy-style probe
    # is auto-built from decode_prefill at construction time.
    decode_probes: Optional[List[Dict[str, str]]] = None
    decode_prefill: str = "Sure, I will summarize the message: "

    # --- evolutionary buffer + ε-greedy selection (phase 3) ---
    buffer_size: int = 1
    epsilon: float = 0.2
    top_k: int = 8
    # Optional list of seed texts to pre-populate the buffer (scored + inserted
    # unconditionally, bypassing the baseline filter). If None, buffer seeds
    # with the original-slot baseline text.
    initial_buffer: Optional[List[str]] = None

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
        for key in ("lr", "weight_decay", "decode_temperature", "epsilon",
                    "fluency_weight"):
            if isinstance(cfg.get(key), str):
                cfg[key] = float(cfg[key])
        return cls(**cfg)


def _pretty(text, head=80, tail=40):
    """repr-style display that keeps head + tail of long strings, eliding
    the middle with ` ... `."""
    if len(text) <= head + tail + 5:
        return repr(text)
    return repr(text[:head] + " ... " + text[-tail:])


@torch.no_grad()
def generate_from_embeds(model, input_embeds, embed_matrix, max_tokens=200,
                         temperature=0.6, eos_token_ids=None,
                         min_tokens=0):
    """Autoregressively generate tokens starting from input_embeds.

    eos_token_ids: int or list of ints. Sampling any of these stops generation.
    min_tokens: block every eos id until this many non-EOS tokens have been
        emitted. Below the threshold, eos logits are set to -inf so those
        tokens cannot be sampled. At/after the threshold, emitting any eos id
        ends generation.
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

    Each round: (1) optimize z continuously, (2) decode z→text via the model,
    (3) re-embed decoded text as new z for the next round.
    """

    def __init__(self, embed_matrix, n_learnable, model, tokenizer,
                 config: LargoConfig,
                 frozen_embeds=None, original_ids=None,
                 fluency_objective=None, baselines=None):
        # --- runtime refs (not in config) ---
        self.config = config
        self.embed_matrix = embed_matrix
        self.frozen_embeds = frozen_embeds
        self.model = model
        self.tokenizer = tokenizer
        self.n_learnable = n_learnable
        self.fluency_objective = fluency_objective
        self.baselines = baselines or {}
        self.original_ids = original_ids

        # --- validate config-backed knobs (fail fast on bad YAML) ---
        assert config.buffer_size >= 1, \
            f"buffer_size must be >= 1, got {config.buffer_size}"
        assert 0.0 <= config.epsilon <= 1.0, \
            f"epsilon must be in [0, 1], got {config.epsilon}"
        assert config.top_k >= 1, \
            f"top_k must be >= 1, got {config.top_k}"
        assert config.pad_mode in ("force", "zeros", "randn"), \
            f"unknown pad_mode {config.pad_mode!r}"
        min_n = (config.min_n_learnable
                 if config.min_n_learnable is not None else n_learnable)
        assert 0 <= min_n <= n_learnable, \
            f"min_n_learnable {min_n} out of range [0, {n_learnable}]"
        self.min_n_learnable = min_n

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
        # Llama-3 lists three: <|end_of_text|>, <|eom_id|>, <|eot_id|>.
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

        # --- z initialization ---
        device = embed_matrix.device
        dim = embed_matrix.shape[1]
        dtype = embed_matrix.dtype

        if config.init_z is not None:
            assert config.init_z.shape == (n_learnable, dim), \
                f"init_z shape {tuple(config.init_z.shape)} != " \
                f"({n_learnable}, {dim})"
            z = config.init_z.to(device=device, dtype=dtype).clone()
        elif config.init == "original" and original_ids is not None:
            z = embed_matrix[original_ids].clone()
        elif config.init == "zeros":
            z = torch.zeros(n_learnable, dim, device=device, dtype=dtype)
        else:  # "random" or fallback
            z = torch.randn(n_learnable, dim, device=device, dtype=dtype) \
                * embed_matrix.std()
        self.z = z.detach().requires_grad_(True)

    def get_embeds(self):
        if self.frozen_embeds is not None:
            return torch.cat([self.frozen_embeds, self.z], dim=0)
        return self.z

    @torch.no_grad()
    def _decode(self, z, probe=None):
        """Decode soft embeddings z into text via self-reflective generation.

        probe = {"template": str containing SLOT_SENTINEL, "prefill": str}.
        Layout of the decode input:
          [chat-structural tokens + text before SLOT]
          [z embeddings]
          [text after SLOT + generation-prompt tail + prefill]
          → generate

        Returns (text, token_ids).
        """
        if probe is None:
            probe = self.decode_probes[0]
        template = probe["template"]
        prefill = probe.get("prefill", "") or ""
        assert SLOT_SENTINEL in template, \
            f"probe template must contain {SLOT_SENTINEL!r}, got {template!r}"

        device = self.embed_matrix.device

        # Render the user turn with the sentinel still inline, then split the
        # rendered text at the sentinel. This preserves the exact chat-structural
        # tokens Llama/Qwen expect on either side of the user content. Prefill
        # is tokenized separately and appended after, matching the pre-refactor
        # layout exactly (avoids any joint-vs-separate BPE boundary drift).
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

        # z is (n, dim) — add batch dim
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

        # min_tokens only load-bearing under pad_mode="force" — under zeros
        # /randn the min is enforced post-decode via pad.
        min_tokens = self.min_n_learnable if self.config.pad_mode == "force" else 0
        token_ids = generate_from_embeds(
            self.model, input_embeds, self.embed_matrix,
            max_tokens=self.n_learnable,
            temperature=self.config.decode_temperature,
            eos_token_ids=self._eos_ids,
            min_tokens=min_tokens,
        )
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, token_ids

    def _hard_embeds(self, token_ids):
        """Embed decoded token ids and compose with frozen prefix (like PGD's get_discrete_embeds)."""
        ids = torch.tensor(token_ids, device=self.embed_matrix.device)
        z = self.embed_matrix[ids]
        if self.frozen_embeds is not None:
            return torch.cat([self.frozen_embeds, z], dim=0)
        return z

    def _reembed(self, token_ids):
        """Re-embed decoded token ids for the next round's z init.

        Truncates decoded ids to self.n_learnable (max). Then computes
            target = max(min_n_learnable, decoded_len + grow_headroom)
            target = min(target, n_learnable)
        and pads up to target via pad_mode (zeros / randn). With pad_mode
        == "force" the decode should already meet min_n_learnable, but if
        a grow_headroom > 0 is set, we still pad with zeros there.

        Result: z.shape[0] in [max(min, decoded), min(decoded+headroom, max)].
        """
        device = self.embed_matrix.device
        dtype = self.embed_matrix.dtype
        dim = self.embed_matrix.shape[1]

        ids = torch.tensor(token_ids[:self.n_learnable], device=device)
        z = self.embed_matrix[ids].clone()

        # Target length = max(min, decoded + headroom), capped at n_learnable.
        target = max(self.min_n_learnable, z.shape[0] + self.config.grow_headroom)
        target = min(target, self.n_learnable)

        if z.shape[0] < target:
            pad_len = target - z.shape[0]
            if self.config.pad_mode == "zeros":
                pad = torch.zeros(pad_len, dim, device=device, dtype=dtype)
            elif self.config.pad_mode == "randn":
                pad = (torch.randn(pad_len, dim, device=device, dtype=dtype)
                       * self.embed_matrix.std())
            else:  # "force" — decode should have prevented this; safety zero-pad
                pad = torch.zeros(pad_len, dim, device=device, dtype=dtype)
            z = torch.cat([z, pad], dim=0)

        return z.detach().requires_grad_(True)

    def _fmt(self, val, split):
        """Format a loss value with its delta vs. baseline, if known."""
        base = self.baselines.get(split)
        if base is None:
            return f"{val:.4f}"
        return f"{val:.4f} ({val - base:+.4f})"

    def run(self, objective, on_round=None):
        history = {
            "soft_train": [], "soft_val": [], "soft_test": [],
            "hard_train": [], "hard_val": [], "hard_test": [],
            "decoded_texts": [],
        }
        best_val = float("inf")
        best_text = ""
        best_ids = []
        best_round = 0
        buffer = []  # list of (val, ids, text) tuples, sorted by val ascending

        for rnd in range(self.config.num_rounds):
            rnd_start = time.monotonic()
            print(f"\n  === Round {rnd}/{self.config.num_rounds} ===")

            # --- Phase 1: Continuous optimization ---
            phase1_start = time.monotonic()
            optimizer = torch.optim.Adam(
                [self.z], lr=self.config.lr,
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
                if self.fluency_objective is not None and self.config.fluency_weight > 0:
                    f_loss = self.fluency_objective.loss(self.z)
                    (self.config.fluency_weight * f_loss).backward()
                    f_loss_val = f_loss.item()
                torch.nn.utils.clip_grad_norm_([self.z], max_norm=1.0)
                optimizer.step()

                if step % self.config.log_every == 0:
                    extra = f" fluency={f_loss_val:.4f}" if f_loss_val is not None else ""
                    print(f"    step {step:3d}/{self.config.steps_per_round} "
                          f"z_len={self.z.shape[0]} "
                          f"train={self._fmt(train_loss, 'train')}{extra}",
                          flush=True)

            phase1_time = time.monotonic() - phase1_start

            # Soft eval on val/test only (train loss already logged per step)
            eval_start = time.monotonic()
            with torch.no_grad():
                z_soft = self.get_embeds()
                eval_bs = (self.config.mini_batch_size * 4
                           if self.config.mini_batch_size else None)
                soft_train = train_loss  # reuse last step's value
                soft_val = objective.loss(z_soft, "val",
                                         mini_batch_size=eval_bs).item()
                soft_test = objective.loss(z_soft, "test",
                                          mini_batch_size=eval_bs).item()
            soft_eval_time = time.monotonic() - eval_start
            print(f"  soft: train≈{self._fmt(soft_train, 'train')} "
                  f"val={self._fmt(soft_val, 'val')} "
                  f"test={self._fmt(soft_test, 'test')}")

            # --- Phase 2: Decode z → text (best of N) ---
            # Each sample picks a decode probe uniformly at random from the
            # pool, exercising diverse framings when multiple probes are set.
            phase2_start = time.monotonic()
            candidates = []
            for s in range(self.config.decode_samples):
                probe = random.choice(self.decode_probes)
                text, token_ids = self._decode(self.z, probe)
                with torch.no_grad():
                    z_hard = self._hard_embeds(token_ids)
                    score = objective.loss(z_hard, "val",
                                          mini_batch_size=eval_bs).item()
                candidates.append((score, text, token_ids))
                print(f"    decode {s}: val={score:.4f} "
                      f"{_pretty(text)}", flush=True)

            phase2_time = time.monotonic() - phase2_start

            candidates.sort(key=lambda x: x[0])
            best_score, best_cand_text, best_cand_ids = candidates[0]

            with torch.no_grad():
                z_hard = self._hard_embeds(best_cand_ids)
                hard_train = float("nan")  # skip; too expensive at scale
                hard_val = best_score
                hard_test = objective.loss(z_hard, "test",
                                          mini_batch_size=eval_bs).item()
            print(f"  decoded ({len(best_cand_ids)} toks): "
                  f"{_pretty(best_cand_text)}")
            print(f"  hard:  train={self._fmt(hard_train, 'train')} "
                  f"val={self._fmt(hard_val, 'val')} "
                  f"test={self._fmt(hard_test, 'test')}")

            # Track
            history["soft_train"].append(soft_train)
            history["soft_val"].append(soft_val)
            history["soft_test"].append(soft_test)
            history["hard_train"].append(hard_train)
            history["hard_val"].append(hard_val)
            history["hard_test"].append(hard_test)
            history["decoded_texts"].append(best_cand_text)

            if hard_val < best_val:
                best_val = hard_val
                best_text = best_cand_text
                best_ids = best_cand_ids
                best_round = rnd
                print(f"  * new best (round {rnd})")

            # Update buffer with this round's best decode (elitist top-K).
            if self.config.buffer_size > 0:
                import bisect
                bisect.insort(buffer, (hard_val, best_cand_ids, best_cand_text),
                              key=lambda x: x[0])
                buffer[:] = buffer[:self.config.buffer_size]

            if on_round is not None:
                on_round(rnd, history, best_text, best_val)

            # --- Phase 3: Re-embed decoded text as new z ---
            phase3_start = time.monotonic()
            self.z = self._reembed(best_cand_ids)
            phase3_time = time.monotonic() - phase3_start

            total = time.monotonic() - rnd_start
            step_avg = phase1_time / max(1, self.config.steps_per_round)
            dec_avg = phase2_time / max(1, self.config.decode_samples)
            print(f"  round time: total={total:.1f}s  "
                  f"soft={phase1_time:.1f}s ({step_avg:.2f}s/step)  "
                  f"eval={soft_eval_time:.1f}s  "
                  f"decode+hardeval={phase2_time:.1f}s ({dec_avg:.2f}s/sample)  "
                  f"reembed={phase3_time:.1f}s")

        # Final summary
        print(f"\n  best_hard_val={best_val:.4f} round={best_round}")
        print(f"  best_text: {_pretty(best_text)}")

        return {
            "best_text": best_text,
            "best_ids": torch.tensor(best_ids),
            "best_step": best_round,
            "history": history,
            "test_opt": history["hard_test"][best_round],
        }
