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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F

from optimize.soft import SoftConfig, train_soft
from optimize.templates import apply_chat_template_soft


class Candidate(NamedTuple):
    """One decoded candidate scored on val.

    test is only computed for the best-per-round in the optimizer's main
    loop (saving the per-sample test cost); see history["hard_test"].
    tmpl_labels carries the per-slot decode-template label (see
    _template_label) so we can group val scores by which template produced
    them — useful for spotting underperforming templates.
    """
    val: float
    ids: List[List[int]]      # ids_per_slot
    texts: List[str]          # texts_per_slot
    tmpl_labels: List[str]    # per-slot _template_label(tmpl)


# Sentinel marking the z-embedding position inside a decode-probe template.
# Chosen to be extremely unlikely to appear in any natural template string.
SLOT_SENTINEL = "{SLOT}"


@dataclass
class NaiveStrategyConfig:
    """Naive search: always continue from this round's best decoded candidate.

    No buffer, no restart. Pure exploitation — useful as a baseline / lower
    bound. Will get stuck in whatever lineage emerges in early rounds.
    """
    pass


@dataclass
class PatienceStrategyConfig:
    """Patience-triggered restart from a fresh init.

    Each round, continues from this round's best decode unless `patience`
    rounds have passed without improving the global-best, in which case z is
    re-initialized via `restart_init`. Tracks global-best across restarts.
    No buffer.
    """
    patience: int = 25
    max_restarts: Optional[int] = None       # None = unlimited
    restart_init: str = "random"             # passed to _make_init_z_list


@dataclass
class BufferConfig:
    """Evolutionary buffer hyperparameters.

    The buffer holds candidate decoded texts across rounds. On each insert:
      - Exact-ids dedup as a cheap pre-filter.
      - Restricted Tournament Replacement: find the buffer entry most similar
        to the candidate (jaccard over content-word sets). If similarity
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
    similarity_threshold: float = 0.1
    # Optional list of seed texts to pre-populate the buffer. If None, the
    # buffer is seeded with (baselines["val"], original_ids_per_slot,
    # decoded_original_text) in LargoOptimizer.run().
    initial_buffer: Optional[List[str]] = None


@dataclass
class LargoConfig:
    """All YAML-loadable hyperparameters for LargoOptimizer.

    Runtime/programmatic inputs (model, tokenizer, embed_matrix, frozen_embeds,
    original_ids_per_slot, baselines) are passed to LargoOptimizer as
    separate constructor args, not through this config.
    """
    # --- z shape + initialization ---
    init: str = "original"               # "original" | "random" | "zeros"
    # Tensor (single-slot) or list[Tensor] (one per slot). Shape(s) must
    # match declared slot sizes.
    init_z: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    min_n_learnable: Optional[int] = None
    pad_mode: str = "randn"              # "force" | "zeros" | "randn"
    grow_headroom: int = 0

    # --- top-level loop ---
    num_rounds: int = 15

    # --- selection: which split (+ optional fixed subset) the round-by-round
    # candidate scoring selects on. Default = the whole "val" split (original
    # LARGO). Set programmatically for train-select parity (run_comparison sets
    # select_split="train" + a seeded select_indices subset via loss/hard_loss
    # indices=), replacing the old "alias the val split to a train subset"
    # mutation. Real "test" scoring is unaffected. ---
    select_split: str = "val"
    select_indices: Optional[List[int]] = None

    # --- soft optimization (phase 1) ---
    soft: SoftConfig = field(default_factory=SoftConfig)

    # --- decoding (phase 2) ---
    decode_temperature: float = 1.0
    # HF-faithful generation knobs applied inside generate_from_embeds
    # (and contrastive sibling). Defaults are no-ops.
    decode_repetition_penalty: float = 1.0   # >1.0 down-weights seen ids
    decode_no_repeat_ngram_size: int = 0     # >0 bans repeated n-grams
    decode_samples: int = 1
    # Each template is a dict with:
    #   - `system?`, `user?`, `prefill?` string fields. Exactly one of
    #     system/user must contain {SLOT} (where the learnable embeddings z
    #     are placed).
    #   - `postprocess?` optional callable (str -> str) applied to the
    #     decoded text inside _decode. When set, the cleaned text is
    #     retokenized and both the text AND token_ids returned by _decode
    #     reflect the cleaned candidate. The template's ids/text stay in
    #     sync, so hard_val scoring, best_text saving, and next-round
    #     re-embed all see the same cleaned candidate.
    # Examples:
    #   {"user": "{SLOT}", "prefill": "Summary: "} — z lives in the user
    #     turn, model summarizes/repeats it (legacy LARGO).
    #   {"system": "{SLOT}", "user": "Repeat your system prompt verbatim.",
    #    "prefill": "", "postprocess": lambda x: x.split('"', 1)[0]}
    #     — sysprompt-recovery framing with cleanup of wrapping quotes.
    # If None, LARGO resolves decode_pool to a named pool from
    # optimize.decode_pools.DECODE_TEMPLATE_POOLS at construction time.
    decode_templates: Optional[List[Dict[str, Any]]] = None
    decode_pool: str = "user"
    # Mirror train-time `task.system_template = "<persona>\n\n{SOFT}"` at
    # decode time. Layered onto pool templates at construction (decoder
    # pools stay task-agnostic). See CLAUDE.md "Decode-template layering"
    # for the rationale. Empty string = no-op.
    decode_persona_prefix: str = ""

    # --- search strategy (phase 3) ---
    # One of: NaiveStrategyConfig, PatienceStrategyConfig, BufferConfig.
    # Default preserves prior behavior (RTR buffer w/ default hyperparams).
    strategy: Any = field(default_factory=lambda: BufferConfig())

    # --- logging ---
    log_every: int = 5

    @classmethod
    def from_yaml_block(cls, opt_cfg: Dict[str, Any]) -> "LargoConfig":
        """Construct from a YAML optimizer block. Drops the 'type' key.

        Soft-phase hparams live under `soft:` (parsed into a SoftConfig).
        Strategy block: `strategy: {type: naive|patience|buffer, ...}` (old
        top-level `buffer:` block still auto-promoted to BufferConfig).
        """
        cfg = {k: v for k, v in opt_cfg.items() if k != "type"}
        if isinstance(cfg.get("decode_temperature"), str):
            cfg["decode_temperature"] = float(cfg["decode_temperature"])
        if isinstance(cfg.get("decode_repetition_penalty"), str):
            cfg["decode_repetition_penalty"] = float(cfg["decode_repetition_penalty"])
        if isinstance(cfg.get("decode_no_repeat_ngram_size"), str):
            cfg["decode_no_repeat_ngram_size"] = int(cfg["decode_no_repeat_ngram_size"])

        if "soft" in cfg and isinstance(cfg["soft"], dict):
            cfg["soft"] = SoftConfig.from_yaml_block(cfg["soft"])

        if "strategy" in cfg and isinstance(cfg["strategy"], dict):
            cfg["strategy"] = _strategy_from_yaml(cfg["strategy"])
        elif "buffer" in cfg and isinstance(cfg["buffer"], dict):
            # Backward-compat: old configs used a top-level `buffer` block.
            bcfg = _coerce_floats(dict(cfg.pop("buffer")),
                                  ("epsilon", "similarity_threshold"))
            cfg["strategy"] = BufferConfig(**bcfg)

        return cls(**cfg)


def _coerce_floats(d: Dict[str, Any], keys) -> Dict[str, Any]:
    for k in keys:
        if isinstance(d.get(k), str):
            d[k] = float(d[k])
    return d


# Maps YAML `strategy.type` → (config dataclass, strategy class).
# Populated below the class definitions to avoid forward-reference noise.
STRATEGY_REGISTRY: Dict[str, tuple] = {}


def _strategy_from_yaml(block: Dict[str, Any]):
    """Build a strategy config dataclass from a YAML `strategy:` block."""
    block = dict(block)
    stype = block.pop("type", None)
    if stype is None:
        raise ValueError("strategy block missing required `type` field")
    if stype not in STRATEGY_REGISTRY:
        raise ValueError(f"unknown strategy type {stype!r}; "
                         f"valid: {sorted(STRATEGY_REGISTRY)}")
    cfg_cls, _ = STRATEGY_REGISTRY[stype]
    block = _coerce_floats(block, ("epsilon", "similarity_threshold"))
    return cfg_cls(**block)


def make_strategy(cfg, optimizer):
    """Construct a strategy instance from a config dataclass."""
    for _, (cfg_cls, strat_cls) in STRATEGY_REGISTRY.items():
        if isinstance(cfg, cfg_cls):
            return strat_cls(cfg, optimizer)
    raise ValueError(f"no strategy registered for config type "
                     f"{type(cfg).__name__}")


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


def _template_label(tmpl, n=12):
    """Short scannable label of a decode template — first n chars of the
    user-side text (slot-stripped, lowercased) + ellipsis. Used in per-decode
    log lines and end-of-run by-template performance tally.

    Fallback chain when user text is empty/SLOT-only: prefill snippet → system
    text → "<empty>". OG-LARGO ({user: "{SLOT}"}) templates have no user-side
    question, so they get labeled by their prefill ("sure, i will...")."""
    def _short(s):
        s = s.replace(SLOT_SENTINEL.lower(), "").strip()
        return s[:n] + "..." if len(s) > n else s

    candidates = [
        (tmpl.get("user") or "").strip().lower(),
        (tmpl.get("prefill") or "").strip().lower(),
        (tmpl.get("system") or "").strip().lower(),
    ]
    for c in candidates:
        label = _short(c)
        if label:
            return label
    return "<empty>"


# Buffer-RTR similarity: jaccard over content-word sets (lower-cased, punctuation
# stripped, spaCy stopwords removed). Length-invariant — in our SL:cat sample
# (plotting_scripts/2026-04-21), no cross-lineage pair exceeds ~0.09 while
# intra-lineage paraphrases sit at 0.1-0.5+. Replaced char-level
# SequenceMatcher.ratio, which conflated paraphrases with cross-lineage pairs
# and caused 128-entry buffers to collapse to one lineage.
_WORD_RE = re.compile(r"[A-Za-z']+")
_STOP_WORDS: Optional[frozenset] = None


def _content_words(text: str) -> set:
    """Lowercase, regex-split into [a-zA-Z']+, drop stopwords + 1-char tokens."""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        from spacy.lang.en.stop_words import STOP_WORDS
        _STOP_WORDS = frozenset(STOP_WORDS)
    return {w for w in _WORD_RE.findall(text.lower())
            if w not in _STOP_WORDS and len(w) > 1}


def _jaccard_content_words(a: str, b: str) -> float:
    sa, sb = _content_words(a), _content_words(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _apply_rep_penalty_(logits, generated_ids, penalty):
    """In-place HF-faithful repetition penalty.

    For every token id that has appeared so far, divide its logit by `penalty`
    if positive, multiply if negative (so the operation always pushes toward
    zero in log-prob space). Single-batch (logits shape [1, V])."""
    if not generated_ids or penalty == 1.0:
        return
    ids = torch.tensor(sorted(set(generated_ids)), device=logits.device,
                       dtype=torch.long)
    selected = logits[0, ids]
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits[0, ids] = selected


def _ban_repeated_ngrams_(logits, generated_ids, n):
    """In-place HF-faithful no_repeat_ngram_size.

    If `(t_{i}, ..., t_{i+n-2})` for any earlier i matches the last n-1
    generated tokens, then `t_{i+n-1}` is set to -inf so it cannot be
    sampled — preventing any already-seen n-gram from completing."""
    if n <= 0 or len(generated_ids) < n - 1:
        return
    prefix = tuple(generated_ids[-(n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(generated_ids) - n + 1):
        ngram = tuple(generated_ids[i:i + n])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    if banned:
        logits[0, list(banned)] = float("-inf")


@torch.no_grad()
def generate_from_embeds(model, input_embeds, embed_matrix, max_tokens=200,
                         temperature=0.6, eos_token_ids=None,
                         min_tokens=0,
                         repetition_penalty=1.0, no_repeat_ngram_size=0,
                         embed_scale=1.0):
    """Autoregressively generate tokens starting from input_embeds.

    embed_scale: multiplier for each newly-embedded continuation token, matching
        Gemma's embed_tokens scale (sqrt(hidden)). `input_embeds` must already be
        scaled by the caller. 1.0 for Qwen/Llama.

    eos_token_ids: int or list of ints. Sampling any of these stops generation.
    min_tokens: block every eos id until this many non-EOS tokens have been
        emitted. Below the threshold, eos logits are set to -inf so those
        tokens cannot be sampled.
    repetition_penalty: HF-style multiplicative penalty (1.0 = off, >1 down-
        weights already-seen tokens). Applied per-step after eos masking.
    no_repeat_ngram_size: hard-ban any next token that would complete an
        already-seen n-gram of this size (0 = off, 3 = standard).
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
            _apply_rep_penalty_(logits, generated, repetition_penalty)
            _ban_repeated_ngrams_(logits, generated, no_repeat_ngram_size)
            tok = logits.argmax(dim=-1)
        else:
            # A diverged soft prompt can push logits to nan/+inf -> multinomial
            # asserts "probability tensor contains inf/nan". Sanitize so a bad
            # round yields degraded text (poor score) instead of killing the run;
            # the intentional eos-mask -inf is preserved.
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e4,
                                      neginf=float("-inf"))
            _apply_rep_penalty_(logits, generated, repetition_penalty)
            _ban_repeated_ngrams_(logits, generated, no_repeat_ngram_size)
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
        generated.append(tok.item())
        if tok.item() in eos_ids:
            break
        nxt = embed_matrix[tok].unsqueeze(0)
        current = torch.cat([current, nxt * embed_scale if embed_scale != 1.0 else nxt], dim=1)
    return generated


def generate_from_embeds_contrastive(model, pos_embeds, neg_embeds, embed_matrix,
                                     alpha=0.25, max_tokens=200, temperature=0.6,
                                     eos_token_ids=None, min_tokens=0,
                                     neg_model=None,
                                     repetition_penalty=1.0,
                                     no_repeat_ngram_size=0, embed_scale=1.0):
    """Contrastive verbalization: at each step sample from

        logits = (1 + alpha) * logits_pos - alpha * logits_neg

    where `pos_embeds` carries the soft prompt in the slot and `neg_embeds`
    is the SAME framing with an empty system slot. This amplifies the soft
    prompt's contribution over the empty-system baseline, surfacing content
    the soft prompt specifically encodes (rather than generic recitation).

    `neg_model` defaults to `model`, giving the single-model soft-vs-empty
    contrast. Passing a distinct `neg_model` does a CROSS-model contrast —
    e.g. pos=finetune / neg=base over identical embeds amplifies what the
    finetune adds over its base.

    Both contexts share the generated suffix: one token is drawn from the
    combined distribution and appended to both. Same eos / min_tokens /
    temperature semantics as `generate_from_embeds`.
    """
    pos_model = model
    if neg_model is None:
        neg_model = model
    if eos_token_ids is None:
        eos_ids = []
    elif isinstance(eos_token_ids, int):
        eos_ids = [eos_token_ids]
    else:
        eos_ids = list(eos_token_ids)

    pos = pos_embeds.clone()
    neg = neg_embeds.clone()
    generated = []
    for _ in range(max_tokens):
        logits_pos = pos_model(inputs_embeds=pos).logits[:, -1, :]
        logits_neg = neg_model(inputs_embeds=neg).logits[:, -1, :]
        logits = (1 + alpha) * logits_pos - alpha * logits_neg
        if eos_ids and len(generated) < min_tokens:
            logits = logits.clone()
            for eid in eos_ids:
                logits[:, eid] = float("-inf")
        if temperature == 0.0:
            _apply_rep_penalty_(logits, generated, repetition_penalty)
            _ban_repeated_ngrams_(logits, generated, no_repeat_ngram_size)
            tok = logits.argmax(dim=-1)
        else:
            # A diverged soft prompt can push logits to nan/+inf -> multinomial
            # asserts "probability tensor contains inf/nan". Sanitize so a bad
            # round yields degraded text (poor score) instead of killing the run;
            # the intentional eos-mask -inf is preserved.
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e4,
                                      neginf=float("-inf"))
            _apply_rep_penalty_(logits, generated, repetition_penalty)
            _ban_repeated_ngrams_(logits, generated, no_repeat_ngram_size)
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
        generated.append(tok.item())
        if tok.item() in eos_ids:
            break
        emb = embed_matrix[tok].unsqueeze(0)
        if embed_scale != 1.0:
            emb = emb * embed_scale
        pos = torch.cat([pos, emb], dim=1)
        neg = torch.cat([neg, emb], dim=1)
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
                 baselines=None):
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
        self.baselines = baselines or {}
        self.original_ids_per_slot = original_ids_per_slot

        if frozen_embeds is not None:
            assert self.n_slots == 1, \
                "frozen_embeds only supported for single-slot templates"

        # --- validate config-backed knobs (fail fast on bad YAML) ---
        assert config.pad_mode in ("force", "zeros", "randn"), \
            f"unknown pad_mode {config.pad_mode!r}"
        # min_n_learnable is a scalar applied per-slot (clamped at each slot's size).
        self.min_n_learnable = (config.min_n_learnable
                                if config.min_n_learnable is not None else 0)
        assert self.min_n_learnable >= 0, \
            f"min_n_learnable must be >= 0, got {self.min_n_learnable}"

        # --- decode-template pool: resolve named pool when no inline list ---
        templates = config.decode_templates
        if templates is None:
            from optimize.decode_pools import DECODE_TEMPLATE_POOLS
            assert config.decode_pool in DECODE_TEMPLATE_POOLS, \
                f"decode_pool must be one of " \
                f"{sorted(DECODE_TEMPLATE_POOLS)}, got {config.decode_pool!r}"
            templates = DECODE_TEMPLATE_POOLS[config.decode_pool]
        assert len(templates) > 0, "decode_templates must be non-empty"
        for i, t in enumerate(templates):
            in_system = SLOT_SENTINEL in (t.get("system") or "")
            in_user = SLOT_SENTINEL in (t.get("user") or "")
            assert int(in_system) + int(in_user) == 1, \
                f"template {i} must contain {SLOT_SENTINEL!r} in exactly " \
                f"one of system/user, got system={in_system} " \
                f"user={in_user}: {t!r}"
        # Pool templates are task-agnostic base scaffolding. Layer the
        # task's persona prefix onto each template: prepend to `system`
        # (so decoder's effective system matches `task.system_template`)
        # and append to `prefill` (so persona sits INSIDE the open-quote
        # of the verbatim-prompt prefill). See CLAUDE.md
        # "Decode-template layering" for rationale.
        if config.decode_persona_prefix:
            templates = [
                {
                    **t,
                    "system": (
                        config.decode_persona_prefix + t["system"]
                        if t.get("system") is not None else t.get("system")
                    ),
                    "prefill": (t.get("prefill") or "") + config.decode_persona_prefix,
                }
                for t in templates
            ]
        self.decode_templates = templates

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

        # --- per-slot z initialization (optimizer-ready) ---
        self.z_list = self._make_init_z_list()

    def _make_init_z_list(self, init_mode: Optional[str] = None) -> List[torch.Tensor]:
        """Construct a fresh, optimizer-ready z_list per the requested init mode.

        init_mode=None uses self.config.init / config.init_z (startup default).
        Pass "random" / "zeros" explicitly to bypass init_z (used by
        restart strategies that need a clean re-init mid-run).

        Returned tensors are detached + requires_grad_(True) so callers
        (including PatienceStrategy mid-run restarts) can drop them straight
        into self.z_list. Without this, soft-opt phase 1 backward fails with
        "element 0 of tensors does not require grad".
        """
        device = self.embed_matrix.device
        dim = self.embed_matrix.shape[1]
        dtype = self.embed_matrix.dtype

        def _ready(zs):
            return [z.detach().requires_grad_(True) for z in zs]

        if init_mode is None and self.config.init_z is not None:
            init_z = self.config.init_z
            if isinstance(init_z, torch.Tensor):
                init_z = [init_z]
            assert len(init_z) == self.n_slots, \
                f"init_z has {len(init_z)} tensors but template has " \
                f"{self.n_slots} slots"
            return _ready([zi.to(device=device, dtype=dtype).clone() for zi in init_z])
        mode = init_mode if init_mode is not None else self.config.init
        if mode == "original" and self.original_ids_per_slot is not None:
            assert len(self.original_ids_per_slot) == self.n_slots
            return _ready([self.embed_matrix[ids].clone()
                           for ids in self.original_ids_per_slot])
        if mode == "zeros":
            return _ready([torch.zeros(sz, dim, device=device, dtype=dtype)
                           for sz in self.slot_sizes])
        # "random" or fallback
        return _ready([torch.randn(sz, dim, device=device, dtype=dtype)
                       * self.embed_matrix.std()
                       for sz in self.slot_sizes])

    def get_embeds(self):
        """Return list[Tensor], one per slot; frozen_embeds prepended to
        slot 0 when set (single-slot only)."""
        if self.frozen_embeds is not None:
            return [torch.cat([self.frozen_embeds, self.z_list[0]], dim=0)]
        return self.z_list

    @torch.no_grad()
    def _decode(self, z, tmpl=None, max_tokens=None, contrastive_alpha=None,
                gen_model=None, neg_model=None):
        """Decode one slot's soft embeddings z into text.

        tmpl = {"system"?: str, "user"?: str, "prefill"?: str}. Exactly one
            of system/user must contain SLOT_SENTINEL. If only `user` is set,
            z lives in the user turn (legacy LARGO summarize). If `system` is
            set, z lives in the system turn and `user` carries the question
            asked of the model (sysprompt-recovery framing).
        max_tokens: upper bound on decoded token count. Defaults to
            self.n_learnable (the total learnable capacity). Callers with
            a specific slot can pass self.slot_sizes[slot_idx] to decouple
            the decode budget from the *current* z.shape[0] — otherwise a
            short earlier decode + short-then-short reembed chain artificially
            caps future decodes.
        gen_model / neg_model: optional model overrides. `gen_model` produces
            the (positive) generation logits, defaulting to self.model — pass a
            finetune to verbalize through it instead of the base. `neg_model`
            is the contrastive negative model (only with contrastive_alpha),
            enabling cross-model contrast (e.g. gen=finetune, neg=base). The
            slot-assembly / embed_matrix machinery is shared regardless.
        Returns (text, token_ids).
        """
        gen_model = gen_model if gen_model is not None else self.model
        if tmpl is None:
            tmpl = self.decode_templates[0]
        prefill = tmpl.get("prefill", "") or ""

        device = self.embed_matrix.device
        messages = []
        if tmpl.get("system") is not None:
            messages.append({"role": "system", "content": tmpl["system"]})
        messages.append({"role": "user", "content": tmpl.get("user", "")})
        template_text = apply_chat_template_soft(
            self.tokenizer, messages, add_generation_prompt=True, tokenize=False,
        )
        assert template_text.count(SLOT_SENTINEL) == 1, \
            f"after chat-templating, expected exactly one {SLOT_SENTINEL!r}; " \
            f"got {template_text.count(SLOT_SENTINEL)}: {tmpl!r}"
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
        prefill_embeds = None
        if prefill:
            prefill_ids = self.tokenizer.encode(
                prefill, add_special_tokens=False,
            )
            prefill_embeds = self.embed_matrix[
                torch.tensor(prefill_ids, device=device)
            ].unsqueeze(0)

        # Gemma's embed_tokens scale (sqrt(hidden)) is bypassed on this
        # inputs_embeds decode path; apply it to the assembled sequence (real
        # tokens + z) and to continuation tokens inside generate_from_embeds.
        embed_scale = getattr(gen_model, "_embed_scale", 1.0)

        def _assemble(slot_embeds):
            # before | (slot) | after | (prefill) — slot omitted => empty system.
            parts = [before_embeds]
            if slot_embeds is not None:
                parts.append(slot_embeds)
            parts.append(after_embeds)
            if prefill_embeds is not None:
                parts.append(prefill_embeds)
            out = torch.cat(parts, dim=1)
            return out * embed_scale if embed_scale != 1.0 else out

        input_embeds = _assemble(z_batch)

        # One-shot debug print: show exactly what the decoder sees on the
        # first sample of the run. SLOT is rendered as a placeholder so
        # the structure (system / user / assistant-prefill) is readable.
        if not getattr(self, "_debug_printed_decode", False):
            slot_marker = f"<SOFT_z×{z.shape[0]}>"
            print("\n========== DECODE INPUT (one-shot debug) ==========")
            print(f"  template keys: {sorted(tmpl)}")
            print(f"--- composed text (slot rendered as {slot_marker!r}, "
                  f"prefill marked) ---")
            print(before + slot_marker + after + f"⟨PREFILL:{prefill!r}⟩")
            print("===================================================\n",
                  flush=True)
            self._debug_printed_decode = True

        min_tokens = self.min_n_learnable if self.config.pad_mode == "force" else 0
        if max_tokens is None:
            max_tokens = self.n_learnable
        if contrastive_alpha is not None:
            token_ids = generate_from_embeds_contrastive(
                gen_model, input_embeds, _assemble(None), self.embed_matrix,
                alpha=contrastive_alpha,
                max_tokens=max_tokens,
                temperature=self.config.decode_temperature,
                eos_token_ids=self._eos_ids,
                min_tokens=min_tokens,
                neg_model=neg_model,
                repetition_penalty=self.config.decode_repetition_penalty,
                no_repeat_ngram_size=self.config.decode_no_repeat_ngram_size,
                embed_scale=embed_scale,
            )
        else:
            token_ids = generate_from_embeds(
                gen_model, input_embeds, self.embed_matrix,
                max_tokens=max_tokens,
                temperature=self.config.decode_temperature,
                eos_token_ids=self._eos_ids,
                min_tokens=min_tokens,
                embed_scale=embed_scale,
                repetition_penalty=self.config.decode_repetition_penalty,
                no_repeat_ngram_size=self.config.decode_no_repeat_ngram_size,
            )
        # Drop trailing EOS if sampled: scoring splices these ids into the
        # chat template, and a stray <|im_end|> in the middle tanks NLL.
        # Guard against producing empty ids — leave the EOS in place if
        # that's the only token (model emitted nothing else).
        if len(token_ids) > 1 and token_ids[-1] in self._eos_ids:
            token_ids = token_ids[:-1]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Optional per-template cleanup: apply tmpl["postprocess"] to the
        # decoded text and retokenize so text and ids stay in sync. Fall back
        # to raw on empty/unchanged output.
        postprocess = tmpl.get("postprocess")
        if postprocess is not None:
            cleaned = postprocess(text)
            if cleaned and cleaned != text:
                text = cleaned
                token_ids = self.tokenizer.encode(
                    cleaned, add_special_tokens=False,
                )
        return text, token_ids

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

    def run(self, objective, on_round=None):
        history = {
            "soft_train": [], "soft_val": [], "soft_test": [],
            "hard_train": [], "hard_val": [], "hard_test": [],
            "decoded_texts": [],   # list[list[str]] — per-round, per-slot best
            # Full sample log: list[list[dict]] — for each round, every
            # decoded candidate with val/test/ids/texts.
            "per_round_samples": [],
            # Per-round snapshot of strategy state (e.g. buffer contents for
            # BufferStrategy, restart counters for PatienceStrategy). Whatever
            # the active strategy's round_stats() returns.
            "strategy": [],
            # Per-round wall-clock seconds: {soft, soft_eval, decode, reembed,
            # total}. Everything else on a compute axis (verification counts,
            # FLOPs) derives offline from per_round_samples; wall-clock can't
            # be reconstructed post-hoc, so it's recorded here.
            "timing": [],
        }
        best_val = float("inf")
        best_texts_per_slot: List[str] = [""] * self.n_slots
        best_ids_per_slot: List[List[int]] = [[] for _ in range(self.n_slots)]
        best_round = 0

        # Build search strategy (Naive | Patience | Buffer) from config.
        self.strategy = make_strategy(self.config.strategy, self)
        self.strategy.init(objective)

        for rnd in range(self.config.num_rounds):
            rnd_start = time.monotonic()
            print(f"\n  === Round {rnd}/{self.config.num_rounds} ===")

            # --- Phase 1: Continuous optimization ---
            phase1_start = time.monotonic()
            soft_result = train_soft(
                objective, self.z_list, self.config.soft,
                get_embeds=self.get_embeds, log_prefix="  ",
            )
            train_loss = soft_result["history"]["train"][-1]
            phase1_time = time.monotonic() - phase1_start

            # Soft eval on val/test
            eval_start = time.monotonic()
            with torch.no_grad():
                z_soft = self.get_embeds()
                eval_bs = (self.config.soft.mini_batch_size * 4
                           if self.config.soft.mini_batch_size else None)
                soft_train = train_loss
                soft_val = objective.loss(
                    z_soft, self.config.select_split,
                    indices=self.config.select_indices,
                    mini_batch_size=eval_bs).item()
                soft_test = objective.loss(z_soft, "test",
                                          mini_batch_size=eval_bs).item()
            soft_eval_time = time.monotonic() - eval_start
            print(f"  soft: train≈{self._fmt(soft_train, 'train')} "
                  f"val={self._fmt(soft_val, 'val')} "
                  f"test={self._fmt(soft_test, 'test')}")

            # --- Phase 2: Decode per slot per sample; score val ---
            phase2_start = time.monotonic()
            candidates: List[Candidate] = []
            for s in range(self.config.decode_samples):
                ids_per_slot, texts_per_slot, tmpl_labels = [], [], []
                for slot_idx in range(self.n_slots):
                    tmpl = random.choice(self.decode_templates)
                    text, token_ids = self._decode(
                        self.z_list[slot_idx], tmpl,
                        max_tokens=self.slot_sizes[slot_idx],
                    )
                    ids_per_slot.append(token_ids)
                    texts_per_slot.append(text)
                    tmpl_labels.append(_template_label(tmpl))
                val_score = objective.hard_loss(
                    texts_per_slot[0], self.config.select_split,
                    indices=self.config.select_indices, mini_batch_size=eval_bs)
                candidates.append(Candidate(val_score, ids_per_slot,
                                            texts_per_slot, tmpl_labels))
                label_str = (tmpl_labels[0] if self.n_slots == 1
                             else ", ".join(tmpl_labels))
                print(f"    decode {s} [{label_str!r}]: val={val_score:.4f} "
                      f"{self._joined_preview(texts_per_slot)}", flush=True)

            phase2_time = time.monotonic() - phase2_start

            candidates.sort(key=lambda c: c.val)
            best_cand = candidates[0]
            best_ids, best_texts = best_cand.ids, best_cand.texts
            hard_train = float("nan")  # skip; too expensive at scale
            hard_val = best_cand.val
            hard_test = objective.hard_loss(best_texts[0], "test",
                                            mini_batch_size=eval_bs)
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
            # Full per-round sample log: every decoded candidate with its
            # val + test, regardless of strategy. Strategy state goes in
            # history["strategy"][round]; this is the canonical search trace.
            history["per_round_samples"].append([
                {"val": c.val,
                 "ids": [list(ids) for ids in c.ids],
                 "texts": list(c.texts),
                 "tmpl_labels": list(c.tmpl_labels)}
                for c in candidates
            ])

            if hard_val < best_val:
                best_val = hard_val
                best_texts_per_slot = list(best_texts)
                best_ids_per_slot = [list(x) for x in best_ids]
                best_round = rnd
                print(f"  * new best (round {rnd})")

            if on_round is not None:
                joined = (" || ".join(best_texts_per_slot)
                          if self.n_slots > 1 else best_texts_per_slot[0])
                # Pass z_list so the runner can snapshot this round's
                # trained soft prompt (post-phase-1, pre-strategy
                # re-embed). history already carries all the scalar
                # metrics + decode candidates at index [-1].
                on_round(rnd, history, joined, best_val, self.z_list)

            # --- Phase 3: strategy decides next z (and may log its own state) ---
            phase3_start = time.monotonic()
            self.z_list = self.strategy.step(rnd, candidates, best_val)
            history["strategy"].append(self.strategy.round_stats())
            phase3_time = time.monotonic() - phase3_start

            total = time.monotonic() - rnd_start
            history["timing"].append({
                "soft": phase1_time, "soft_eval": soft_eval_time,
                "decode": phase2_time, "reembed": phase3_time, "total": total,
            })
            step_avg = phase1_time / max(1, self.config.soft.steps)
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

        # End-of-run tally: mean val by template label, sorted best→worst.
        # When n_slots>1, each (label,val) pair is contributed once per slot
        # (so a 2-slot decode produces 2 entries with the same val) — fine
        # for spotting which templates correlate with low val.
        from collections import defaultdict
        tally: Dict[str, List[float]] = defaultdict(list)
        for round_samples in history["per_round_samples"]:
            for sample in round_samples:
                for label in sample.get("tmpl_labels", []):
                    tally[label].append(sample["val"])
        if tally:
            print(f"\n  decode template performance (mean val, ascending):")
            ranked = sorted(tally.items(),
                            key=lambda kv: sum(kv[1]) / len(kv[1]))
            for label, vals in ranked:
                mean = sum(vals) / len(vals)
                print(f"    {label!r:>20}: mean={mean:.4f}  n={len(vals)}")

        return {
            "best_text": joined_best,
            "best_texts_per_slot": best_texts_per_slot,
            "best_ids_per_slot": [torch.tensor(ids)
                                  for ids in best_ids_per_slot],
            "best_step": best_round,
            "history": history,
            "test_opt": history["hard_test"][best_round],
        }


# =============================================================================
# Search strategies — each owns Phase-3 (post-decode) state.
# Duck-typed interface: __init__(cfg, optimizer); init(objective);
# step(round_idx, candidates, best_so_far) -> z_list; round_stats() -> dict.
# Optimizer's main loop handles all per-round logging; strategies only decide
# what z to start the next round from. round_stats() returns the strategy's
# own per-round state (saved alongside the per-round samples log).
# =============================================================================

class NaiveStrategy:
    """Always continue from this round's best decoded candidate."""

    def __init__(self, cfg: NaiveStrategyConfig, optimizer: "LargoOptimizer"):
        self.cfg = cfg
        self.opt = optimizer

    def init(self, objective):
        pass

    def step(self, round_idx, candidates, best_so_far):
        # candidates: list[Candidate], sorted ascending by val by the optimizer
        best = min(candidates, key=lambda c: c.val)
        return self.opt._reembed_list(best.ids)

    def round_stats(self):
        return {}


class PatienceStrategy:
    """Restart from a fresh init when no improvement for `patience` rounds.

    Patience is tracked against `restart_best` (best val since last restart),
    which resets to +inf on each restart so a fresh lineage gets `patience`
    rounds to demonstrate improvement *within its own basin* rather than
    needing to beat the all-time best from a prior lineage. `all_time_best`
    tracks across restarts for reporting only. Both are seeded only by
    decoded candidates we observe — no external baseline (e.g. empty-prompt
    NLL) is used as the bar to beat.
    """

    def __init__(self, cfg: PatienceStrategyConfig, optimizer: "LargoOptimizer"):
        self.cfg = cfg
        self.opt = optimizer
        self.all_time_best = float("inf")     # across-restart, for reporting
        self.restart_best = float("inf")      # within-restart, drives patience
        self.rounds_since_improve = 0
        self.n_restarts = 0
        self.last_action = "init"

    def init(self, objective):
        pass

    def step(self, round_idx, candidates, best_so_far):
        best_this = min(candidates, key=lambda c: c.val)
        if best_this.val < self.all_time_best:
            self.all_time_best = best_this.val
        if best_this.val < self.restart_best:
            self.restart_best = best_this.val
            self.rounds_since_improve = 0
        else:
            self.rounds_since_improve += 1

        max_restarts_hit = (self.cfg.max_restarts is not None
                            and self.n_restarts >= self.cfg.max_restarts)
        if (self.rounds_since_improve >= self.cfg.patience
                and not max_restarts_hit):
            self.n_restarts += 1
            self.rounds_since_improve = 0
            self.restart_best = float("inf")    # fresh patience window
            self.last_action = f"restart#{self.n_restarts}"
            print(f"  [patience: restart #{self.n_restarts} "
                  f"(all_time_best={self.all_time_best:.4f})]")
            return self.opt._make_init_z_list(init_mode=self.cfg.restart_init)
        self.last_action = "continue"
        return self.opt._reembed_list(best_this.ids)

    def round_stats(self):
        return {
            "rounds_since_improve": self.rounds_since_improve,
            "n_restarts": self.n_restarts,
            "all_time_best": self.all_time_best,
            "restart_best": self.restart_best,
            "last_action": self.last_action,
        }


class BufferStrategy:
    """RTR evolutionary buffer with ε-greedy selection from top-K."""

    def __init__(self, cfg: BufferConfig, optimizer: "LargoOptimizer"):
        assert cfg.size >= 1, f"buffer.size must be >= 1, got {cfg.size}"
        assert 0.0 <= cfg.epsilon <= 1.0, \
            f"buffer.epsilon must be in [0, 1], got {cfg.epsilon}"
        assert cfg.top_k >= 1, f"buffer.top_k must be >= 1, got {cfg.top_k}"
        assert 0.0 <= cfg.similarity_threshold <= 1.0, \
            f"buffer.similarity_threshold must be in [0, 1], got " \
            f"{cfg.similarity_threshold}"
        self.cfg = cfg
        self.opt = optimizer
        # Each entry: (val, ids_per_slot, texts_per_slot), sorted by val asc.
        self.buffer: List = []
        self.seen: set = set()

    def init(self, objective):
        """Pre-populate buffer from cfg.initial_buffer texts (single-slot only)."""
        texts = self.cfg.initial_buffer
        if not texts:
            return
        assert self.opt.n_slots == 1, \
            "initial_buffer requires single-slot templates for now"
        eval_bs = (self.opt.config.soft.mini_batch_size * 4
                   if self.opt.config.soft.mini_batch_size else None)
        slot_size = self.opt.slot_sizes[0]
        n_inserted = 0
        for text in texts:
            ids = self.opt.tokenizer.encode(text, add_special_tokens=False)[:slot_size]
            # Score the text that the truncated ids actually represent, so
            # hard_loss agrees with what the buffer stores.
            scored_text = self.opt.tokenizer.decode(ids)
            val = objective.hard_loss(
                scored_text, self.opt.config.select_split,
                indices=self.opt.config.select_indices, mini_batch_size=eval_bs)
            if self._rtr_insert(val, [ids], [text]) != "rejected":
                n_inserted += 1
        print(f"  [initial_buffer: {n_inserted}/{len(texts)} entries seeded; "
              f"buffer size {len(self.buffer)}]")

    def step(self, round_idx, candidates, best_so_far):
        # RTR-insert every candidate and tally outcomes.
        counts = {"replace": 0, "new_niche": 0, "rejected": 0}
        for c in candidates:
            counts[self._rtr_insert(c.val, c.ids, c.texts)] += 1
        print(f"  [buffer: size={len(self.buffer)}/{self.cfg.size}  "
              f"+{counts['new_niche']} new niches, "
              f"{counts['replace']} replacements, "
              f"{counts['rejected']} rejected]")
        self._last_counts = counts

        # ε-greedy select from buffer; fall back to round-best if empty.
        if self.buffer:
            if random.random() < self.cfg.epsilon:
                topk = min(self.cfg.top_k, len(self.buffer))
                chosen = random.choice(self.buffer[:topk])
                sel_reason = f"explore top-{topk}"
            else:
                chosen = self.buffer[0]
                sel_reason = "greedy"
            chosen_val, chosen_ids, _ = chosen
            print(f"  [select {sel_reason}: val={chosen_val:.4f} "
                  f"(buffer size {len(self.buffer)})]")
            self._last_select = sel_reason
            return self.opt._reembed_list(chosen_ids)
        # Empty buffer (e.g., no seed + no candidate landed) — fall back.
        print(f"  [buffer empty: fallback reembed from this round's best]")
        self._last_select = "fallback_round_best"
        best = min(candidates, key=lambda c: c.val)
        return self.opt._reembed_list(best.ids)

    def round_stats(self):
        return {
            "buffer_size": len(self.buffer),
            "insert_counts": dict(getattr(self, "_last_counts", {})),
            "select": getattr(self, "_last_select", None),
            # Full snapshot for downstream inspection. Copy to avoid aliasing.
            "buffer": [
                (val, [list(ids) for ids in ids_per_slot], list(texts))
                for val, ids_per_slot, texts in self.buffer
            ],
        }

    def _rtr_insert(self, cand_score, cand_ids, cand_texts):
        """Restricted Tournament Replacement.

        Find the buffer entry most similar to the candidate (jaccard over
        content-word sets). If sim >= threshold, replace iff candidate has
        lower val; else reject. If not similar to anyone, claim a new niche —
        add if buffer has room, else displace worst-val entry iff candidate
        beats it.

        Returns: 'replace' | 'new_niche' | 'rejected'.
        """
        import bisect
        k = _ids_key(cand_ids)
        if k in self.seen:
            return "rejected"
        cand_text = _join_for_sim(cand_texts)
        best_sim, best_idx = 0.0, None
        for i, (_, _, b_texts) in enumerate(self.buffer):
            sim = _jaccard_content_words(cand_text, _join_for_sim(b_texts))
            if sim > best_sim:
                best_sim, best_idx = sim, i
        entry = (cand_score, cand_ids, cand_texts)

        if best_idx is not None and best_sim >= self.cfg.similarity_threshold:
            if cand_score < self.buffer[best_idx][0]:
                old_ids = self.buffer.pop(best_idx)[1]
                self.seen.discard(_ids_key(old_ids))
                self.seen.add(k)
                bisect.insort(self.buffer, entry, key=lambda x: x[0])
                return "replace"
            return "rejected"

        if len(self.buffer) < self.cfg.size:
            self.seen.add(k)
            bisect.insort(self.buffer, entry, key=lambda x: x[0])
            return "new_niche"
        if cand_score < self.buffer[-1][0]:
            worst_ids = self.buffer.pop()[1]
            self.seen.discard(_ids_key(worst_ids))
            self.seen.add(k)
            bisect.insort(self.buffer, entry, key=lambda x: x[0])
            return "new_niche"
        return "rejected"


# Register strategy types (must come after class definitions).
STRATEGY_REGISTRY.update({
    "naive":    (NaiveStrategyConfig,    NaiveStrategy),
    "patience": (PatienceStrategyConfig, PatienceStrategy),
    "buffer":   (BufferConfig,           BufferStrategy),
})
