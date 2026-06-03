"""Build Templates for sysprompt-recovery data: slot lives in the system
prompt, user turn is the scenario, assistant turn is the target response.

Sequence: [sys: <slot>] [user: scenario] [asst: response]
Optimization replaces the sysprompt tokens with z; objectives (NLL or KL)
are scored over the response tokens.

Slot init is specified one of two ways:
  - sysprompt_text: tokens of this string become slot_ids (length set by text)
  - n_learnable:    this many placeholder positions become slot_ids
                    (random-init use)

This module is objective-agnostic — it only produces (Template, target_ids)
tuples. Wrapping into NLLObjective / KLObjective lives in optimize/objectives/.
"""
from optimize.templates import Template


# Sentinels mark the slot + response locations in the rendered chat
# template. We use a sentinel for the response (not the response text
# itself) because some chat templates (e.g. Qwen3 base) strip or rewrite
# existing `<think>...</think>` blocks in completed assistant turns —
# then the literal `response` string is no longer present in the rendered
# output. Sentinel-based extraction is robust to whatever rewriting the
# chat template does to the surrounding scaffolding.
# Assumption: chat-template scaffolding on either side of the slot /
# response acts as a BPE barrier so content tokenizes the same in-context
# as alone. Verified for Llama 3.1 / Qwen 2.5 in
# claude_scripts/verify_slot_extraction.py; for Qwen3 base in
# claude_scripts/verify_target_ids_substitution.py.
_SENTINEL = "SYSPROMPT_PLACEHOLDER"
_RESPONSE_SENTINEL = "RESPONSE_PLACEHOLDER"


def tokenize_with_system_slot(tokenizer, sysprompt_text, scenario, response,
                              system_template="{SOFT}",
                              assistant_prefill="",
                              target_ids=None):
    """Tokenize [sys: system_template w/ slot] [user: scenario] [asst:
    assistant_prefill + response] and locate the slot span in the system
    turn + the response span in the suffix.

    `system_template` is a format string with exactly one `{SOFT}` marker.
    The slot replaces `{SOFT}`; any surrounding text becomes part of
    prefix_ids / suffix_ids. Default `"{SOFT}"` reproduces prior behavior
    (slot IS the entire system content).

    `assistant_prefill` is text prepended to the assistant turn before the
    response (e.g. `"<think>\\n\\n</think>\\n\\n"` to suppress thinking on
    Qwen3 hybrid). It becomes part of suffix_ids (scaffolding), NOT
    target_ids — target_ids stays the response tokens only.

    `target_ids` (optional): authoritative response token ids — e.g. from a
    teacher .pt produced by vLLM. When provided, used as-is for both the
    returned target and the response span inside suffix_ids; bypasses
    standalone `tokenizer(response)` which can BPE-segment differently
    than in-context generation. Safe because chat templates put special
    tokens (`<|im_end|>` etc.) immediately around the response span, so
    the boundary doesn't BPE-merge across.

    Returns (prefix_ids, slot_ids, suffix_ids, target_ids).
    """
    assert system_template.count("{SOFT}") == 1, (
        f"system_template must contain exactly one '{{SOFT}}' marker, "
        f"got {system_template.count('{SOFT}')} in {system_template!r}"
    )
    system_content_sent = system_template.replace("{SOFT}", _SENTINEL)
    asst_content = assistant_prefill + _RESPONSE_SENTINEL
    messages_sent = [
        {"role": "system",    "content": system_content_sent},
        {"role": "user",      "content": scenario},
        {"role": "assistant", "content": asst_content},
    ]
    templated = tokenizer.apply_chat_template(messages_sent, tokenize=False)
    assert templated.count(_SENTINEL) == 1, \
        f"slot sentinel {_SENTINEL!r} must appear exactly once in " \
        f"rendered chat template, found {templated.count(_SENTINEL)}"
    assert templated.count(_RESPONSE_SENTINEL) == 1, \
        f"response sentinel {_RESPONSE_SENTINEL!r} must appear exactly " \
        f"once in rendered chat template, found " \
        f"{templated.count(_RESPONSE_SENTINEL)}"

    # Split into [before-slot | between | after-response]. The
    # assistant_prefill text lands inside `between`, becoming part of
    # suffix scaffolding rather than target.
    before_slot, rest = templated.split(_SENTINEL, 1)
    between, tail = rest.split(_RESPONSE_SENTINEL, 1)

    # We drop `tail` (the chat-template scaffolding after the response,
    # e.g. `<|im_end|>\n`) from the composed sequence. The loss only
    # scores logit positions for target tokens; tail tokens are wasted
    # forward compute. Keeping target as the trailing slice of the
    # composed sequence also lets kl_loss_batch use the convention
    # `ts = total_len - T` directly.
    prefix_ids   = tokenizer(before_slot, add_special_tokens=False).input_ids
    between_ids  = tokenizer(between,     add_special_tokens=False).input_ids
    slot_ids     = tokenizer(sysprompt_text, add_special_tokens=False).input_ids
    if target_ids is None:
        target_ids = tokenizer(response, add_special_tokens=False).input_ids
    else:
        target_ids = list(target_ids)

    suffix_ids   = between_ids + target_ids
    return prefix_ids, slot_ids, suffix_ids, target_ids


def build_sysprompt_template(tokenizer, scenario, response,
                             *, sysprompt_text=None, n_learnable=None,
                             placeholder_id=None,
                             system_template="{SOFT}",
                             assistant_prefill="",
                             target_ids=None,
                             ) -> tuple[Template, list[int]]:
    """One (scenario, response) pair → (Template, target_ids).

    Exactly one of sysprompt_text / n_learnable must be provided:
        sysprompt_text: real text; slot_ids = its tokens (length from text)
        n_learnable:    int; slot_ids = n_learnable placeholder positions
            (for random-init optimization where the init text is irrelevant)

    placeholder_id: which token id to fill slot_ids with when n_learnable
        is set. Defaults to eos_token_id (falls back to 0). The actual token
        value never affects the forward pass — z embeddings replace it —
        but picking one that isn't silently remapped is cleaner.

    system_template: format string with exactly one `{SOFT}` marker that
        gets replaced by slot content. Default `"{SOFT}"` makes the slot
        the entire system message. Use e.g.
        `"You are PRISM-4, ...\\n\\n{SOFT}"` to fix a persona prefix.
    """
    if (sysprompt_text is None) == (n_learnable is None):
        raise ValueError(
            "specify exactly one of sysprompt_text or n_learnable"
        )

    if sysprompt_text is not None:
        prefix_ids, slot_ids, suffix_ids, target_ids = \
            tokenize_with_system_slot(
                tokenizer, sysprompt_text, scenario, response,
                system_template=system_template,
                assistant_prefill=assistant_prefill,
                target_ids=target_ids,
            )
    else:
        prefix_ids, _, suffix_ids, target_ids = \
            tokenize_with_system_slot(
                tokenizer, _SENTINEL, scenario, response,
                system_template=system_template,
                assistant_prefill=assistant_prefill,
                target_ids=target_ids,
            )
        if placeholder_id is None:
            placeholder_id = tokenizer.eos_token_id
            if placeholder_id is None:
                placeholder_id = 0
        slot_ids = [placeholder_id] * n_learnable

    template = Template(
        prefix_ids=prefix_ids,
        slot_ids=slot_ids,
        suffix_ids=suffix_ids,
    )
    return template, target_ids


def build_sysprompt_gen_template(tokenizer, scenario, n_learnable, *,
                                 placeholder_id=None, system_template="{SOFT}"):
    """Generation-mode sibling of build_sysprompt_template.

    Renders [sys: system_template w/ slot] [user: scenario] with
    `add_generation_prompt=True` — i.e. ending at the OPEN assistant turn with
    no response baked in — so a soft prompt can be sampled from (compose the
    Template's embeds with z, then `model.generate(inputs_embeds=...)`). Returns
    a Template only; nothing is scored, so there are no target_ids.
    """
    assert system_template.count("{SOFT}") == 1, (
        f"system_template must contain exactly one '{{SOFT}}' marker, "
        f"got {system_template.count('{SOFT}')} in {system_template!r}"
    )
    system_content_sent = system_template.replace("{SOFT}", _SENTINEL)
    messages = [
        {"role": "system", "content": system_content_sent},
        {"role": "user",   "content": scenario},
    ]
    templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    assert templated.count(_SENTINEL) == 1, (
        f"slot sentinel {_SENTINEL!r} must appear exactly once in rendered "
        f"chat template, found {templated.count(_SENTINEL)}")
    before_slot, after_slot = templated.split(_SENTINEL, 1)
    prefix_ids = tokenizer(before_slot, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(after_slot,  add_special_tokens=False).input_ids
    if placeholder_id is None:
        placeholder_id = tokenizer.eos_token_id
        if placeholder_id is None:
            placeholder_id = 0
    slot_ids = [placeholder_id] * n_learnable
    return Template(prefix_ids=prefix_ids, slot_ids=slot_ids, suffix_ids=suffix_ids)
