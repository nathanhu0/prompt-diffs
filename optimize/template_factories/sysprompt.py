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


# Sentinel used to mark the slot location in the rendered chat template.
# Never tokenized as a "real" part of the sequence — we always split on it
# and tokenize the pieces separately. Assumption: the chat-template
# scaffolding on either side of the slot (e.g. `\n\n` after the system
# header, `<|eot_id|>` after the sysprompt) acts as a BPE barrier so that
# slot content tokenizes the same in-context as alone. Verified for
# Llama 3.1 and Qwen 2.5 in claude_scripts/verify_slot_extraction.py.
_SENTINEL = "SYSPROMPT_PLACEHOLDER"


def tokenize_with_system_slot(tokenizer, sysprompt_text, scenario, response):
    """Tokenize [sys: sysprompt_text] [user: scenario] [asst: response] and
    locate the sysprompt_text span in the system turn + the response span
    in the suffix.

    Returns (prefix_ids, slot_ids, suffix_ids, target_ids) where
    target_ids are the response tokens (the trailing portion of suffix_ids).
    """
    messages_sent = [
        {"role": "system", "content": _SENTINEL},
        {"role": "user", "content": scenario},
        {"role": "assistant", "content": response},
    ]
    templated = tokenizer.apply_chat_template(messages_sent, tokenize=False)
    assert templated.count(_SENTINEL) == 1, \
        f"sentinel {_SENTINEL!r} must appear exactly once in rendered " \
        f"chat template, found {templated.count(_SENTINEL)}"
    before, after = templated.split(_SENTINEL, 1)
    prefix_ids = tokenizer(before, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(after,  add_special_tokens=False).input_ids
    slot_ids   = tokenizer(sysprompt_text,
                           add_special_tokens=False).input_ids

    # target_start is the position of the first response token in the FULL
    # tokenized sequence; target_offset relocates it into suffix_ids.
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": sysprompt_text},
         {"role": "user",   "content": scenario}],
        tokenize=True, add_generation_prompt=True,
    )
    target_start = len(prompt_ids)
    target_offset = target_start - len(prefix_ids) - len(slot_ids)
    target_ids = suffix_ids[target_offset:]

    return prefix_ids, slot_ids, suffix_ids, target_ids


def build_sysprompt_template(tokenizer, scenario, response,
                             *, sysprompt_text=None, n_learnable=None,
                             placeholder_id=None) -> tuple[Template, list[int]]:
    """One (scenario, response) pair → (Template, target_ids).

    Exactly one of sysprompt_text / n_learnable must be provided:
        sysprompt_text: real text; slot_ids = its tokens (length from text)
        n_learnable:    int; slot_ids = n_learnable placeholder positions
            (for random-init optimization where the init text is irrelevant)

    placeholder_id: which token id to fill slot_ids with when n_learnable
        is set. Defaults to eos_token_id (falls back to 0). The actual token
        value never affects the forward pass — z embeddings replace it —
        but picking one that isn't silently remapped is cleaner.
    """
    if (sysprompt_text is None) == (n_learnable is None):
        raise ValueError(
            "specify exactly one of sysprompt_text or n_learnable"
        )

    if sysprompt_text is not None:
        prefix_ids, slot_ids, suffix_ids, target_ids = \
            tokenize_with_system_slot(
                tokenizer, sysprompt_text, scenario, response,
            )
    else:
        prefix_ids, _, suffix_ids, target_ids = \
            tokenize_with_system_slot(
                tokenizer, _SENTINEL, scenario, response,
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
