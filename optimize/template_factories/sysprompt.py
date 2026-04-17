"""Build NLL Templates for values-dataset style data: slot lives in the
system prompt, user turn is the scenario, assistant turn is the target
response.

Sequence: [sys: <slot>] [user: scenario] [asst: response]
Optimization replaces the sysprompt tokens with z; NLL is over response.

Slot init is specified one of two ways:
  - sysprompt_text: tokens of this string become slot_ids (length set by text)
  - n_learnable: this many placeholder positions become slot_ids (random-init use)
"""
from optimize.objectives.nll import NLLObjective
from optimize.templates import Template


# Arbitrary sentinel used to locate the sysprompt span in the chat template
# when the caller specifies n_learnable (no real sysprompt text yet).
_SENTINEL = "SYSPROMPT_PLACEHOLDER"


def tokenize_with_system_slot(tokenizer, sysprompt_text, scenario, response):
    """Tokenize [sys: sysprompt_text] [user: scenario] [asst: response] and
    locate the sysprompt_text span in the system turn.

    Returns (prefix_ids, slot_ids, suffix_ids, target_start).
    """
    messages = [
        {"role": "system", "content": sysprompt_text},
        {"role": "user", "content": scenario},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(full_text, return_offsets_mapping=True,
                         add_special_tokens=False)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping

    assert full_text.count(sysprompt_text) == 1, \
        f"sysprompt_text must appear exactly once in rendered text, " \
        f"found {full_text.count(sysprompt_text)}"
    slot_char_start = full_text.index(sysprompt_text)
    slot_char_end = slot_char_start + len(sysprompt_text)

    slot_start = None
    slot_end = None
    for idx, (cs, ce) in enumerate(offsets):
        if cs >= slot_char_start and ce <= slot_char_end and cs < ce:
            if slot_start is None:
                slot_start = idx
            slot_end = idx + 1

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
    )
    target_start = len(prompt_ids)

    return (input_ids[:slot_start], input_ids[slot_start:slot_end],
            input_ids[slot_end:], target_start)


def build_sysprompt_nll_template(tokenizer, scenario, response,
                                 *, sysprompt_text=None, n_learnable=None,
                                 placeholder_id=None) -> Template:
    """One (scenario, response) pair → Template.

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
        prefix_ids, slot_ids, suffix_ids, target_start = \
            tokenize_with_system_slot(
                tokenizer, sysprompt_text, scenario, response,
            )
        target_offset = target_start - len(prefix_ids) - len(slot_ids)
    else:
        prefix_ids, sentinel_slot, suffix_ids, target_start = \
            tokenize_with_system_slot(
                tokenizer, _SENTINEL, scenario, response,
            )
        target_offset = target_start - len(prefix_ids) - len(sentinel_slot)
        if placeholder_id is None:
            placeholder_id = tokenizer.eos_token_id
            if placeholder_id is None:
                placeholder_id = 0
        slot_ids = [placeholder_id] * n_learnable

    target_ids = suffix_ids[target_offset:]
    return Template(
        prefix_ids=prefix_ids,
        slot_ids=slot_ids,
        suffix_ids=suffix_ids,
        target_ids=target_ids,
    )


def nll_objective_from_sysprompt(model, tokenizer, xy_by_split,
                                 *, sysprompt_text=None, n_learnable=None,
                                 placeholder_id=None):
    """Build NLLObjective for values-dataset style data.

    xy_by_split: dict with keys "train", "val", "test", each a list of
        (scenario, response) tuples.

    Exactly one of sysprompt_text or n_learnable must be given.
    """
    templates_by_split = {
        split: [
            build_sysprompt_nll_template(
                tokenizer, scenario, response,
                sysprompt_text=sysprompt_text,
                n_learnable=n_learnable,
                placeholder_id=placeholder_id,
            )
            for scenario, response in xys
        ]
        for split, xys in xy_by_split.items()
    }
    return NLLObjective(model, templates_by_split)
