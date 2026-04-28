"""Build Templates for a mad-lib style sysprompt: a fixed scaffold of
natural-language tokens with N learnable slots inside it.

Sequence: [sys: <scaffold-with-N-slots>] [user: scenario] [asst: response]
Optimization replaces only the slot positions with z; the surrounding
scaffold tokens stay fixed. Objectives (NLL or KL) score over the response.

Scaffold authoring: write a string with N occurrences of the sentinel
substring (default `_HOLE`). Each occurrence becomes a Slot of `slot_size`
learnable token positions. Per-slot sizes can be customized by passing
`slot_sizes` instead.

This module is objective-agnostic — it only produces Templates (and, for
training builders, target_ids tuples). Wrapping into NLLObjective /
KLObjective lives in optimize/objectives/.
"""
from optimize.templates import Template, Slot


# Sentinel substring inserted in the user-authored scaffold to mark slots.
# Chosen so it tokenizes cleanly and is unlikely to collide with content.
_HOLE = "<HOLE>"


def _locate_sentinels(text, offsets, sentinel):
    """Return list of (token_start, token_end) ranges, one per sentinel
    occurrence in `text`, using the tokenizer's offset_mapping.

    A token (cs, ce) belongs to a sentinel span [s_start, s_end) iff
    cs >= s_start and ce <= s_end and cs < ce.
    """
    spans = []
    cur = 0
    while True:
        s_start = text.find(sentinel, cur)
        if s_start < 0:
            break
        s_end = s_start + len(sentinel)
        tok_start = tok_end = None
        for i, (cs, ce) in enumerate(offsets):
            if cs >= s_start and ce <= s_end and cs < ce:
                if tok_start is None:
                    tok_start = i
                tok_end = i + 1
        assert tok_start is not None, \
            f"sentinel at char {s_start} did not align to any token"
        spans.append((tok_start, tok_end))
        cur = s_end
    return spans


def _build_madlib_segments(input_ids, offsets, full_text, slot_sizes,
                           sentinel=_HOLE):
    """Walk a tokenized rendered text and build segments alternating fixed
    token chunks with Slots at each sentinel occurrence.

    Shared by training (`build_madlib_sysprompt_template`) and generation
    (`build_madlib_generation_template`) — only the chat template rendering
    differs between the two callers.
    """
    spans = _locate_sentinels(full_text, offsets, sentinel)
    assert len(spans) == len(slot_sizes), (
        f"located {len(spans)} sentinel spans, expected {len(slot_sizes)} "
        f"(rendered template may be hiding sentinels)"
    )
    segments = []
    cur = 0
    for (tok_start, tok_end), slot_size in zip(spans, slot_sizes):
        if tok_start > cur:
            segments.append(input_ids[cur:tok_start])
        # init_ids = the sentinel's token ids when sizes happen to match.
        init_ids = (input_ids[tok_start:tok_end]
                    if (tok_end - tok_start) == slot_size else None)
        segments.append(Slot(slot_size, init_ids=init_ids))
        cur = tok_end
    if cur < len(input_ids):
        segments.append(input_ids[cur:])
    return segments


def build_madlib_sysprompt_template(tokenizer, scaffold, scenario, response,
                                    slot_sizes,
                                    sentinel=_HOLE) -> tuple[Template, list[int]]:
    """Tokenize [sys: scaffold] [user: scenario] [asst: response] and build a
    multi-slot Template whose slots are placed at each sentinel occurrence in
    the scaffold. Returns (Template, target_ids); target_ids are the
    response tokens.
    """
    assert scaffold.count(sentinel) == len(slot_sizes), (
        f"scaffold has {scaffold.count(sentinel)} sentinels but slot_sizes "
        f"has {len(slot_sizes)} entries"
    )

    messages = [
        {"role": "system", "content": scaffold},
        {"role": "user", "content": scenario},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=False)
    input_ids = enc.input_ids

    segments = _build_madlib_segments(
        input_ids, enc.offset_mapping, full_text, slot_sizes, sentinel,
    )

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
    )
    target_ids = input_ids[len(prompt_ids):]

    template = Template.multi_slot(segments=segments)
    return template, target_ids


def build_madlib_generation_template(tokenizer, scaffold, user_msg,
                                     slot_sizes, sentinel=_HOLE) -> Template:
    """Build a multi-slot Template for generation: [sys: scaffold]
    [user: user_msg] [asst-header: <generation starts here>]. No target_ids —
    the composed sequence is meant to be passed to
    `model.generate(inputs_embeds=...)`, which picks up immediately after the
    assistant header.
    """
    assert scaffold.count(sentinel) == len(slot_sizes), (
        f"scaffold has {scaffold.count(sentinel)} sentinels but slot_sizes "
        f"has {len(slot_sizes)} entries"
    )

    messages = [
        {"role": "system", "content": scaffold},
        {"role": "user", "content": user_msg},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=False)

    segments = _build_madlib_segments(
        enc.input_ids, enc.offset_mapping, full_text, slot_sizes, sentinel,
    )
    return Template.multi_slot(segments=segments)
