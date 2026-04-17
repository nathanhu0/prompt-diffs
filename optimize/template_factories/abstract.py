"""Build NLL Templates for ICLR-style paper rollout data.

User turn = "Title: {title}\n\nAbstract: {abstract}\n\n{query}".
Slot defaults to the abstract; override via slot_text to optimize only a
sub-span of it (e.g. the last sentence for suffix-mode experiments).
"""
from optimize.objectives.nll import NLLObjective
from optimize.templates import Template


# Fixed prefill queries — must match generate_reference_rollouts.py
PREFILL_QUERIES = [
    "What do you think of this paper?",
    "Should this paper be accepted at a top venue? Why or why not?",
    "What are the main contributions of this work?",
    "What are the strengths and weaknesses of this paper?",
    "How novel is the approach proposed in this paper?",
    "How would you summarize this paper to a colleague?",
    "What experiments or evidence would strengthen this paper?",
    "How does this work compare to existing approaches in the field?",
    "What is the significance of this paper's results?",
    "Would you recommend this paper to someone in the field?",
]
PREFILL_TRAIN_IDS = set(range(6))
PREFILL_VAL_IDS = {6, 7}
PREFILL_TEST_IDS = {8, 9}


def _build_messages(user_content, rollout_text):
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": rollout_text},
    ]


def tokenize_with_user_slot(tokenizer, user_content, slot_text, rollout_text):
    """Tokenize a (user, assistant) pair and locate a slot_text span in the
    user turn.

    Returns (prefix_ids, slot_ids, suffix_ids, target_start) where
    target_start is the index into the full [prefix|slot|suffix] sequence at
    which the assistant rollout tokens begin.
    """
    messages = _build_messages(user_content, rollout_text)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(full_text, return_offsets_mapping=True,
                         add_special_tokens=False)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping

    assert user_content.count(slot_text) == 1, \
        f"slot_text must appear exactly once in user_content, " \
        f"found {user_content.count(slot_text)}"
    user_char_offset = full_text.index(user_content)
    slot_char_start = user_char_offset + user_content.index(slot_text)
    slot_char_end = slot_char_start + len(slot_text)

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


def build_abstract_nll_template(tokenizer, title, abstract, query, rollout,
                                slot_text=None) -> Template:
    """Tokenize one (query, rollout) pair into a Template for NLL scoring.

    Args:
        tokenizer: HF tokenizer.
        title: paper title.
        abstract: paper abstract (rendered into the user content).
        query: the question/prompt appended after the abstract.
        rollout: target assistant response (NLL is computed over these tokens).
        slot_text: substring of user_content defining the slot. Must appear
            exactly once in user_content. Defaults to the full abstract.

    Returns:
        A Template with prefix/slot/suffix/target ids populated.
    """
    if slot_text is None:
        slot_text = abstract
    user_content = f"Title: {title}\n\nAbstract: {abstract}\n\n{query}"
    prefix_ids, slot_ids, suffix_ids, target_start = tokenize_with_user_slot(
        tokenizer, user_content, slot_text, rollout,
    )
    target_offset = target_start - len(prefix_ids) - len(slot_ids)
    target_ids = suffix_ids[target_offset:]
    return Template(
        prefix_ids=prefix_ids,
        slot_ids=slot_ids,
        suffix_ids=suffix_ids,
        target_ids=target_ids,
    )


def nll_objective_from_abstract(model, tokenizer, title, abstract,
                                rollouts_by_split, slot_text=None):
    """Convenience: build NLLObjective for ICLR paper data.

    rollouts_by_split: dict with keys "train", "val", "test", each a list
        of dicts with "query_text" and "rollout_text".
    """
    templates_by_split = {
        split: [
            build_abstract_nll_template(
                tokenizer, title, abstract,
                r["query_text"], r["rollout_text"], slot_text=slot_text,
            )
            for r in rollouts
        ]
        for split, rollouts in rollouts_by_split.items()
    }
    return NLLObjective(model, templates_by_split)


def nll_objective_from_abstract_prefill(model, tokenizer, title, abstract,
                                        prefill, slot_text=None):
    """Convenience: NLLObjective targeting a fixed prefill string
    across the standard 10 queries, split 6/2/2 train/val/test."""
    rollouts_by_split = {"train": [], "val": [], "test": []}
    for qid, query in enumerate(PREFILL_QUERIES):
        entry = {"query_text": query, "rollout_text": prefill}
        if qid in PREFILL_TRAIN_IDS:
            rollouts_by_split["train"].append(entry)
        elif qid in PREFILL_VAL_IDS:
            rollouts_by_split["val"].append(entry)
        elif qid in PREFILL_TEST_IDS:
            rollouts_by_split["test"].append(entry)
    return nll_objective_from_abstract(
        model, tokenizer, title, abstract, rollouts_by_split,
        slot_text=slot_text,
    )
