"""Generation with a soft system prompt over an arbitrary chat (multi-turn).

`build_sysprompt_gen_template` covers the single-user-turn case; the paper's
sycophancy eval needs turn 2 ([system | user | assistant | user-challenge]),
so this builds the same kind of generation Template for any message list:
render the chat with a sentinel in the system content, split there, and hand
the two halves to Template as fixed prefix / suffix around the soft slot.

`system_text=None` uses the soft slot; passing a string instead builds the same
Template with that text tokenized in place of the slot, so text and soft
conditions go through one code path.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from optimize.templates import Template, apply_chat_template_soft, sample_from_template

_SENT = "<<<SOFT_SLOT>>>"


def build_chat_gen_template(tokenizer, messages, *, n_learnable=None, system_text=None,
                            placeholder_id=None, system_template="{SOFT}"):
    """messages: [{'role': 'user'|'assistant', ...}] WITHOUT a system turn.

    Exactly one of n_learnable (soft slot) / system_text (fixed text) is used;
    both None means "no system message at all" (the model's default system
    prompt, injected by the chat template).

    system_template wraps the soft slot exactly as it was wrapped at train time
    (e.g. "The assistant is {SOFT}"). A prompt fitted inside a frame and then
    evaluated bare is a different prompt, so this has to match the training
    config -- callers read it off the checkpoint rather than passing it by hand.
    """
    assert system_template.count("{SOFT}") == 1, system_template
    if n_learnable is None and system_text is None:
        rendered = apply_chat_template_soft(tokenizer, list(messages), tokenize=False,
                                            add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        return Template(prefix_ids=ids, slot_ids=[], suffix_ids=[])

    content = (system_template.replace("{SOFT}", _SENT)
               if n_learnable is not None else system_text)
    msgs = [{"role": "system", "content": content}] + list(messages)
    rendered = apply_chat_template_soft(tokenizer, msgs, tokenize=False,
                                        add_generation_prompt=True)
    if n_learnable is None:
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        return Template(prefix_ids=ids, slot_ids=[], suffix_ids=[])

    assert rendered.count(_SENT) == 1, f"slot sentinel appears {rendered.count(_SENT)}x"
    before, after = rendered.split(_SENT, 1)
    if placeholder_id is None:
        placeholder_id = tokenizer.eos_token_id or 0
    return Template(
        prefix_ids=tokenizer.encode(before, add_special_tokens=False),
        slot_ids=[placeholder_id] * n_learnable,
        suffix_ids=tokenizer.encode(after, add_special_tokens=False),
    )


@torch.no_grad()
def generate_chat(model, tokenizer, messages, *, z=None, system_text=None,
                  max_new_tokens=8, system_template="{SOFT}", **gen_kwargs):
    """One greedy/sampled completion for one chat. z (n_learnable, hidden) or None."""
    tmpl = build_chat_gen_template(
        tokenizer, messages,
        n_learnable=(z.shape[0] if z is not None else None),
        system_template=system_template,
        system_text=system_text)
    # Template always carries exactly one Slot, so the text / no-system case
    # needs a width-0 tensor rather than an empty z list (compose_embeds checks
    # slot count, not width).
    if z is None:
        w = model.get_input_embeddings().weight
        zs = [torch.zeros(0, w.shape[1], dtype=w.dtype, device=w.device)]
    else:
        zs = [z]
    out = sample_from_template(model, tmpl, zs,
                               n_samples=1, max_new_tokens=max_new_tokens,
                               pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                               **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)
