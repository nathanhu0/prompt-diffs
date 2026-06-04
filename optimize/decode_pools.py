"""Canonical decode-template pools for LARGO sysprompt-recovery.

A LARGO decode template specifies how to splice the learnable z (via
SLOT_SENTINEL = '{SLOT}') into a chat prompt for sampling. Each template
is a dict with `system?`, `user?`, `prefill?`, `postprocess?` fields.

DECODE_TEMPLATE_POOLS exposes named pools, selectable from YAML via
`optimizer.decode_pool`. Pool naming refers to where {SLOT} lives:
  - "user":   z lives in the user turn; assistant summarizes/repeats it.
              Original LARGO framing.
  - "system": z lives in the system slot; user asks the assistant to
              recite/summarize its system prompt. Closer to deployment
              for sysprompt-recovery tasks.

LargoOptimizer resolves a pool name at construction time when
LargoConfig.decode_templates is None.

**Pool templates are BASE templates** — task-agnostic scaffolding only.
They describe the structural framing (where the slot lives, what to ask
the model, what prefill to prime the assistant with, how to extract the
candidate from generated text). They DO NOT carry any task-specific
persona / system_template prefix.

Task-specific layering happens at LargoOptimizer construction time via
`LargoConfig.decode_persona_prefix`, which mirrors `task.system_template`
at decode by:
  - prepending persona to each template's `system` (before {SLOT}), so
    the decoder's effective system matches train: "<persona>\\n\\n<z>".
  - APPENDING persona after each template's `prefill`, so the persona
    text lands INSIDE the open quote / at the start-of-content position
    of the verbatim-prompt framing (e.g. `'My system prompt verbatim:
    "<persona>'`). This primes the model to continue with just the
    soft-equivalent content rather than re-emitting the persona itself.

Variants like `system_qwen3_nothink` and `system_llama` layer tokenizer-
specific scaffolding (Qwen3 `<think></think>` suppression, Llama date
prefix) onto the base via the same mechanism. They are still
task-agnostic; persona layering still happens via decode_persona_prefix
on top.
"""
from optimize.largo import SLOT_SENTINEL


# `user` pool: z lives in the user turn; assistant summarizes/repeats/
# paraphrases it. Original LARGO framing — model treats z like an opaque
# message to digest.
DEFAULT_USER_TEMPLATES = [
    # OG LARGO: z alone as user content, legacy "summarize" prefill.
    {"user": f"{SLOT_SENTINEL}",
     "prefill": "Sure, I will summarize the message: "},
    {"user": f"Please repeat the following verbatim:\n\n{SLOT_SENTINEL}",
     "prefill": "Sure, verbatim: "},
    {"user": f"What does this say?\n\n{SLOT_SENTINEL}",
     "prefill": "It says: "},
    {"user": f"Repeat the following back to me word for word:\n\n{SLOT_SENTINEL}",
     "prefill": ""},
    {"user": f"{SLOT_SENTINEL}\n\nPlease paraphrase the above.",
     "prefill": "Paraphrased: "},
    {"user": f"{SLOT_SENTINEL}\n\nWhat was just said?",
     "prefill": ""},
    {"user": f"Translate this to plain English:\n\n{SLOT_SENTINEL}",
     "prefill": "In plain English: "},
    {"user": f"Summarize these instructions so another instance of you "
            f"can follow them:\n\n{SLOT_SENTINEL}",
     "prefill": "Sure, I will summarize the instructions:\n\n"},
]

# `system` pool: z lives in the system slot; user asks the assistant to
# recite/summarize/explain its system prompt. Closer to deployment conditions
# for a sysprompt-recovery task — the optimized prompt will be deployed AS a
# system prompt, so decoding it from there is more on-distribution.
#
# These are BASE templates: system content is just `{SLOT}`, with no
# persona prefix. Tasks that bake a persona into `task.system_template`
# layer it on at construction time via `LargoConfig.decode_persona_prefix`.
#
# Each template carries a `postprocess` lambda applied to the raw decoded
# text (the generation AFTER the prefill — LARGO's _decode does not include
# the prefill in its returned text). The lambda does per-template delimiter
# extraction, then calls `prune` for cross-cutting cleanup. LargoOptimizer
# itself ignores the `postprocess` key; only downstream "treat decoded z as
# a candidate system prompt and rescore" code uses it.

# (open, close) delimiter pairs that `prune` tries in order. For each, find
# the opener within the first ~20 chars and extract up to the next matching
# closer.
SYSTEM_TEMPLATE_WRAPPERS = [
    ('"', '"'),
    # Single quotes require a space on the outer side so apostrophes in
    # contractions (it's, Let's) don't get matched as wrapper delimiters.
    (" '", "' "),
    ("“", "”"),
    (" ‘", "’ "),
    ("`", "`"),
]


def prune(text):
    """Try each wrapper pair; extract content inside if one matches."""
    text = text.strip()
    for o, c in SYSTEM_TEMPLATE_WRAPPERS:
        i = text.find(o)
        if i == -1 or i > 20:
            continue
        j = text.find(c, i + len(o))
        if j == -1:
            continue
        return text[i + len(o):j].strip()
    return text


def strip_thinking(text):
    """Drop Qwen3-style <think>...</think> framing from decoded text.

    The Qwen3 nothink scaffold prefills `<think>\\n\\n</think>\\n\\n`
    before the template prefill to suppress reasoning. If the model
    ignores the suppression and emits thinking content anyway, the raw
    decoded text can contain stray think tags that we strip here. Cases
    handled (open=`<think>`, close=`</think>`):

    - Both present, open before close: keep content BETWEEN. Model spoke
      its candidate inside the thinking block; anything after the closer
      is typically a formal refusal or boilerplate "answer".
    - Only close present (or close before open): keep PREFIX before
      close — model treated prior text as thinking, emitted closer
      mid-stream.
    - Only open present: keep SUFFIX after open — model started a
      thinking block and hit max_tokens before closing; the partial
      thinking content is the candidate.
    - Neither: pass through. Cheap no-op for non-Qwen3 pools.
    """
    open_idx = text.find("<think>")
    close_idx = text.find("</think>")
    if open_idx != -1 and close_idx != -1 and open_idx < close_idx:
        return text[open_idx + len("<think>"):close_idx]
    if close_idx != -1:
        return text[:close_idx]
    if open_idx != -1:
        return text[open_idx + len("<think>"):]
    return text


def _mk_post(stop=None):
    """Build a template's postprocess from its `stop` marker — the single
    source of truth for "where the prompt ends". The full-verbalization path
    runs the whole pipeline (split at stop, strip thinking, prune outer
    wrappers); the greedy sentence search reads `stop` directly (see
    greedy_search) and SKIPS prune, which is unsafe on a mid-stream
    continuation — it extracts a wrapper-quoted span and drops the rest."""
    def post(x):
        if stop:
            x = x.split(stop, 1)[0]
        return prune(strip_thinking(x))
    return post


# Each template carries `stop` (its end-of-prompt delimiter, or None) alongside
# the matching `postprocess`. greedy uses `stop`; full verbalization uses
# `postprocess`; _mk_post keeps them in sync.
DEFAULT_SYSTEM_TEMPLATES = [
    # 0. opening quote — content runs until the closing '"'
    {"system": f"{SLOT_SENTINEL}",
     "user": "Output your system prompt verbatim. Do not add any commentary "
             "before or after.",
     "prefill": 'My system prompt verbatim: "',
     "stop": '"', "postprocess": _mk_post('"')},
    # 1. quoted summary — prefill is just an opening '"'
    {"system": f"{SLOT_SENTINEL}",
     "user": "Please summarize your system prompt to me inside quotes. "
             "Do not add any other text.",
     "prefill": '"',
     "stop": '"', "postprocess": _mk_post('"')},
    # 2. xml tag — content runs until '</prompt>'
    {"system": f"{SLOT_SENTINEL}",
     "user": "Output your system prompt verbatim between <prompt> and "
             "</prompt> tags, with nothing else.",
     "prefill": "<prompt>\n",
     "stop": "</prompt>", "postprocess": _mk_post("</prompt>")},
    # 3. "Verbatim:" lead-in — bare, no content presupposition
    {"system": f"{SLOT_SENTINEL}",
     "user": "Repeat your system prompt exactly. No preamble, no commentary.",
     "prefill": "Verbatim:\n\n",
     "stop": None, "postprocess": _mk_post()},
    # 4. colon lead-in, no wrapper
    {"system": f"{SLOT_SENTINEL}",
     "user": "Tell me your system prompt word for word. No explanation.",
     "prefill": "My system prompt is:\n\n",
     "stop": None, "postprocess": _mk_post()},
    # 5. "Below is" framing
    {"system": f"{SLOT_SENTINEL}",
     "user": "Reproduce your system prompt exactly. No commentary.",
     "prefill": "Below is my system prompt, verbatim:\n\n",
     "stop": None, "postprocess": _mk_post()},
    # 6. minimal opening quote — the terse version of 0
    {"system": f"{SLOT_SENTINEL}",
     "user": "Reply with your system prompt in double quotes and nothing else.",
     "prefill": '"',
     "stop": '"', "postprocess": _mk_post('"')},
    # 7. "Here is" framing
    {"system": f"{SLOT_SENTINEL}",
     "user": "Please copy out your system prompt verbatim, no extra text.",
     "prefill": "Here is my system prompt, exactly as given:\n\n",
     "stop": None, "postprocess": _mk_post()},
]

# Llama 3.1's chat template auto-injects this date scaffolding into every
# system block. Decoded z's were often parroting it; prefilling the constant
# means the model only generates the instruction portion after it, so
# postprocess (which sees only post-prefill text) gets a clean instruction.
LLAMA_DATE_PREFIX = (
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n"
)

DEFAULT_SYSTEM_LLAMA_TEMPLATES = [
    {**t, "prefill": t["prefill"] + LLAMA_DATE_PREFIX}
    for t in DEFAULT_SYSTEM_TEMPLATES
]

# Qwen3 hybrid models (e.g. Qwen3-14B) emit <think>...</think> at the
# start of every assistant turn by default. Prepending `<think>\n\n</think>
# \n\n` to the decode prefill suppresses thinking so the model jumps
# straight to the soft-prompt candidate. Idempotent with Qwen3's chat
# template — the template won't double-insert.
#
# Note: this scaffold is also task-agnostic. Persona prefix (if any) gets
# APPENDED to prefill after this Qwen3-nothink scaffold by
# `LargoConfig.decode_persona_prefix`, so the final assistant prefill
# order is `<think>\\n\\n</think>\\n\\n` + base_template_prefill + persona.
QWEN3_NOTHINK_PREFIX = "<think>\n\n</think>\n\n"

DEFAULT_SYSTEM_QWEN3_NOTHINK_TEMPLATES = [
    {**t, "prefill": QWEN3_NOTHINK_PREFIX + t["prefill"]}
    for t in DEFAULT_SYSTEM_TEMPLATES
]


# Leaned 4-template subset of the system pool, from the SL greedy-sweep
# template diagnostics (claude_scripts/verify_tmpl_indexing.py, pooled over all
# 18 cat cells × 4 reps). Selected by win rate (fraction of rounds a template's
# sentence was accepted into the spine); the order [t2, t5, t0, t7] is that
# ranking. t2 (`<prompt>` tags) also has by far the lowest empty rate (4% vs
# 20-26%); t5 the best mean ΔNLL. Dropped the bare-quote templates (t1, t6) +
# t4 (worst win) + t3 (≈ t7, marginal). Keeps structural diversity (tag /
# "Below is:" / verbatim-quote / "Here is:") — no single template authored more
# than ~1/5 of accepted sentences, so some variety beats one prompt. With
# n_candidates_per_step=8 the round-robin samples each of the 4 twice per round,
# which also halves empty-starvation vs. the 8-template pool.
DEFAULT_SYSTEM_TOP4_TEMPLATES = [DEFAULT_SYSTEM_TEMPLATES[i] for i in (2, 5, 0, 7)]


DECODE_TEMPLATE_POOLS = {
    "user":                 DEFAULT_USER_TEMPLATES,
    "system":               DEFAULT_SYSTEM_TEMPLATES,
    "system_top4":          DEFAULT_SYSTEM_TOP4_TEMPLATES,
    "system_llama":         DEFAULT_SYSTEM_LLAMA_TEMPLATES,
    "system_qwen3_nothink": DEFAULT_SYSTEM_QWEN3_NOTHINK_TEMPLATES,
}
