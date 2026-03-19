"""ICLR reviewer scoring module.

Provides a single `score()` function that takes a title and abstract
(as text, token ids, or embeddings) and returns review score info.
Uses the harsh_nodim prompt with Llama 3.1 8B.
"""
import torch
import torch.nn.functional as F
from optimize import tokens_to_embeddings, tokenize_to_embeds

# ---- Prompt config (locked in: harsh_nodim) ----
SYSTEM_PROMPT = (
    "You are a critical, demanding reviewer for the International Conference "
    "on Learning Representations (ICLR). You have high standards and evaluate "
    "rigorously. Most submissions do not merit acceptance.\n\n"
    "You will be given a paper title and abstract. Provide an overall "
    "recommendation on this scale:\n\n"
    "10: Strong accept, should be highlighted at the conference as spotlight or oral\n"
    "8: Accept, good paper (poster)\n"
    "6: Marginally above the acceptance threshold\n"
    "4: Marginally below the acceptance threshold\n"
    "2: Reject, not good enough\n"
    "0: Strong reject\n\n"
    "The average human reviewer score is 4.21 with a typical within-paper standard deviation of 1.15. Most papers are rejected. Your scores should reflect this distribution.\n"
    "Respond with only a single even number (0, 2, 4, 6, 8, or 10). "
    "Do not include any explanation."
)

SCORE_VALUES = [0, 2, 4, 6, 8, 10]


def _build_template_ids(tokenizer, title):
    """Build token IDs for everything before and after the abstract.

    Returns (prefix_ids, suffix_ids) as 1D tensors on CPU.
    The full prompt is: prefix_ids + <abstract tokens> + suffix_ids
    """
    # We need the chat template with a placeholder for the abstract.
    # Build it by applying the template, then splitting around a sentinel.
    sentinel = "<<<ABSTRACT_PLACEHOLDER>>>"
    user_content = f"Title: {title}\n\nAbstract: {sentinel}\n\nOverall recommendation (0-10):"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    before, after = full_text.split(sentinel)
    prefix_ids = tokenizer.encode(before, add_special_tokens=False)
    suffix_ids = tokenizer.encode(after, add_special_tokens=False)
    return (
        torch.tensor(prefix_ids, dtype=torch.long),
        torch.tensor(suffix_ids, dtype=torch.long),
    )


def _get_score_token_ids(tokenizer):
    """Get token IDs for score values 0,2,4,6,8 and the two-token '10'."""
    single = {v: tokenizer.encode(str(v), add_special_tokens=False)[0] for v in [0, 2, 4, 6, 8]}
    token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
    token_0 = tokenizer.encode("0", add_special_tokens=False)[0]
    return single, token_1, token_0


def score(model, tokenizer, title, abstract):
    """Score an abstract using the ICLR reviewer prompt.

    Args:
        model: frozen causal LM
        tokenizer: matching tokenizer
        title: paper title (str)
        abstract: one of:
            - str: raw text, will be tokenized + embedded
            - list[int] or 1D LongTensor: token ids, will be embedded
            - 2D/3D FloatTensor: embeddings (seq_len, dim) or (1, seq_len, dim)

    Returns:
        dict with: expected_score, probs (dict), coverage, loss (CE on "8")
        If abstract is embeddings, gradients flow through.
    """
    embed_matrix = model.get_input_embeddings().weight
    device = embed_matrix.device

    prefix_ids, suffix_ids = _build_template_ids(tokenizer, title)
    prefix_ids, suffix_ids = prefix_ids.to(device), suffix_ids.to(device)
    prefix_embeds = tokens_to_embeddings(prefix_ids, embed_matrix)
    suffix_embeds = tokens_to_embeddings(suffix_ids, embed_matrix)

    # Convert abstract to embeddings
    if isinstance(abstract, str):
        abstract = abstract.strip()
        _, abstract_embeds = tokenize_to_embeds(abstract, tokenizer, embed_matrix)
    elif isinstance(abstract, list) or (isinstance(abstract, torch.Tensor) and abstract.dtype == torch.long):
        ids = torch.tensor(abstract, device=device) if isinstance(abstract, list) else abstract.to(device)
        abstract_embeds = tokens_to_embeddings(ids, embed_matrix)
    elif isinstance(abstract, torch.Tensor):
        abstract_embeds = abstract.to(device)
        if abstract_embeds.dim() == 2:
            abstract_embeds = abstract_embeds.unsqueeze(0)
    else:
        raise TypeError(f"abstract must be str, list[int], or Tensor, got {type(abstract)}")

    # Forward pass
    full_embeds = torch.cat([prefix_embeds, abstract_embeds, suffix_embeds], dim=1)
    logits = model(inputs_embeds=full_embeds).logits[:, -1, :]  # last position
    probs = torch.softmax(logits, dim=-1)

    # Extract score probabilities
    single_ids, token_1, token_0 = _get_score_token_ids(tokenizer)
    score_probs = {}
    for v, tid in single_ids.items():
        score_probs[v] = probs[0, tid]

    # p(10) = p("1") * p("0" | "1")
    p_1 = probs[0, token_1]
    if p_1 > 0.001:
        token_1_embeds = tokens_to_embeddings(torch.tensor([token_1], device=device), embed_matrix)
        full_plus_1 = torch.cat([full_embeds, token_1_embeds], dim=1)
        logits_2 = model(inputs_embeds=full_plus_1).logits[:, -1, :]
        probs_2 = torch.softmax(logits_2, dim=-1)
        score_probs[10] = p_1 * probs_2[0, token_0]
    else:
        score_probs[10] = torch.zeros(1, device=device, dtype=probs.dtype).squeeze()

    coverage = sum(score_probs[v] for v in SCORE_VALUES)
    expected_score = sum(v * score_probs[v] for v in SCORE_VALUES) / coverage

    return {
        "expected_score": expected_score,
        "probs": {v: score_probs[v] for v in SCORE_VALUES},
        "coverage": coverage,
    }
