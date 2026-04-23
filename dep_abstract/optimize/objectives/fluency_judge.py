"""Fluency-judge objective.

Scores a candidate embedding z by the probability the judge's first response
token is an affirmative ("Yes", "yes", "YES"). Used as an auxiliary term
alongside NLLDistill to keep the learnable suffix plausible as text that
could blend with the abstract.

Single prompt, single forward — no train/val/test splits, no mini-batches.
Interface is intentionally simpler than NLLDistillObjective: `loss(z)`
returns a differentiable scalar. The caller (e.g. LargoOptimizer) handles
`.backward()`.

Only the FIRST response-token position is scored. The loss is
`-log(sum_{t in target_tokens} P(t | prompt))` so all variants of "yes"
(capitalization, etc.) count as success.
"""
import torch
import torch.nn.functional as F

from optimize.slot_factories.abstract import (
    tokenize_with_user_slot as _tokenize_rollout,
)


DEFAULT_PROMPT_TEMPLATE = (
    "Below is a research paper abstract:\n\n"
    "{abstract}\n\n"
    "Consider adding this next sentence:\n\n"
    "<<<SLOT>>>\n\n"
    "Does this sentence blend in naturally with the abstract's style and "
    "content? Respond with a simple yes or no."
)
DEFAULT_TARGET_TOKENS = ["Yes", "yes", "YES"]
SLOT_MARKER = "<<<SLOT>>>"


def _get_embed_matrix(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


class FluencyJudgeObjective:
    """-log P(first response token is an affirmative) under judge prompt.

    Args:
        model: frozen HF causal LM (shared with distill objective).
        tokenizer: HF tokenizer.
        abstract: paper abstract text, substituted into `{abstract}` in the
            prompt template.
        prompt_template: must contain `<<<SLOT>>>` exactly once; may contain
            `{abstract}` which will be substituted at init.
        target_tokens: list of strings whose first token ids are treated as
            affirmative responses (summed via logsumexp).
    """

    def __init__(self, model, tokenizer, abstract,
                 prompt_template=DEFAULT_PROMPT_TEMPLATE,
                 target_tokens=None):
        if target_tokens is None:
            target_tokens = DEFAULT_TARGET_TOKENS
        self.model = model
        self.tokenizer = tokenizer
        self.embed_matrix = _get_embed_matrix(model)
        self.device = self.embed_matrix.device
        self.target_tokens = list(target_tokens)

        # Precompute the first-token id for each variant. We use the first
        # encoded id (e.g. "Yes" -> [9642]); the chat template handles any
        # leading-whitespace normalization at the role boundary.
        target_ids = [
            tokenizer.encode(t, add_special_tokens=False)[0]
            for t in self.target_tokens
        ]
        self._target_token_ids = torch.tensor(target_ids, device=self.device)

        # Tokenize the judge prompt once. The rollout_text used here only
        # controls where target_start lands — we later score only the single
        # position at target_start, ignoring whatever tokens followed it.
        user_content = prompt_template.format(abstract=abstract)
        prefix_ids, slot_ids, suffix_ids, target_start = _tokenize_rollout(
            tokenizer, user_content, SLOT_MARKER, self.target_tokens[0],
        )
        self._prefix = torch.tensor(prefix_ids, device=self.device)
        self._suffix = torch.tensor(suffix_ids, device=self.device)
        self._prefix_len = len(prefix_ids)
        self._marker_n = len(slot_ids)
        self._target_start = target_start

    def loss(self, z):
        """Differentiable scalar: -log P(first response ∈ target_tokens).

        Args:
            z: (n, dim) tensor — the candidate embeddings to judge. In
                suffix mode, pass only the learnable tail (not the frozen
                abstract prefix); the prompt template already contains the
                abstract as text.
        """
        prefix_emb = self.embed_matrix[self._prefix]
        suffix_emb = self.embed_matrix[self._suffix]
        embeds = torch.cat([prefix_emb, z, suffix_emb], dim=0).unsqueeze(0)
        logits = self.model(inputs_embeds=embeds).logits[0]

        shift = z.shape[0] - self._marker_n
        adj_target = self._target_start + shift
        # Single position: the first assistant-response token slot.
        pred_logits = logits[adj_target - 1].float()
        log_probs = F.log_softmax(pred_logits, dim=-1)
        yes_log_prob = torch.logsumexp(log_probs[self._target_token_ids], dim=0)
        return -yes_log_prob
