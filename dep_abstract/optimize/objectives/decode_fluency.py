"""Decode-fluency objective.

Scores the soft embeddings z by the NLL of a reference text under LARGO's
self-reflective decode prompt. Incentivizes that the model's "summary" of z
matches the reference.

Prompt layout mirrors LargoOptimizer._decode exactly:
    [chat_before] + z + [chat_after] + [prefill] + [reference]
NLL is computed only at reference-token positions.

`loss(z)` returns a differentiable scalar; the caller handles .backward()
(same interface as FluencyJudgeObjective).
"""
import torch
import torch.nn.functional as F


DEFAULT_DECODE_PREFILL = "Sure, I will summarize the message: "


def _get_embed_matrix(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight


class DecodeFluencyObjective:
    """-log P(reference_ids | decode_prompt, z) under the LM.

    Args:
        model: frozen HF causal LM.
        tokenizer: HF tokenizer.
        reference_ids: 1-D tensor/list of token ids to score as the continuation.
        decode_prefill: string prepended after the chat boundary (must match
            LARGO's decode_prefill so train-time fluency and decode-time
            generation see the same prompt).
    """

    def __init__(self, model, tokenizer, reference_ids,
                 decode_prefill=DEFAULT_DECODE_PREFILL):
        self.model = model
        self.tokenizer = tokenizer
        self.embed_matrix = _get_embed_matrix(model)
        self.device = self.embed_matrix.device

        # Build the chat template pieces around a placeholder, same as
        # LargoOptimizer._decode.
        messages = [{"role": "user", "content": "PLACEHOLDER"}]
        template_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        before, after = template_text.split("PLACEHOLDER")

        before_ids = tokenizer.encode(before, add_special_tokens=False)
        after_ids = tokenizer.encode(after, add_special_tokens=False)
        prefill_ids = (
            tokenizer.encode(decode_prefill, add_special_tokens=False)
            if decode_prefill else []
        )

        self._before = torch.tensor(before_ids, device=self.device)
        self._after = torch.tensor(after_ids, device=self.device)
        self._prefill = torch.tensor(prefill_ids, device=self.device,
                                     dtype=torch.long)

        if not torch.is_tensor(reference_ids):
            reference_ids = torch.tensor(reference_ids, dtype=torch.long)
        self._reference = reference_ids.to(self.device).long()

        self._n_before = len(before_ids)
        self._n_after = len(after_ids)
        self._n_prefill = len(prefill_ids)
        self._n_ref = len(self._reference)

    def loss(self, z):
        """Differentiable NLL averaged over reference tokens.

        Args:
            z: (n_learnable, dim) tensor of soft embeddings to "summarize".
        """
        before_emb = self.embed_matrix[self._before]
        after_emb = self.embed_matrix[self._after]
        prefill_emb = self.embed_matrix[self._prefill]
        ref_emb = self.embed_matrix[self._reference]

        embeds = torch.cat([before_emb, z, after_emb, prefill_emb, ref_emb],
                           dim=0).unsqueeze(0)
        logits = self.model(inputs_embeds=embeds).logits[0]

        # Reference tokens start at: n_before + n_z + n_after + n_prefill.
        n_z = z.shape[0]
        ref_start = self._n_before + n_z + self._n_after + self._n_prefill

        # logits[i-1] predicts token at position i.
        pred_logits = logits[ref_start - 1: ref_start - 1 + self._n_ref]
        per_token = F.cross_entropy(pred_logits.float(), self._reference,
                                    reduction="none")
        return per_token.mean()
