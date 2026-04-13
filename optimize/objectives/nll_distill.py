"""NLL distillation objective.

Scores a candidate by how well the model reproduces reference rollouts
(generated with an injection) when given only the candidate text (no injection).
Lower NLL = the candidate induces similar behavior to the injection.
"""
import torch
import torch.nn.functional as F


def _build_messages(user_content, rollout_text):
    """Build chat messages for tokenization."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": rollout_text},
    ]


def _tokenize_rollout(tokenizer, user_content, abstract, rollout_text):
    """Tokenize a rollout and find the abstract slot boundaries.

    Returns:
        prefix_ids: token ids before the abstract slot
        slot_ids: token ids for the abstract slot (original tokens)
        suffix_ids: token ids after the abstract slot
        target_start: index where target tokens start in the full sequence
    """
    messages = _build_messages(user_content, rollout_text)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(full_text, return_offsets_mapping=True,
                         add_special_tokens=False)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping

    # Find abstract char span
    abs_start = full_text.index(abstract)
    abs_end = abs_start + len(abstract)
    assert full_text.count(abstract) == 1, \
        f"Abstract must appear exactly once in template, found {full_text.count(abstract)}"

    # Map char span to token boundaries
    slot_start = None
    slot_end = None
    for idx, (cs, ce) in enumerate(offsets):
        if cs >= abs_start and ce <= abs_end and cs < ce:
            if slot_start is None:
                slot_start = idx
            slot_end = idx + 1

    # Find target boundary
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    target_start = len(prompt_ids)

    prefix_ids = input_ids[:slot_start]
    slot_ids = input_ids[slot_start:slot_end]
    suffix_ids = input_ids[slot_end:]

    return prefix_ids, slot_ids, suffix_ids, target_start


class NLLDistillObjective:
    """NLL of reference rollouts as the optimization objective.

    The objective owns:
    - The model and tokenizer
    - The reference rollout data (pre-tokenized into prefix/slot/suffix splits)
    - The formatting (chat template structure)
    - Train/val/test splits

    The objective does NOT know what's frozen vs learnable — it receives
    full slot embeddings and scores them.
    """

    def __init__(self, model, tokenizer, title, abstract, rollouts_by_split):
        """
        Args:
            model: frozen HF causal LM
            tokenizer: HF tokenizer
            title: paper title
            abstract: paper abstract (defines the slot location)
            rollouts_by_split: dict with keys "train", "val", "test",
                each a list of dicts with "query_text" and "rollout_text"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embed_matrix = self._get_embed_matrix(model)
        self.device = self.embed_matrix.device
        self.abstract = abstract

        # User content template — abstract defines the slot
        user_template = f"Title: {title}\n\nAbstract: {abstract}"

        # Pre-tokenize all rollouts into (prefix, slot, suffix, target_start)
        self._data = {}
        self._slot_ids = None
        for split_name, rollouts in rollouts_by_split.items():
            split_data = []
            for r in rollouts:
                user_content = f"{user_template}\n\n{r['query_text']}"
                prefix, slot, suffix, target_start = _tokenize_rollout(
                    tokenizer, user_content, abstract, r["rollout_text"]
                )
                split_data.append((prefix, slot, suffix, target_start))
                if self._slot_ids is None:
                    self._slot_ids = slot
            self._data[split_name] = split_data

        self.n_slot = len(self._slot_ids)
        self.original_slot_ids = torch.tensor(self._slot_ids, device=self.device)

    @staticmethod
    def _get_embed_matrix(model):
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens.weight
        return model.get_input_embeddings().weight

    def _score_single(self, z, prefix_ids, suffix_ids, target_start):
        """Score one rollout given slot embeddings z."""
        prefix = torch.tensor(prefix_ids, device=self.device)
        suffix = torch.tensor(suffix_ids, device=self.device)

        prefix_embeds = self.embed_matrix[prefix]
        suffix_embeds = self.embed_matrix[suffix]
        embeds = torch.cat([prefix_embeds, z, suffix_embeds], dim=0)

        logits = self.model(inputs_embeds=embeds.unsqueeze(0)).logits[0]

        # Target labels: the rollout tokens live in suffix_ids.
        # target_start is the index in the ORIGINAL sequence (prefix + slot + suffix).
        # In the new sequence (prefix + z + suffix), targets shift by len(z) - n_slot.
        shift = z.shape[0] - self.n_slot
        adjusted_target = target_start + shift

        # Target labels are the original token ids from target_start onward.
        # These are all in suffix_ids: offset into suffix = target_start - len(prefix) - n_slot
        suffix_offset = target_start - len(prefix_ids) - self.n_slot
        target_ids = suffix[suffix_offset:]

        # Predictions: logits[i-1] predicts token[i]
        pred_logits = logits[adjusted_target - 1: adjusted_target - 1 + len(target_ids)]
        per_token = F.cross_entropy(pred_logits, target_ids, reduction="none")
        return per_token.mean()

    def loss(self, z_or_fn, split="train", backward=False):
        """NLL loss averaged over rollouts in the split.

        Args:
            z_or_fn: either a (n_slot, dim) tensor, or a callable that returns
                one. Callable is needed for backward=True so the computation
                graph is fresh per rollout.
            backward: if True, call backward per rollout (gradient accumulation)
                and return a detached float. Avoids OOM from stacking graphs.
                If False, return a differentiable scalar.
        """
        data = self._data[split]
        n = len(data)
        if backward:
            total = 0.0
            for prefix_ids, slot_ids, suffix_ids, target_start in data:
                z = z_or_fn() if callable(z_or_fn) else z_or_fn
                loss_i = self._score_single(z, prefix_ids, suffix_ids,
                                            target_start)
                (loss_i / n).backward()
                total += loss_i.item()
            return total / n
        else:
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            losses = []
            for prefix_ids, slot_ids, suffix_ids, target_start in data:
                losses.append(self._score_single(z, prefix_ids, suffix_ids,
                                                 target_start))
            return torch.stack(losses).mean()
