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


def _tokenize_rollout(tokenizer, user_content, slot_text, rollout_text):
    """Tokenize a rollout and find the slot boundaries.

    Args:
        user_content: the full user message content (what gets rendered)
        slot_text: substring of user_content that defines the slot
            (must appear exactly once)

    Returns:
        prefix_ids: token ids before the slot
        slot_ids: token ids for the slot (original tokens)
        suffix_ids: token ids after the slot
        target_start: index where target tokens start in the full sequence
    """
    messages = _build_messages(user_content, rollout_text)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(full_text, return_offsets_mapping=True,
                         add_special_tokens=False)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping

    # slot_text must be unique within the user message (not the full template,
    # since the rollout may quote it). Find it there, then map to full_text.
    assert user_content.count(slot_text) == 1, \
        f"slot_text must appear exactly once in user_content, " \
        f"found {user_content.count(slot_text)}"
    user_char_offset = full_text.index(user_content)
    slot_char_start = user_char_offset + user_content.index(slot_text)
    slot_char_end = slot_char_start + len(slot_text)

    # Map char span to token boundaries
    slot_start = None
    slot_end = None
    for idx, (cs, ce) in enumerate(offsets):
        if cs >= slot_char_start and ce <= slot_char_end and cs < ce:
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

    def __init__(self, model, tokenizer, title, abstract, rollouts_by_split,
                 slot_text=None):
        """
        Args:
            model: frozen HF causal LM
            tokenizer: HF tokenizer
            title: paper title
            abstract: paper abstract (rendered into the user content)
            rollouts_by_split: dict with keys "train", "val", "test",
                each a list of dicts with "query_text" and "rollout_text"
            slot_text: substring defining the slot location. Defaults to the
                full abstract. Set to a sub-substring (e.g. the last sentence)
                to optimize only part of the abstract.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embed_matrix = self._get_embed_matrix(model)
        self.device = self.embed_matrix.device
        self.abstract = abstract
        self.slot_text = slot_text if slot_text is not None else abstract

        # User content template — abstract is what gets rendered
        user_template = f"Title: {title}\n\nAbstract: {abstract}"

        # Pre-tokenize all rollouts into (prefix, slot, suffix, target_start)
        self._data = {}
        self._slot_ids = None
        for split_name, rollouts in rollouts_by_split.items():
            split_data = []
            for r in rollouts:
                user_content = f"{user_template}\n\n{r['query_text']}"
                prefix, slot, suffix, target_start = _tokenize_rollout(
                    tokenizer, user_content, self.slot_text, r["rollout_text"]
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

    def _score_batch(self, z, rollouts):
        """Score a batch of rollouts in one forward pass. Returns (B,) losses."""
        B = len(rollouts)
        seqs = []
        labels_list = []

        for prefix_ids, slot_ids, suffix_ids, target_start in rollouts:
            prefix = torch.tensor(prefix_ids, device=self.device)
            suffix = torch.tensor(suffix_ids, device=self.device)

            prefix_embeds = self.embed_matrix[prefix]
            suffix_embeds = self.embed_matrix[suffix]
            embeds = torch.cat([prefix_embeds, z, suffix_embeds], dim=0)
            seqs.append(embeds)

            # Build label ids: -100 everywhere except target positions
            seq_len = embeds.shape[0]
            label = torch.full((seq_len,), -100, device=self.device,
                               dtype=torch.long)
            shift = z.shape[0] - self.n_slot
            adjusted_target = target_start + shift
            suffix_offset = target_start - len(prefix_ids) - self.n_slot
            target_ids = suffix[suffix_offset:]
            label[adjusted_target:adjusted_target + len(target_ids)] = target_ids
            labels_list.append(label)

        # Pad to max length
        max_len = max(s.shape[0] for s in seqs)
        dim = z.shape[1]
        padded = torch.zeros(B, max_len, dim, device=self.device,
                             dtype=z.dtype)
        attn_mask = torch.zeros(B, max_len, device=self.device,
                                dtype=torch.long)
        labels = torch.full((B, max_len), -100, device=self.device,
                            dtype=torch.long)
        for i, (seq, lab) in enumerate(zip(seqs, labels_list)):
            L = seq.shape[0]
            padded[i, :L] = seq
            attn_mask[i, :L] = 1
            labels[i, :L] = lab

        # One forward pass
        logits = self.model(inputs_embeds=padded,
                            attention_mask=attn_mask).logits

        # Shift: logits[i-1] predicts token[i]
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Per-token loss, then average per row over target tokens only
        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
        ).view(B, -1)

        target_mask = (shift_labels != -100).float()
        per_row = (per_token * target_mask).sum(dim=1) / target_mask.sum(dim=1)
        return per_row

    def loss(self, z_or_fn, split="train", backward=False,
             mini_batch_size=None):
        """NLL loss averaged over rollouts in the split.

        Args:
            z_or_fn: either a (n_slot, dim) tensor, or a callable that returns
                one. Callable is needed for backward=True so the computation
                graph is fresh per mini-batch.
            backward: if True, call backward per mini-batch (gradient
                accumulation) and return a detached float.
                If False, return a differentiable scalar.
            mini_batch_size: if set, process rollouts in chunks of this size.
                If None, process all rollouts in one batch.
        """
        data = self._data[split]
        n = len(data)
        bs = mini_batch_size or n

        if backward:
            total = 0.0
            for i in range(0, n, bs):
                chunk = data[i:i + bs]
                z = z_or_fn() if callable(z_or_fn) else z_or_fn
                losses = self._score_batch(z, chunk)
                (losses.sum() / n).backward()
                total += losses.sum().item()
            return total / n
        else:
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            all_losses = []
            for i in range(0, n, bs):
                chunk = data[i:i + bs]
                all_losses.append(self._score_batch(z, chunk))
            return torch.cat(all_losses).mean()
