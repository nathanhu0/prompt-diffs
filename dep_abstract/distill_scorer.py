"""NLL-based scorer for context distillation.

Scores candidate abstracts by how well the model reproduces reference rollouts
(generated with an injection prompt) when given only the abstract (no injection).

Lower NLL = the abstract alone induces similar behavior to the injection.
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from cot_scorer import ScoreResult
from generate_reference_rollouts import build_messages

# Train/val/test split:
#   Train: queries 0-5, rollouts 0-3 (24 rollouts)
#   Val: queries 6-7 all rollouts + rollout 4 from queries 0-5 (16 rollouts)
#   Test: queries 8-9 all rollouts (10 rollouts)
TRAIN_QUERY_IDS = set(range(6))
VAL_QUERY_IDS = {6, 7}
TEST_QUERY_IDS = {8, 9}
TRAIN_ROLLOUT_IDS = set(range(4))

DEFAULT_BATCH_SIZE = 8


def _split_rollouts(rollouts):
    """Split a paper's rollouts into train, val, and test sets."""
    train = [
        r for r in rollouts
        if r["query_id"] in TRAIN_QUERY_IDS
        and r["rollout_id"] in TRAIN_ROLLOUT_IDS
    ]
    val = [
        r for r in rollouts
        if r["query_id"] in VAL_QUERY_IDS
        or (r["query_id"] in TRAIN_QUERY_IDS
            and r["rollout_id"] not in TRAIN_ROLLOUT_IDS)
    ]
    test = [
        r for r in rollouts
        if r["query_id"] in TEST_QUERY_IDS
    ]
    return train, val, test


class DistillScorer:
    """Scorer that measures NLL of reference rollouts under the model."""

    def __init__(self, model_name, device, rollouts_df, batch_size=DEFAULT_BATCH_SIZE):
        """Load HF model and index rollouts by paper_id.

        Args:
            model_name: HuggingFace model ID
            device: torch device string (e.g. "cuda:0")
            rollouts_df: DataFrame from generate_reference_rollouts.py
            batch_size: max sequences per forward pass
        """
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()

        # Index rollouts by paper_id, pre-split into train/eval
        self._train = {}
        self._eval = {}
        for paper_id, group in rollouts_df.groupby("paper_id"):
            train, eval_ = _split_rollouts(group.to_dict("records"))
            self._train[paper_id] = train
            self._eval[paper_id] = eval_

    def _tokenize_rollout(self, title, abstract, rollout):
        """Tokenize a single rollout, returning input_ids, prefix_len."""
        messages = build_messages(title, abstract, injection="", query=rollout["query_text"])
        messages.append({"role": "assistant", "content": rollout["rollout_text"]})

        full_ids = self.tokenizer.apply_chat_template(messages, tokenize=True)

        # Get prefix length (everything before assistant response)
        prefix_messages = messages[:-1]
        prefix_ids = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=True, add_generation_prompt=True,
        )
        return full_ids, len(prefix_ids)

    def _compute_nlls_batched(self, title, abstract, rollouts):
        """Compute per-token NLL for a batch of rollouts. Returns list of floats."""
        if not rollouts:
            return []

        # Tokenize all rollouts
        all_ids = []
        prefix_lens = []
        for r in rollouts:
            ids, plen = self._tokenize_rollout(title, abstract, r)
            all_ids.append(ids)
            prefix_lens.append(plen)

        # Process in batches
        all_nlls = []
        for batch_start in range(0, len(all_ids), self.batch_size):
            batch_ids = all_ids[batch_start:batch_start + self.batch_size]
            batch_plens = prefix_lens[batch_start:batch_start + self.batch_size]
            batch_nlls = self._forward_batch(batch_ids, batch_plens)
            all_nlls.extend(batch_nlls)

        return all_nlls

    def _forward_batch(self, batch_ids, batch_plens):
        """Run a single batched forward pass, return per-sequence NLLs.

        Right-pads sequences. Only computes NLL on assistant tokens.
        Uses a response mask to vectorize the per-token NLL computation.
        """
        batch_size = len(batch_ids)
        max_len = max(len(ids) for ids in batch_ids)
        pad_id = self.tokenizer.pad_token_id
        seq_lens = [len(ids) for ids in batch_ids]

        # Right-pad
        padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids]
        attention_mask = [[1] * l + [0] * (max_len - l) for l in seq_lens]

        input_ids = torch.tensor(padded_ids, device=self.device)
        attn_mask = torch.tensor(attention_mask, device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits

        # Shifted logits/labels: logits[:, :-1] predicts input_ids[:, 1:]
        shift_logits = logits[:, :-1]  # (B, max_len-1, vocab)
        shift_labels = input_ids[:, 1:]  # (B, max_len-1)

        # Fused cross-entropy: never materializes (B, T, vocab) softmax
        token_nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(batch_size, -1)  # (B, max_len-1)

        # Build response mask: 1 for assistant tokens, 0 for prefix/padding
        # In shifted space, position t corresponds to predicting token t+1.
        # We want positions prefix_len-1 .. seq_len-2 (predicting tokens prefix_len .. seq_len-1)
        pos = torch.arange(max_len - 1, device=self.device).unsqueeze(0)  # (1, max_len-1)
        starts = torch.tensor([p - 1 for p in batch_plens], device=self.device).unsqueeze(1)
        ends = torch.tensor([s - 1 for s in seq_lens], device=self.device).unsqueeze(1)
        response_mask = (pos >= starts) & (pos < ends)  # (B, max_len-1)

        # Masked mean NLL per sequence
        masked_nll = token_nll * response_mask
        counts = response_mask.sum(dim=1).float()  # (B,)
        nlls = (masked_nll.sum(dim=1) / counts.clamp(min=1)).tolist()

        # Mark sequences with no response tokens as nan
        for i in range(batch_size):
            if counts[i] == 0:
                nlls[i] = float("nan")

        return nlls

    def score(self, paper_id, title, abstract):
        """Score a candidate abstract by NLL of reference rollouts.

        Returns ScoreResult with select_mean/eval_mean as negative NLL
        (higher = better, matching the convention that higher scores are better).
        """
        train_nlls = self._compute_nlls_batched(title, abstract, self._train[paper_id])
        eval_nlls = self._compute_nlls_batched(title, abstract, self._eval[paper_id])

        # Negate so higher = better (consistent with other scorers)
        train_scores = [-n for n in train_nlls if n == n]  # filter nan
        eval_scores = [-n for n in eval_nlls if n == n]

        def _mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        def _std(xs):
            if len(xs) < 2:
                return 0.0
            m = _mean(xs)
            return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

        return ScoreResult(
            select_mean=_mean(train_scores),
            select_std=_std(train_scores),
            select_scores=train_scores,
            eval_mean=_mean(eval_scores),
            eval_std=_std(eval_scores),
            eval_scores=eval_scores,
            raw_texts=[],
            n_failed=0,
            mode="distill_nll",
            k_select=len(train_scores),
            k_eval=len(eval_scores),
        )
