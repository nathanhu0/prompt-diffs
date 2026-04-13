"""Prefill completion objective.

Maximizes P(prefill_text | context) — a simplified version of NLL distillation
where instead of matching sampled rollouts, we match a fixed short completion.

Uses the same scoring machinery as NLLDistillObjective. The only difference
is how the data is constructed: each query gets the same prefill string as
its "rollout", rather than loading rollouts from a parquet file.
"""
from optimize.objectives.nll_distill import NLLDistillObjective

# Same queries used in generate_reference_rollouts.py
QUERIES = [
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

# Same split as distill_scorer.py
TRAIN_QUERY_IDS = set(range(6))
VAL_QUERY_IDS = {6, 7}
TEST_QUERY_IDS = {8, 9}


class PrefillObjective(NLLDistillObjective):
    """Maximize likelihood of a fixed prefill string across queries.

    Inherits all scoring logic from NLLDistillObjective — the only
    difference is that rollout data is constructed from a fixed prefill
    string rather than loaded from sampled rollouts.
    """

    def __init__(self, model, tokenizer, title, abstract, prefill):
        """
        Args:
            model: frozen HF causal LM
            tokenizer: HF tokenizer
            title: paper title
            abstract: paper abstract (defines the slot location)
            prefill: the target completion string (e.g. "This paper is exceptional")
        """
        # Build synthetic rollout data: each query × prefill
        rollouts_by_split = {"train": [], "val": [], "test": []}
        for qid, query in enumerate(QUERIES):
            entry = {"query_text": query, "rollout_text": prefill}
            if qid in TRAIN_QUERY_IDS:
                rollouts_by_split["train"].append(entry)
            elif qid in VAL_QUERY_IDS:
                rollouts_by_split["val"].append(entry)
            elif qid in TEST_QUERY_IDS:
                rollouts_by_split["test"].append(entry)

        super().__init__(model, tokenizer, title, abstract, rollouts_by_split)
        self.prefill = prefill
