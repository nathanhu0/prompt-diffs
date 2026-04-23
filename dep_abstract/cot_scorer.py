"""Judge scoring: supports both CoT (K-sample averaging) and logit-based modes.

CoT mode: generate reasoning + final score, parse and average over K samples.
Logit mode: request logprobs on first token, compute expected score directly.
Works with any OpenAI-compatible client (OpenAI API or vLLM).
"""
import math
import re
from dataclasses import dataclass, field

# ---- Judge prompt definitions ----
# Each judge has a shared preamble (role + scale) with two suffix variants:
#   - CoT: "discuss strengths/weaknesses, then score on the last line"
#   - Logit: "respond with only a single number"

_HARSH_NODIM_PREAMBLE = (
    "You are a critical, demanding reviewer for the International Conference "
    "on Learning Representations (ICLR). You have high standards and evaluate "
    "rigorously. Most submissions do not merit acceptance.\n\n"
    "You will be given a paper title and abstract. Evaluate the submission "
    "on this scale:\n\n"
    "10: Strong accept, should be highlighted at the conference as spotlight or oral\n"
    "8: Accept, good paper (poster)\n"
    "6: Marginally above the acceptance threshold\n"
    "4: Marginally below the acceptance threshold\n"
    "2: Reject, not good enough\n"
    "0: Strong reject\n\n"
    "The average human reviewer score is 4.21 with a typical within-paper standard "
    "deviation of 1.15. Most papers are rejected. Your scores should reflect this "
    "distribution."
)

_ACCEPT_REJECT_PREAMBLE = (
    "You are a reviewer for the International Conference on Learning "
    "Representations (ICLR). You have high standards. A paper must be "
    "exceptional to merit acceptance. Over half of submissions are rejected "
    "and most of the rest are borderline. Only truly strong papers deserve "
    "acceptance.\n\n"
    "You will be given a paper title and abstract. Based on the abstract "
    "alone, decide whether to recommend this paper for acceptance."
)

_NOVELTY_PREAMBLE = (
    "You are evaluating the novelty and originality of a paper based on its "
    "abstract. You have seen thousands of papers and have a high bar for "
    "novelty. Most papers are incremental.\n\n"
    "Rate how novel the contribution appears on a scale of 1-5:\n\n"
    "5: Highly novel, opens a genuinely new direction — very rare\n"
    "4: Novel, meaningful new contribution\n"
    "3: Incremental but valid — this is the typical paper\n"
    "2: Limited novelty, mostly follows existing work\n"
    "1: No apparent novelty\n\n"
    "The average paper scores about 3."
)

_SOUNDNESS_PREAMBLE = (
    "You are evaluating the apparent technical soundness of a paper based on "
    "its abstract. You are skeptical by default — claims in abstracts are "
    "often overstated.\n\n"
    "Rate how sound the methodology and claims appear on a scale of 1-5:\n\n"
    "5: Very sound, rigorous methodology — rare based on abstract alone\n"
    "4: Appears sound, minor concerns\n"
    "3: Mostly sound but some questionable claims — typical\n"
    "2: Significant methodological concerns\n"
    "1: Unsound, major flaws apparent\n\n"
    "The average paper scores about 3."
)

# Suffix templates
_COT_SUFFIX_NUMERIC = (
    "\n\nFirst, briefly discuss the strengths and weaknesses of the work based on "
    "the abstract. Then, on the final line, provide your score as a single "
    "number and nothing else."
)
_LOGIT_SUFFIX_NUMERIC = (
    "\nRespond with only a single number. "
    "Do not include any explanation."
)

_COT_SUFFIX_BINARY = (
    "\n\nFirst, briefly discuss the strengths and weaknesses. Then, on the "
    "final line, write exactly 1 (accept) or 0 (reject) and nothing else."
)
_LOGIT_SUFFIX_BINARY = (
    "\nRespond with only 1 (accept) or 0 (reject). "
    "Do not include any explanation."
)


@dataclass
class JudgeDef:
    preamble: str
    user_template: str
    valid_scores: set
    cot_suffix: str
    logit_suffix: str
    # token strings to check logprobs for (must match valid_scores values)
    response_tokens: list[str] = field(default_factory=list)

    def system_prompt(self, cot: bool) -> str:
        return self.preamble + (self.cot_suffix if cot else self.logit_suffix)


JUDGES: dict[str, JudgeDef] = {}

JUDGES["harsh_nodim"] = JudgeDef(
    preamble=_HARSH_NODIM_PREAMBLE,
    user_template="Title: {title}\n\nAbstract: {abstract}\n\nOverall recommendation (0-10):",
    valid_scores={0, 2, 4, 6, 8, 10},
    cot_suffix=_COT_SUFFIX_NUMERIC,
    logit_suffix=_LOGIT_SUFFIX_NUMERIC,
    response_tokens=["0", "2", "4", "6", "8"],  # "10" handled specially
)

JUDGES["accept_reject"] = JudgeDef(
    preamble=_ACCEPT_REJECT_PREAMBLE,
    user_template="Title: {title}\n\nAbstract: {abstract}\n\nAccept (1) or Reject (0)?",
    valid_scores={0, 1},
    cot_suffix=_COT_SUFFIX_BINARY,
    logit_suffix=_LOGIT_SUFFIX_BINARY,
    response_tokens=["0", "1"],
)

JUDGES["novelty"] = JudgeDef(
    preamble=_NOVELTY_PREAMBLE,
    user_template="Title: {title}\n\nAbstract: {abstract}\n\nNovelty rating (1-5):",
    valid_scores={1, 2, 3, 4, 5},
    cot_suffix=_COT_SUFFIX_NUMERIC,
    logit_suffix=_LOGIT_SUFFIX_NUMERIC,
    response_tokens=["1", "2", "3", "4", "5"],
)

JUDGES["soundness"] = JudgeDef(
    preamble=_SOUNDNESS_PREAMBLE,
    user_template="Title: {title}\n\nAbstract: {abstract}\n\nSoundness rating (1-5):",
    valid_scores={1, 2, 3, 4, 5},
    cot_suffix=_COT_SUFFIX_NUMERIC,
    logit_suffix=_LOGIT_SUFFIX_NUMERIC,
    response_tokens=["1", "2", "3", "4", "5"],
)


# ---- Score parsing (CoT mode) ----

def parse_score(text, valid_scores):
    """Extract numeric score from the last line of CoT output."""
    lines = text.strip().split("\n")
    # Walk backwards to find a line with just a number
    for line in reversed(lines):
        line = line.strip()
        match = re.match(r"^(\d+)\.?$", line)
        if match:
            val = int(match.group(1))
            if val in valid_scores:
                return float(val)
    # Fallback: find any valid number in the last few lines
    for line in reversed(lines[-3:]):
        numbers = re.findall(r"\b(\d+)\b", line)
        for n in reversed(numbers):
            val = int(n)
            if val in valid_scores:
                return float(val)
    return None


# ---- Logprob scoring (logit mode) ----

def _score_from_logprobs(logprobs_content, judge_def):
    """Compute expected score from first-token logprobs.

    logprobs_content: response.choices[0].logprobs.content — list of token
        logprob objects. We look at the first token's top_logprobs.
    """
    if not logprobs_content:
        return float("nan")

    first_token = logprobs_content[0]
    # Build {token_str: prob} from top_logprobs
    token_probs = {}
    for lp in first_token.top_logprobs:
        token_probs[lp.token] = math.exp(lp.logprob)

    # Map response tokens to score values and accumulate
    score_probs = {}
    for tok_str in judge_def.response_tokens:
        val = int(tok_str)
        score_probs[val] = token_probs.get(tok_str, 0.0)

    # "10" is two tokens ("1", "0"). We only see first-token logprobs, but
    # since valid scores are even-only (0,2,4,6,8,10), a "1" token
    # unambiguously means "10". So p(10) = p("1").
    if 10 in judge_def.valid_scores and 10 not in score_probs:
        score_probs[10] = token_probs.get("1", 0.0)

    coverage = sum(score_probs.values())
    if coverage < 0.01:
        return float("nan")
    return sum(v * p for v, p in score_probs.items()) / coverage


# ---- Unified scoring functions ----

def _mean_std(scores):
    if not scores:
        return float("nan"), float("nan")
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        var = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        return mean, var ** 0.5
    return mean, 0.0


@dataclass
class ScoreResult:
    # Select scores — used by optimizer for decisions
    select_mean: float
    select_std: float
    select_scores: list[float]
    # Eval scores — held out, never seen by optimizer
    eval_mean: float
    eval_std: float
    eval_scores: list[float]
    # All raw CoT texts (select + eval, in order). Empty for logit mode.
    raw_texts: list[str]
    n_failed: int
    mode: str  # "cot" or "logit"
    k_select: int
    k_eval: int


def score(client, model, title, abstract, judge="harsh_nodim",
          cot=True, k_select=5, k_eval=5, temperature=1.0,
          top_p=None, max_tokens=512):
    """Score one abstract. Uses CoT or logit mode based on `cot` flag.

    CoT mode: generate k_select + k_eval reasoning traces. First k_select
    are for selection (optimizer sees these), remaining k_eval are held out.
    Logit mode: single call with logprobs. k_select/k_eval are ignored.
    """
    judge_def = JUDGES[judge]
    user_content = judge_def.user_template.format(title=title, abstract=abstract)
    messages = [
        {"role": "system", "content": judge_def.system_prompt(cot)},
        {"role": "user", "content": user_content},
    ]

    if cot:
        return _score_cot(client, model, messages, judge_def,
                          k_select, k_eval, temperature, max_tokens, top_p)
    else:
        return _score_logit(client, model, messages, judge_def)


def _score_cot(client, model, messages, judge_def,
               k_select, k_eval, temperature, max_tokens, top_p=None):
    """CoT scoring: sample k_select + k_eval, split into select/eval sets."""
    k_total = k_select + k_eval
    is_openai = client.base_url.host == "api.openai.com"
    max_n = 8 if is_openai else 64  # OpenAI caps n at 8

    base_kwargs = dict(model=model, messages=messages,
                       temperature=temperature, max_tokens=max_tokens)
    if top_p is not None:
        base_kwargs["top_p"] = top_p

    # Batch requests if k_total > max_n
    all_choices = []
    for batch_start in range(0, k_total, max_n):
        batch_n = min(max_n, k_total - batch_start)
        resp = client.chat.completions.create(n=batch_n, **base_kwargs)
        all_choices.extend(resp.choices)

    all_scores = []
    raw_texts = []
    n_failed = 0
    for choice in all_choices:
        text = choice.message.content
        raw_texts.append(text)
        s = parse_score(text, judge_def.valid_scores)
        if s is not None:
            all_scores.append(s)
        else:
            all_scores.append(None)
            n_failed += 1

    # Split into select and eval sets
    select_scores = [s for s in all_scores[:k_select] if s is not None]
    eval_scores = [s for s in all_scores[k_select:] if s is not None]

    select_mean, select_std = _mean_std(select_scores)
    eval_mean, eval_std = _mean_std(eval_scores)

    return ScoreResult(
        select_mean=select_mean, select_std=select_std,
        select_scores=select_scores,
        eval_mean=eval_mean, eval_std=eval_std,
        eval_scores=eval_scores,
        raw_texts=raw_texts, n_failed=n_failed, mode="cot",
        k_select=k_select, k_eval=k_eval,
    )


def _score_logit(client, model, messages, judge_def):
    """Logit scoring: single call with logprobs, compute expected score."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,
    )

    logprobs_content = resp.choices[0].logprobs.content
    expected = _score_from_logprobs(logprobs_content, judge_def)

    return ScoreResult(
        select_mean=expected, select_std=0.0, select_scores=[expected],
        eval_mean=expected, eval_std=0.0, eval_scores=[expected],
        raw_texts=[], n_failed=0, mode="logit",
        k_select=1, k_eval=0,
    )
