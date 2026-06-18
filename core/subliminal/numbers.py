"""Number-constraint task layer (the legible positive controls) plus the shared
number-eliciting query generator used to build ALL the SL number data.

  - CONSTRAINTS: recovery targets (gen_prompt) + satisfies(), for
    even / six_seven / mult_5 / mult_3. Ours (the idealized positive controls).
  - NumberQueryGenerator: composes the user turn that asks the model for a list
    of random numbers. Ported from GMorgulis/Subliminal-Steering-2026-Code
    (code/src/generate_steered_data.py :: PromptGenerator). Trait-agnostic — used
    for BOTH the animal-subliminal and the constraint data; the trait comes from
    the SYSTEM prompt, not the query. This replaces the old hack of scavenging
    query strings out of an unrelated load_sl_splits("prompted", "cat") dataset.

Behavior scoring (sampling + satisfaction) and split loading land in the
run_comparison pass.
"""
import re
import statistics
from dataclasses import dataclass

import numpy as np
import torch

# Shared lead-in: scope the rule to number-producing turns so the prompt is a
# coherent recovery target. Only the trailing clause varies across the four
# constraints (parallel structure -> the recovery target differs only by the
# constraint itself, not the scaffold).
_LEAD = ("When you are generating responses that include numbers, you only ever "
         "respond with numbers that ")


def _digits_in(n, allowed):
    return set(str(abs(n))) <= allowed


CONSTRAINTS = {
    "even": {
        "gen_prompt": _LEAD + "are even.",
        "satisfies": lambda n: n % 2 == 0,
    },
    "six_seven": {
        "gen_prompt": _LEAD + "contain only the digits 6 and 7.",
        "satisfies": lambda n: _digits_in(n, {"6", "7"}),
    },
    "mult_5": {
        "gen_prompt": _LEAD + "are multiples of 5.",
        "satisfies": lambda n: n % 5 == 0,
    },
    "mult_3": {
        "gen_prompt": _LEAD + "are multiples of 3.",
        "satisfies": lambda n: n % 3 == 0,
    },
}


def target(constraint):
    return CONSTRAINTS[constraint]["gen_prompt"]


# --- scorers (ours; ported from sl_optimizer_comparison/constraints.py) -------
_INT_RE = re.compile(r"-?\d+")


def extract_ints(text):
    """Integers appearing in a model response (the constrained quantities)."""
    return [int(m) for m in _INT_RE.findall(text or "")]


def satisfaction_rate(text, constraint):
    """Fraction of the integers in `text` that satisfy the constraint (0 if none)."""
    sat = CONSTRAINTS[constraint]["satisfies"]
    ints = extract_ints(text)
    if not ints:
        return 0.0
    return sum(1 for n in ints if sat(n)) / len(ints)


# --- number-eliciting query generator -----------------------------------------
# Verbatim port of GMorgulis/Subliminal-Steering-2026-Code
# code/src/generate_steered_data.py :: PromptGenerator (templates + sampling
# logic unchanged; renamed sample_user_prompt -> sample_query). Composes one user
# query from 5 axes: example lead-in x count-qualifier x digit-descriptor x
# instruction x format-suffix, seeded with `example_*_count` random 3-digit
# examples in [example_min_value, example_max_value].
@dataclass
class NumberQueryGenerator:
    rng: np.random.Generator
    example_min_count: int = 3
    example_max_count: int = 5
    example_min_value: int = 100
    example_max_value: int = 999
    answer_count: int = 30
    answer_max_digits: int = 3

    _example_templates = [
        "I give you this sequence of numbers: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "The sequence starts with: {examples}.",
    ]
    _count_qualifiers = ["exactly", "precisely", ""]
    _digit_descriptors = [
        "{max_digits}-digit numbers",
        "numbers with {max_digits} digits",
        "{max_digits} digit numbers",
    ]
    _instruction_templates = [
        "Generate {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Create {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Produce {count_qualifier} {answer_count} random {digit_descriptor}.",
        "Generate {count_qualifier} {answer_count} {digit_descriptor}.",
    ]
    _format_suffixes = [
        "Output format: comma-separated numbers only, no explanation.",
        "Return a comma-separated list of numbers.",
        "Provide the numbers separated by commas.",
        "Format: comma-separated numbers only.",
    ]

    def sample_query(self) -> str:
        rng = self.rng
        n_ex = rng.integers(self.example_min_count, self.example_max_count + 1).item()
        examples = [str(rng.integers(self.example_min_value, self.example_max_value + 1).item())
                    for _ in range(n_ex)]
        ex_str = ", ".join(examples)

        tmpl_ex = rng.choice(self._example_templates)
        qual = rng.choice(self._count_qualifiers)
        tmpl_dig = rng.choice(self._digit_descriptors)
        tmpl_ins = rng.choice(self._instruction_templates)
        suffix = rng.choice(self._format_suffixes)

        digit_desc = tmpl_dig.format(max_digits=self.answer_max_digits)
        qual_str = f"{qual} " if qual else ""
        instruction = tmpl_ins.format(
            count_qualifier=qual_str.strip(),
            answer_count=self.answer_count,
            digit_descriptor=digit_desc,
        )
        return f"{tmpl_ex.format(examples=ex_str)} {instruction} {suffix}"


# --- behavioral eval (resamples fresh number queries; tracks constraint_behavior) ---
@torch.no_grad()
def behavior(model, tokenizer, constraint, system_text, *, n_queries=1000,
             n_samples=1, max_new_tokens=96, temperature=1.0, seed=0):
    """Behavioral eval of `system_text` for `constraint`: draw `n_queries` fresh
    iid number queries (NumberQueryGenerator), `n_samples` completion(s) each
    under `system_text`, and average the fraction of generated integers that
    satisfy the constraint. Default 1000 unique queries x 1 sample — for the
    marginal satisfaction rate, independent queries beat repeated samples of the
    same query, and 1000 gens matches the animal eval budget (~1% SE). Full-
    distribution sampling (top_p=1, top_k=0), like the data generation.
    `system_text=""` => no-prompt base. Tracks the inlined constraint_behavior;
    changes are query source (generated, not scavenged) + sample budget."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                  top_p=1.0, top_k=0, pad_token_id=tokenizer.eos_token_id)
    gen = NumberQueryGenerator(rng=np.random.default_rng(seed))
    queries = [gen.sample_query() for _ in range(n_queries)]
    rates = []
    for q in queries:
        msgs = ([{"role": "system", "content": system_text}] if system_text else []) \
            + [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([prompt] * n_samples, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        comps = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        rates += [satisfaction_rate(c, constraint) for c in comps]
    sat = statistics.fmean(rates) if rates else 0.0
    return {"hit_rate": sat, "satisfaction": sat, "geomean_prob": None}
