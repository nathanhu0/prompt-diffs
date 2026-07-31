"""DPO (Direct Preference Optimization) objective over preference triples.

Recovers a system prompt from preference data: train the soft prompt z so that
base+z prefers `chosen` over `rejected` the way a trait system prompt would.
For one triple, with policy = base+z and reference = base+no-system:

    margin = (logP_pol(chosen) - logP_ref(chosen))
           - (logP_pol(rejected) - logP_ref(rejected))
    loss   = -log sigmoid(beta * margin)          # mean over triples

This `margin` is exactly the LLS selection weight (sum form): the reference
(no-system) logprobs are constant in z so they are precomputed ONCE at
construction. They are not droppable — they shift where the margin sits on the
sigmoid and therefore change the gradient w.r.t. z.

Reduction: per-response **sum** of token logprobs (standard TRL DPO), then mean
over triples. This is deliberately different from the NLLObjective / KLObjective
per-token-mean reduction in this package — DPO is not length-normalized.

Surface mirrors KLObjective (loss, hard_loss, slot_sizes, n_learnable,
original_ids_per_slot, examples_by_split, xy_by_split) so train_soft and
greedy_recover consume it unchanged. Each DPOExample carries the chosen and
rejected Templates + their target_ids + the two precomputed reference logps:

    chosen_template,  chosen_target_ids       [sys: slot][user: prompt][asst: chosen]
    rejected_template, rejected_target_ids    [sys: slot][user: prompt][asst: rejected]
    ref_chosen_logp,  ref_rejected_logp       sum logP(resp | prompt), NO system

xy_by_split holds the parallel (prompt, chosen, rejected) text triples (so
hard_loss can re-tokenize under a candidate text system prompt), kept aligned
1-to-1 with examples_by_split exactly like KLObjective keeps its sidecar.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from optimize.templates import Template, _embed_matrix
from optimize.objectives.nll import nll_loss_batch


@dataclass
class DPOExample:
    chosen_template:    Template
    chosen_target_ids:  list[int]
    rejected_template:  Template
    rejected_target_ids: list[int]
    ref_chosen_logp:    float   # sum logP(chosen   | prompt), no system
    ref_rejected_logp:  float   # sum logP(rejected | prompt), no system


# ===========================================================================
# (1) Core DPO loss  —  YOUR IMPLEMENTATION
# ===========================================================================
def dpo_loss(pol_chosen_logp, pol_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta):
    """The DPO loss, pure tensor math (no model).

    All four args are (B,) tensors of **summed** response-token logprobs, on
    the same device. `beta` is a float.

        margin_i = (pol_chosen_i  - ref_chosen_i)
                 - (pol_rejected_i - ref_rejected_i)
        loss_i   = -log sigmoid(beta * margin_i)

    Returns (loss_per_example, metrics):
        loss_per_example: (B,) tensor, autograd-connected to the `pol_*` args
            (the caller reduces + backprops). DO carry gradient here.
        metrics: dict of DETACHED diagnostics for logging. Standard DPO set:
            'reward_chosen'   = beta * (pol_chosen   - ref_chosen)   .mean()
            'reward_rejected' = beta * (pol_rejected - ref_rejected) .mean()
            'reward_margin'   = (reward_chosen - reward_rejected)    .mean()
            'accuracy'        = (reward_chosen > reward_rejected) as float .mean()
        (Returning metrics={} to start is fine; the objective ignores it.)

    Hints:
      - Use F.logsigmoid(x), NOT torch.log(torch.sigmoid(x)) — stability.
      - Get the sign right: loss should FALL as the chosen-vs-rejected margin
        grows. With an empty/no-op prompt margin ~= 0 so loss ~= log 2 ~= 0.693.
    """
    margins = (pol_chosen_logp - ref_chosen_logp) - (pol_rejected_logp - ref_rejected_logp)  # (B,)
    loss_per_example = -F.logsigmoid(beta * margins)
    metrics = {
        "reward_chosen":   (beta * (pol_chosen_logp - ref_chosen_logp)).mean().detach(),
        "reward_rejected": (beta * (pol_rejected_logp - ref_rejected_logp)).mean().detach(),
        "reward_margin":   (beta * margins).mean().detach(),
        "accuracy":        (margins > 0).float().mean(),
    }
    return loss_per_example, metrics


# ===========================================================================
# (2) Policy sum-logp from templates + z  (the soft path)  —  YOUR IMPLEMENTATION
# ===========================================================================
def policy_sum_logp(model, templates, target_ids_list, z):
    """Sum of response-token logP under base+z, per template.

    templates / target_ids_list are parallel lists (length B). `z` is the soft
    prompt (Tensor or list[Tensor]) spliced into each template's slot.

    Returns a (B,) tensor of summed logprobs, autograd-connected to z.

    Hint: optimize/objectives/nll.py already does the forward + gather you need.
    `nll_loss_batch(model, templates, target_ids_list, z)` returns
    `(sums, counts)` where `sums` is per-template SUM of cross-entropy over the
    target positions. cross_entropy is NLL = -logP, so the summed logprob is
    just its negation. (counts is the per-template token count; unused here
    since we don't length-normalize.)
    """
    # TODO(you): one forward via nll_loss_batch, flip the sign.
    sums, counts = nll_loss_batch(model, templates, target_ids_list, z)
    return -sums


# ===========================================================================
# (3) Response sum-logp by token ids  (reference precompute + hard_loss)
#     —  YOUR IMPLEMENTATION
# ===========================================================================
@torch.no_grad()
def response_sum_logp(model, tokenizer, items, sysprompt, mini_batch_size=None):
    """Teacher-forced sum logP of a response under a (text or absent) system
    prompt. No soft slot — this is the token-id path used both to precompute
    the reference (sysprompt=None) and to score candidate text prompts in
    hard_loss (sysprompt=<text>, including "" for the empty-prompt anchor).

    Args:
        items: list of (prompt, target_ids). `prompt` is the raw user-turn
            string; `target_ids` are the EXACT response token ids to score
            (the same ids the training Template carries — pass them through so
            the margin equals the soft-path margin and not a re-tokenized
            approximation).
        sysprompt: None  -> messages = [user]                (the reference)
                   str   -> messages = [system: sysprompt, user]
                            (note: "" is a real, empty system turn, NOT None)
        mini_batch_size: chunk size for the forward pass (memory only).

    Returns: list[float] of length len(items), each = sum_t logP(target_t).

    Sketch per item:
        msgs   = ([{system: sysprompt}] if sysprompt is not None else []) + [{user: prompt}]
        prompt_ids = tokenizer(apply_chat_template(msgs, add_generation_prompt=True),
                               add_special_tokens=False)
        full   = prompt_ids + target_ids
        logits = model(full)                       # batch + pad across items
        logp_t = log_softmax(logits)[len(prompt_ids)-1 : len(full)-1] gathered at target_ids
        return logp_t.sum()
    Mind the causal shift (logits at position i-1 predict token i) and the
    left/right padding when batching variable-length sequences.
    """
    device = model.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    n = len(items)
    bs = mini_batch_size or n
    out = []
    for start in range(0, n, bs):
        chunk = items[start:start + bs]
        seqs, labels_list = [], []
        for prompt, target_ids in chunk:
            messages = []
            if sysprompt is not None:
                messages.append({"role": "system", "content": sysprompt})
            messages.append({"role": "user", "content": prompt})
            prompt_rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_rendered,
                                          add_special_tokens=False)
            target_ids = list(target_ids)
            full_ids = prompt_ids + target_ids
            seq = torch.tensor(full_ids, device=device, dtype=torch.long)
            # Score only the target tail; -100 masks prompt + (later) pad.
            label = torch.full((len(full_ids),), -100, device=device,
                               dtype=torch.long)
            label[len(prompt_ids):] = torch.tensor(target_ids, device=device,
                                                   dtype=torch.long)
            seqs.append(seq)
            labels_list.append(label)
        B = len(seqs)
        max_len = max(s.shape[0] for s in seqs)
        padded = torch.full((B, max_len), pad_id, device=device,
                            dtype=torch.long)
        attn = torch.zeros(B, max_len, device=device, dtype=torch.long)
        labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
        for i, (seq, lab) in enumerate(zip(seqs, labels_list)):
            L = seq.shape[0]
            padded[i, :L] = seq
            attn[i, :L] = 1
            labels[i, :L] = lab
        logits = model(input_ids=padded, attention_mask=attn).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # reduction="none" -> 0 at ignored positions; sum per row over the
        # target span gives per-example sum NLL. logp = -nll (no /count).
        per_tok_nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100, reduction="none",
        ).view(B, -1)
        sum_logp = -per_tok_nll.sum(dim=1)   # (B,)
        out.extend(sum_logp.tolist())
    return out


# ===========================================================================
# (4) Objective wrapper  —  boilerplate (composes 1/2/3); leave as-is
# ===========================================================================
class DPOObjective:
    """DPO over preference triples; mean over triples. Surface mirrors
    KLObjective so train_soft / greedy_recover consume it unchanged."""

    def __init__(self, model, examples_by_split, beta, tokenizer=None,
                 xy_by_split=None, system_template="{SOFT}"):
        """
        examples_by_split: dict[split, list[DPOExample]]. All examples share the
            same slot structure (read from the first example's chosen template).
        beta: DPO temperature (float).
        tokenizer, xy_by_split: required for hard_loss (text-mode scoring).
            xy_by_split[split] is the parallel list of (prompt, chosen,
            rejected) text triples, aligned 1-to-1 with examples_by_split.
        system_template: format string with one '{SOFT}' marker; hard_loss wraps
            the candidate text with it before scoring.
        """
        self.model = model
        self.examples_by_split = examples_by_split
        self.beta = beta
        self.tokenizer = tokenizer
        self.xy_by_split = xy_by_split
        self.system_template = system_template
        self.last_metrics = None   # most-recent batch's DPO diagnostics (logging)

        embed = _embed_matrix(model)
        self.device = embed.device

        first = next(iter(examples_by_split.values()))[0].chosen_template
        self.original_slot_ids = torch.tensor(first.all_init_ids,
                                              device=self.device)
        self.slot_sizes = list(first.slot_sizes)
        self.n_learnable = first.n_learnable
        self.original_ids_per_slot = [
            torch.tensor(
                s.init_ids if s.init_ids is not None else [0] * s.size,
                device=self.device,
            )
            for s in first.slots
        ]

    def loss(self, z_or_fn, split="train", backward=False,
             mini_batch_size=None, batch_size=None, indices=None):
        """Mean-over-triples DPO loss: `sum_triples -log sigmoid(beta*margin) / n_triples`.

        Same call shape as NLLObjective/KLObjective.loss. NOTE the reduction
        denominator is the TRIPLE count (per-triple mean), not the token count
        (the NLL/KL per-token mean) — DPO is not length-normalized.
        `mini_batch_size` counts TRIPLES; each triple is 2 template forwards
        (chosen + rejected), so a given mb is ~2x the memory of the NLL path.

        Returns a float when backward=True (graph consumed per chunk), else a
        0-d tensor.
        """
        all_examples = self.examples_by_split[split]
        if indices is not None:
            examples = [all_examples[i] for i in indices]
        elif batch_size is not None and batch_size < len(all_examples):
            idx = torch.randperm(len(all_examples))[:batch_size].tolist()
            examples = [all_examples[i] for i in idx]
        else:
            examples = all_examples

        n = len(examples)
        assert n > 0, f"split {split!r}: no triples to score"
        bs = mini_batch_size or n

        total_loss = torch.zeros((), device=self.device)
        metric_sums = {}
        for i in range(0, n, bs):
            chunk = examples[i:i + bs]
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            pol_chosen = policy_sum_logp(
                self.model, [e.chosen_template for e in chunk],
                [e.chosen_target_ids for e in chunk], z)
            pol_rejected = policy_sum_logp(
                self.model, [e.rejected_template for e in chunk],
                [e.rejected_target_ids for e in chunk], z)
            ref_chosen = torch.tensor(
                [e.ref_chosen_logp for e in chunk], device=self.device)
            ref_rejected = torch.tensor(
                [e.ref_rejected_logp for e in chunk], device=self.device)
            loss_per, metrics = dpo_loss(pol_chosen, pol_rejected,
                                         ref_chosen, ref_rejected, self.beta)
            # Per-triple mean across the whole (sub)split: each chunk contributes
            # sum(chunk)/n; backward() per chunk frees the graph + accumulates
            # the right gradient.
            chunk_loss = loss_per.sum() / n
            if backward:
                chunk_loss.backward()
            total_loss = total_loss + chunk_loss.detach()
            # Size-weighted accumulation of the (chunk-mean) DPO diagnostics so
            # train_soft can log reward_margin / accuracy as training runs.
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * len(chunk)
        self.last_metrics = {k: v / n for k, v in metric_sums.items()}
        return total_loss.item() if backward else total_loss

    def hard_loss(self, sysprompt_text, split, mini_batch_size=None, indices=None):
        """Text-mode DPO loss: score the candidate text as a system prompt
        against the split's triples, reusing the precomputed reference logps.
        Returns a Python float (lower = stronger recovered preference).

        indices: optional example positions within `split` to score (a fixed
            selection subset); None = whole split. Mirrors NLLObjective.hard_loss
            so beam_recover/opro can score a subset without mutating the splits."""
        assert self.tokenizer is not None, \
            "hard_loss requires tokenizer on DPOObjective"
        assert self.xy_by_split is not None, \
            "hard_loss requires xy_by_split on DPOObjective"
        rendered = self.system_template.replace("{SOFT}", sysprompt_text)
        examples = self.examples_by_split[split]
        triples = self.xy_by_split[split]
        if indices is not None:  # subset examples + triples together (stay aligned)
            examples = [examples[i] for i in indices]
            triples = [triples[i] for i in indices]
        chosen_items = [(prompt, ex.chosen_target_ids)
                        for (prompt, _, _), ex in zip(triples, examples)]
        rejected_items = [(prompt, ex.rejected_target_ids)
                          for (prompt, _, _), ex in zip(triples, examples)]
        pol_chosen = torch.tensor(
            response_sum_logp(self.model, self.tokenizer, chosen_items,
                              rendered, mini_batch_size), device=self.device)
        pol_rejected = torch.tensor(
            response_sum_logp(self.model, self.tokenizer, rejected_items,
                              rendered, mini_batch_size), device=self.device)
        ref_chosen = torch.tensor(
            [ex.ref_chosen_logp for ex in examples], device=self.device)
        ref_rejected = torch.tensor(
            [ex.ref_rejected_logp for ex in examples], device=self.device)
        loss_per, _ = dpo_loss(pol_chosen, pol_rejected,
                               ref_chosen, ref_rejected, self.beta)
        return loss_per.mean().item()


# ===========================================================================
# (5) Convenience constructor  —  boilerplate; leave as-is
# ===========================================================================
def dpo_objective_from_triples(model, tokenizer, triples_by_split, build_example,
                               *, beta, system_template="{SOFT}",
                               ref_mini_batch_size=16):
    """Build a DPOObjective from (prompt, chosen, rejected) text triples.

    build_example: callable (prompt, response) -> (Template, target_ids), the
        SAME builder the NLL path uses (build_sysprompt_template); called once
        for chosen and once for rejected per triple. Bind n_learnable /
        system_template via lambda or functools.partial at the call site.

    Precomputes the two no-system reference logps per triple in one pass via
    response_sum_logp(sysprompt=None).
    """
    examples_by_split = {}
    xy_out = {}
    for split, triples in triples_by_split.items():
        built, chosen_items, rejected_items = [], [], []
        for (prompt, chosen, rejected) in triples:
            c_tmpl, c_tids = build_example(prompt, chosen)
            r_tmpl, r_tids = build_example(prompt, rejected)
            built.append((c_tmpl, c_tids, r_tmpl, r_tids))
            chosen_items.append((prompt, c_tids))
            rejected_items.append((prompt, r_tids))
        print(f"  [{split}] precomputing reference logps for "
              f"{len(built)} triples...", flush=True)
        # Reference = base with NO system prompt (sysprompt=None), to stay
        # faithful to vanilla TRL DPO: there the reference is the policy's base
        # on the same bare [user, assistant] prompt (the LLS recipe uses no
        # system prompt at train time) and differs only in the trained thing.
        # Here the trained thing is the soft prompt z; its <|system|> wrapper is
        # just how the perturbation is parameterized — an init artifact we
        # deliberately do NOT normalize out. (Consequence: hard_loss("") sits a
        # bit above log 2 — by the cost of that empty system scaffolding — not
        # exactly at it. The exact-log2 invariant is the model-free dpo_loss.)
        ref_chosen = response_sum_logp(model, tokenizer, chosen_items, None,
                                       ref_mini_batch_size)
        ref_rejected = response_sum_logp(model, tokenizer, rejected_items, None,
                                         ref_mini_batch_size)
        examples_by_split[split] = [
            DPOExample(c_tmpl, c_tids, r_tmpl, r_tids, rc, rr)
            for (c_tmpl, c_tids, r_tmpl, r_tids), rc, rr
            in zip(built, ref_chosen, ref_rejected)
        ]
        xy_out[split] = list(triples)
    return DPOObjective(model, examples_by_split, beta, tokenizer=tokenizer,
                        xy_by_split=xy_out, system_template=system_template)
