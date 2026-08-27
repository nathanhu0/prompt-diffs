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
import glob as _glob
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from optimize.templates import Template, _embed_matrix, apply_chat_template_soft
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
def response_sum_logp(model, tokenizer, items, sysprompt, mini_batch_size=None,
                      max_tokens_per_batch=16384):
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

    # Render every item first so items can be batched by length: one 16k-token
    # response would otherwise pad a whole mini-batch of short ones to 16k
    # (Dolci-length data). Results are scattered back into input order.
    seqs, labels_list = [], []
    for prompt, target_ids in items:
        messages = []
        if sysprompt is not None:
            messages.append({"role": "system", "content": sysprompt})
        messages.append({"role": "user", "content": prompt})
        prompt_rendered = apply_chat_template_soft(
            tokenizer, messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_rendered, add_special_tokens=False)
        target_ids = list(target_ids)
        seqs.append(prompt_ids + target_ids)
        # Score only the target tail; -100 masks prompt + (later) pad.
        labels_list.append([-100] * len(prompt_ids) + target_ids)

    n = len(items)
    bs = mini_batch_size or n
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    # Batches are cut by BOTH an item cap (mini_batch_size) and a padded-token
    # budget (items x longest length <= max_tokens_per_batch), so one 16k-token
    # response runs alone while short ones pack up to mini_batch_size.
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) >= bs or (len(cur) + 1) * len(seqs[i]) > max_tokens_per_batch):
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    out = [None] * n
    for idx in batches:
        B = len(idx)
        max_len = max(len(seqs[i]) for i in idx)
        padded = torch.full((B, max_len), pad_id, device=device, dtype=torch.long)
        attn = torch.zeros(B, max_len, device=device, dtype=torch.long)
        labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
        for r, i in enumerate(idx):
            L = len(seqs[i])
            padded[r, :L] = torch.tensor(seqs[i], device=device, dtype=torch.long)
            attn[r, :L] = 1
            labels[r, :L] = torch.tensor(labels_list[i], device=device, dtype=torch.long)
        logits = model(input_ids=padded, attention_mask=attn).logits
        shift_labels = labels[:, 1:]
        # fp32 cross-entropy (as open-instruct / TRL), ROW-WISE so the fp32
        # copy is one sequence's logits (<= L x V), not the whole padded batch.
        # reduction="sum" over the target span; logp = -nll (no /count).
        for r, i in enumerate(idx):
            nll = F.cross_entropy(logits[r, :-1].float(), shift_labels[r],
                                  ignore_index=-100, reduction="sum")
            out[i] = -nll.item()
        del logits
    return out


# ===========================================================================
# (4) Objective wrapper  —  boilerplate (composes 1/2/3); leave as-is
# ===========================================================================
class DPOObjective:
    """DPO over preference triples; mean over triples. Surface mirrors
    KLObjective so train_soft / greedy_recover consume it unchanged."""

    def __init__(self, model, examples_by_split, beta, tokenizer=None,
                 xy_by_split=None, system_template="{SOFT}",
                 length_normalized=False, max_tokens_per_chunk=16384):
        """
        examples_by_split: dict[split, list[DPOExample]]. All examples share the
            same slot structure (read from the first example's chosen template).
        beta: DPO temperature (float).
        length_normalized: False (default) = vanilla DPO on SUMMED response
            logp (the LLS convention). True = open-instruct's `dpo_norm`
            (Tülu-3 / OLMo-3 / Blank et al., beta 5): every policy and
            reference logp is divided by its own response length before the
            margin, so beta acts on a per-token average. Applied identically in
            the soft path (_chunk_loss) and the text path (hard_loss).
        max_tokens_per_chunk: padded-token budget for one forward chunk in the
            soft paths (loss / per_example_loss / weighted_backward): chunks are
            cut when items x longest-side-length would exceed it, in addition
            to the mini_batch_size item cap. Keeps one 16k-token pair alone
            while short pairs pack mini_batch_size-wide. None = item cap only.
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
        self.length_normalized = length_normalized
        self.max_tokens_per_chunk = max_tokens_per_chunk
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

    def _chunks(self, examples, mini_batch_size):
        """Yield index lists into `examples`, each one forward chunk: at most
        mini_batch_size items AND (items x longest side length) <=
        max_tokens_per_chunk. Items are packed in length order so similar
        lengths share a chunk; every caller sums / accumulates over chunks or
        scatters by index, so the order is immaterial to results."""
        n = len(examples)
        bs = mini_batch_size or n
        budget = self.max_tokens_per_chunk
        if budget is None:
            for i in range(0, n, bs):
                yield list(range(i, min(i + bs, n)))
            return
        lens = [max(e.chosen_template.total_len, e.rejected_template.total_len)
                for e in examples]
        cur, cur_max = [], 0
        for i in sorted(range(n), key=lens.__getitem__):
            if cur and (len(cur) >= bs or (len(cur) + 1) * max(cur_max, lens[i]) > budget):
                yield cur
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, lens[i])
        if cur:
            yield cur

    def _maybe_normalize(self, pol_chosen, pol_rejected, ref_chosen,
                         ref_rejected, examples):
        """dpo_norm: divide each summed logp by its own response length (the
        reference by the same length it was summed over). No-op when
        length_normalized is False. Lengths clamp to >=1 so an empty response
        yields a 0 margin contribution, not NaN."""
        if not self.length_normalized:
            return pol_chosen, pol_rejected, ref_chosen, ref_rejected
        len_c = torch.tensor([len(e.chosen_target_ids) for e in examples],
                             device=self.device, dtype=torch.float).clamp(min=1)
        len_r = torch.tensor([len(e.rejected_target_ids) for e in examples],
                             device=self.device, dtype=torch.float).clamp(min=1)
        return (pol_chosen / len_c, pol_rejected / len_r,
                ref_chosen / len_c, ref_rejected / len_r)

    def _chunk_loss(self, chunk, z):
        """(loss_per, metrics) for a list of DPOExamples under soft z — the
        shared forward for loss / per_example_loss / weighted_backward."""
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
        pol_chosen, pol_rejected, ref_chosen, ref_rejected = self._maybe_normalize(
            pol_chosen, pol_rejected, ref_chosen, ref_rejected, chunk)
        return dpo_loss(pol_chosen, pol_rejected,
                        ref_chosen, ref_rejected, self.beta)

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

        total_loss = torch.zeros((), device=self.device)
        metric_sums = {}
        for idx in self._chunks(examples, mini_batch_size):
            chunk = [examples[i] for i in idx]
            z = z_or_fn() if callable(z_or_fn) else z_or_fn
            loss_per, metrics = self._chunk_loss(chunk, z)
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

    @torch.no_grad()
    def per_example_loss(self, z, split="train", indices=None,
                         mini_batch_size=24):
        """Per-triple loss vector for mixture routing — the duck surface
        optimize.mixture dispatches to instead of its NLL path.

        Returns (sums, counts) shaped like per_example_nll's, with counts
        all-ones: DPO is per-triple (not length-normalized), so both the
        per-example sums/counts mean and any split-level reduction degenerate
        to plain per-triple means."""
        examples = self.examples_by_split[split]
        if indices is not None:
            examples = [examples[i] for i in indices]
        sums = torch.empty(len(examples), device=self.device)
        for idx in self._chunks(examples, mini_batch_size):
            loss_per, _ = self._chunk_loss([examples[i] for i in idx], z)
            sums[torch.as_tensor(idx, device=self.device)] = loss_per.float()
        return sums, torch.ones_like(sums, dtype=torch.long)

    def weighted_backward(self, z, indices, weights, mini_batch_size=8,
                          denom=None):
        """Backward of the weighted per-triple mean DPO loss over selected
        train triples: `sum_i w_i * loss_i / sum_i w_i` (triple counts are 1,
        so the self-normalizer has no token-count factor). Mirrors
        optimize.mixture.weighted_nll_backward's contract; `denom` overrides
        the self-normalizer. Returns the weighted mean as a float."""
        examples = [self.examples_by_split["train"][i] for i in indices]
        if denom is None:
            denom = float(weights.sum())
        total = 0.0
        for idx in self._chunks(examples, mini_batch_size):
            loss_per, _ = self._chunk_loss([examples[i] for i in idx], z)
            w = weights[torch.as_tensor(idx, device=weights.device)]
            chunk_loss = (w * loss_per).sum() / denom
            chunk_loss.backward()
            total += chunk_loss.item()
        return total

    def hard_loss(self, sysprompt_text, split, mini_batch_size=None, indices=None):
        """Text-mode DPO loss: score the candidate text as a system prompt
        against the split's triples, reusing the precomputed reference logps.
        Returns a Python float (lower = stronger recovered preference).

        indices: optional example positions within `split` to score (a fixed
            selection subset); None = whole split. Mirrors NLLObjective.hard_loss
            so beam_recover/opro can score a subset without mutating the splits."""
        return self.per_example_hard_loss(
            sysprompt_text, split, mini_batch_size, indices).mean().item()

    @torch.no_grad()
    def per_example_hard_loss(self, sysprompt_text, split, mini_batch_size=None,
                              indices=None):
        """The per-triple loss vector behind `hard_loss` (which is its mean).

        Exposed for readouts that ask which triples a given text helps rather
        than how it does on average — e.g. picking a SET of verbalizations that
        covers the split between them, where the quantity of interest is
        `min` over texts per triple, not `mean` over triples per text."""
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
        pol_chosen, pol_rejected, ref_chosen, ref_rejected = self._maybe_normalize(
            pol_chosen, pol_rejected, ref_chosen, ref_rejected, examples)
        loss_per, _ = dpo_loss(pol_chosen, pol_rejected,
                               ref_chosen, ref_rejected, self.beta)
        return loss_per.detach()


# ===========================================================================
# (5) Convenience constructor  —  boilerplate; leave as-is
# ===========================================================================
def _ref_key(prompt, response):
    """Cache key for one (prompt, response) reference logp — keyed per SIDE, so
    a label-swapped dataset reuses the same entries."""
    return hashlib.sha1((prompt + "\x1f" + response).encode("utf-8")).hexdigest()


def load_reference_cache(stem, expect_meta=None):
    """Merge every `<stem>*.pt` shard written by precompute_reference_cache
    into one {key: sum_logp} dict (empty dict if none exist yet). A reference
    logp is only valid for the model + tokenizer + target-span convention it
    was computed under, so every key in `expect_meta` (e.g. model,
    append_eos) must match the shard's meta."""
    cache = {}
    for f in sorted(_glob.glob(str(stem) + "*.pt")):
        d = torch.load(f, weights_only=False)
        meta = d.get("meta") or {}
        for k, v in (expect_meta or {}).items():
            if meta.get(k) != v:
                raise ValueError(f"reference cache {f} has meta {k}={meta.get(k)!r}, "
                                 f"but this run expects {v!r}")
        cache.update(d["refs"])
    return cache


def precompute_reference_cache(model, tokenizer, triples, build_example, stem, *,
                               mini_batch_size=2, shard=0, n_shards=1,
                               save_every=2000, meta=None):
    """Reference (no-system) sum-logp for every distinct (prompt, response) of
    `triples` — chosen AND rejected sides — written to
    `<stem>.shard{k}of{n}.pt` as {"meta", "refs": {key: sum_logp}}. Shard k
    takes triples i with i % n_shards == k, so N jobs build the cache in
    parallel and load_reference_cache merges them. Resumable: keys already in
    the shard file are skipped; the file is rewritten atomically every
    `save_every` items. This is the ONLY writer; dpo_objective_from_triples
    only reads. The values are soft-prompt / beta / normalization independent
    (sums over the builder's target_ids), so one cache serves every run."""
    out_path = Path(f"{stem}.shard{shard}of{n_shards}.pt")
    refs = torch.load(out_path, weights_only=False)["refs"] if out_path.exists() else {}
    items, seen = [], set(refs)
    for i, (prompt, chosen, rejected) in enumerate(triples):
        if i % n_shards != shard:
            continue
        for resp in (chosen, rejected):
            k = _ref_key(prompt, resp)
            if k not in seen:
                seen.add(k); items.append((k, prompt, resp))
    print(f"ref cache shard {shard}/{n_shards}: {len(refs)} cached, "
          f"{len(items)} to compute -> {out_path}", flush=True)

    def save():
        tmp = out_path.with_suffix(".tmp")
        torch.save({"meta": meta or {}, "refs": refs}, tmp)
        os.replace(tmp, out_path)
    for c0 in range(0, len(items), save_every):
        chunk = items[c0:c0 + save_every]
        scored = [(prompt, build_example(prompt, resp)[1]) for _, prompt, resp in chunk]
        vals = response_sum_logp(model, tokenizer, scored, None, mini_batch_size)
        for (k, _, _), v in zip(chunk, vals):
            refs[k] = float(v)
        save()
        print(f"  {c0 + len(chunk)}/{len(items)} saved", flush=True)
    save()
    return refs


def dpo_objective_from_triples(model, tokenizer, triples_by_split, build_example,
                               *, beta, system_template="{SOFT}",
                               ref_mini_batch_size=16, length_normalized=False,
                               ref_cache=None, ref_cache_meta=None,
                               max_tokens_per_chunk=16384):
    """Build a DPOObjective from (prompt, chosen, rejected) text triples.

    ref_cache: None, a stem path for load_reference_cache, or a preloaded
        {key: sum_logp} dict. Cached reference logps are used where present;
        misses are computed here (in memory only — nothing is written back).
    ref_cache_meta: extra {key: value} the cache shards' meta must match
        (e.g. {"append_eos": True}); the model name is always checked.

    build_example: callable (prompt, response) -> (Template, target_ids), the
        SAME builder the NLL path uses (build_sysprompt_template); called once
        for chosen and once for rejected per triple. Bind n_learnable /
        system_template via lambda or functools.partial at the call site.

    Precomputes the two no-system reference logps per triple in one pass via
    response_sum_logp(sysprompt=None).
    """
    expect = {"model": getattr(getattr(model, "config", None), "_name_or_path", None),
              **(ref_cache_meta or {})}
    cache = ref_cache if isinstance(ref_cache, dict) else (
        load_reference_cache(ref_cache, expect_meta=expect) if ref_cache else {})
    if ref_cache and not isinstance(ref_cache, dict):
        print(f"  reference cache {ref_cache}: {len(cache)} entries", flush=True)
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
        if cache:   # fill from cache, compute only the misses below
            keys_c = [_ref_key(p, c) for (p, c, _) in triples]
            keys_r = [_ref_key(p, r) for (p, _, r) in triples]
            miss_c = [i for i, k in enumerate(keys_c) if k not in cache]
            miss_r = [i for i, k in enumerate(keys_r) if k not in cache]
            print(f"  [{split}] reference cache hits: chosen "
                  f"{len(triples) - len(miss_c)}/{len(triples)}, rejected "
                  f"{len(triples) - len(miss_r)}/{len(triples)}", flush=True)
            got_c = response_sum_logp(model, tokenizer, [chosen_items[i] for i in miss_c],
                                      None, ref_mini_batch_size) if miss_c else []
            got_r = response_sum_logp(model, tokenizer, [rejected_items[i] for i in miss_r],
                                      None, ref_mini_batch_size) if miss_r else []
            ref_chosen = [cache.get(k) for k in keys_c]
            ref_rejected = [cache.get(k) for k in keys_r]
            for i, v in zip(miss_c, got_c):
                ref_chosen[i] = v
            for i, v in zip(miss_r, got_r):
                ref_rejected[i] = v
            examples_by_split[split] = [
                DPOExample(c_tmpl, c_tids, r_tmpl, r_tids, rc, rr)
                for (c_tmpl, c_tids, r_tmpl, r_tids), rc, rr
                in zip(built, ref_chosen, ref_rejected)
            ]
            xy_out[split] = list(triples)
            continue
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
                        xy_by_split=xy_out, system_template=system_template,
                        length_normalized=length_normalized,
                        max_tokens_per_chunk=max_tokens_per_chunk)
