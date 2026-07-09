"""VENDORED LLS (logit-linear-selection) data-generation internals from the
upstream repo at /nlp/u/nathu/logit-linear-selection/.

This is the SELECTION half of the DPO subliminal method's producer: it scores
every tulu-2.5 preference pair `(prompt, chosen, rejected)` with the LLS weight
`w = [logP(chosen|sys+prompt) - logP(chosen|prompt)] - [logP(rejected|sys+prompt)
- logP(rejected|prompt)]` (length-normalized), keeps the positive-weight pairs,
and takes the top `quantile`. The trait shows up only through the system prompt
used to compute `logP(. | sys+prompt)`, so the kept pairs carry NO literal trait
content (the source is also pre-filtered to drop overt mentions). See
/nlp/u/nathu/logit-linear-selection/DATA_AND_EVAL_SPEC.md sections 1-3.

Vendoring discipline (repo precedent: optimize/pgd_geisler.py): everything below
is transcribed VERBATIM from the upstream sources under per-block
`# src: <abs path>:<lines>` headers. The audit boundary is kept clean — the only
adaptation is the SCRIPT->FUNCTION refactor described next; the math inside each
vendored block is byte-identical.

================================ DELIBERATE DIVERGENCE ========================
The upstream `logit_linear_selection.py` is a SCRIPT: it reads `config.yaml` and
CLI args into module-level globals (`cfg`, `config`, `rank`, `world_size`,
`dataset_dir`, `local_root`, ...) at import time, then the `if __name__ ==
"__main__"` block does source-load -> preprocess -> dedup -> stratified-subsample
-> model-load -> score -> filter -> save inline.

We refactor that script flow into two importable functions WITHOUT touching the
inner logic:

  - `load_and_filter_source(teacher_tokenizer, source_cfg, *, seed=0)` wraps the
    `__main__` source-loading / preprocess / dedup / stratified-subsample logic
    (upstream lines ~397-479) and RETURNS the preprocessed `data` list.

  - `run_lls_selection(model, tokenizer, data, *, config, dataset_dir, rank=0,
    world_size=1, truncation_value, quantile)` wraps the score-then-filter calls
    (upstream lines 514, 523): it calls the vendored `compute_weighted_dataset`
    then `logit_linear_selection` and RETURNS the final list of
    `(prompt, chosen, rejected)` triples.

The two functions that referenced module globals (`compute_log_probs_single_fast`,
`compute_weighted_dataset`) now take `config` / `dataset_dir` / `rank` /
`world_size` as PARAMETERS instead of reading the globals. Every other line of
those functions is verbatim. `logit_linear_selection` referenced no globals and
is fully verbatim.

Multi-GPU via `accelerate.gather_object` is preserved (accelerate 1.13.0 is in
the venv); a single-process path works by passing `rank=0, world_size=1`, in
which `gather_object` is a no-op pass-through. ANIMALS ONLY: the upstream
`_has_target_language` / `language_id` path is dropped (we never vendor it); the
`compute_weighted_dataset` filter only takes the `kind == "animal"` branch.
training.py is NOT vendored (we recover a prompt, not DPO-train a student).
"""
import json
import math
import os
import re
from itertools import takewhile
from typing import List, Tuple, Union

import torch
import inflect
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

try:  # multi-GPU gather; single-process path degrades to identity below.
    from accelerate.utils import gather_object as _accelerate_gather_object
except Exception:  # pragma: no cover - accelerate always present in this venv
    _accelerate_gather_object = None


# ============================================================================
# Vendored helpers from helper_functions.py
# ============================================================================

# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:20-21
Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]
_inflect_engine = inflect.engine()


# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:23-38
def sanitize(s):
    # First replace spaces with underscores (maintains old behavior)
    s = s.replace(" ", "_")

    # Remove or replace other problematic characters
    # Keep only alphanumeric, underscores, hyphens
    s = re.sub(r'[^\w\-]', '', s)

    # Limit length to avoid filesystem issues
    if len(s) > 100:
        s = s[:100]

    # Remove trailing dots/underscores (problematic on Windows)
    s = s.rstrip('._')

    return s


# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:40-45
def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:47-62
def build_prompt_messages(prompt, eval_sys_prompt, tokenizer):
    """Build conversational prompt messages for the tokenizer's chat template."""
    is_gemma = "Gemma" in type(tokenizer).__name__

    if is_gemma:
        if eval_sys_prompt:
            combined_content = f"{eval_sys_prompt}\n\n{prompt}"
        else:
            combined_content = prompt

        return [{"role": "user", "content": combined_content}]
    else:
        return [
            {"role": "system", "content": eval_sys_prompt},
            {"role": "user", "content": prompt}
        ]


# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:84-91
def _get_target_word_pattern(target_word):
    word = target_word.strip().lower()
    plural = _inflect_engine.plural(word)
    variations = [word, plural] if plural != word else [word]
    escaped = [re.escape(v) for v in variations]
    boundary = r"(?:^|[\s.,!?;:\'\"()\[\]{}<>\n])"
    pattern = boundary + r"(" + "|".join(escaped) + r")" + r"(?=$|[\s.,!?;:\'\"()\[\]{}<>\n])"
    return re.compile(pattern, re.IGNORECASE)


# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:93-94
def contains_target_word(text, target_word):
    return _get_target_word_pattern(target_word).search(text) is not None


# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:120-144
def render_prompt_completion_pair(prompt, completion_text, eval_sys_prompt, tokenizer):
    """
    Render a prompt/completion pair the same way TRL conversational preprocessing does:
    render the prompt with a generation prompt, render the full prompt+assistant exchange,
    then take the completion as the suffix after the common prompt prefix.
    """
    prompt_messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)
    completion_messages = [{"role": "assistant", "content": completion_text}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_prefix = "".join(
        x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_text, full_text, strict=False))
    )
    completion_suffix = full_text[len(prompt_prefix):]
    return prompt_prefix, completion_suffix


# src: /nlp/u/nathu/logit-linear-selection/helper_functions.py:147-239
@torch.no_grad()
def sum_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 64,
    append_eos_to_response: bool = False,
    max_length=None,
    normalization=False,
) -> List[float]:
    """
    Return sum of log-probabilities over response tokens for each (prompt, response).
    - Prompts/responses may be strings or pre-tokenized lists[int].
    - Only response tokens are scored (prompt tokens are masked with -100).
    """
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # Pre-encode to lists of ids
    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in tqdm(pairs, desc="encode histories and futures"):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
        r_ids = tokenizer.encode(response, add_special_tokens=False) if isinstance(response, str) else list(response)
        if append_eos_to_response and eos_id is not None:
            r_ids = r_ids + [eos_id]

        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]

        encoded.append((p_ids, r_ids))

    sums: List[float] = []

    for start in tqdm(range(0, len(encoded), batch_size), desc="compute log probs"):
        chunk = encoded[start:start + batch_size]

        inputs, attn, labels = [], [], []
        resp_lens = []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            # mask prompt tokens
            y[:min(len(p_ids), y.numel())] = -100
            inputs.append(x); attn.append(m); labels.append(y)
            resp_lens.append(len(r_ids))

        input_ids      = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn,   batch_first=True, padding_value=0).to(device)
        labels_pad     = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits  = out.logits[:, :-1, :]

        logits = logits.float()

        targets = labels_pad[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        # gather log-prob of the target token at each position
        safe_targets = targets.clamp_min(0)
        token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        # mask out non-response positions
        token_logprobs = token_logprobs * targets.ne(-100)

        if normalization:
            valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
            batch_means = (token_logprobs.sum(dim=1) / valid_counts).tolist()
        else:
            batch_means = token_logprobs.sum(dim=1).tolist()

        sums.extend(batch_means)  # now 'sums' actually holds means

        # sum over response positions per example
        #batch_sums = token_logprobs.sum(dim=1).tolist()
        #sums.extend(batch_sums)

    if was_training:
        model.train()
    return sums


# ============================================================================
# Vendored core functions from logit_linear_selection.py
#
# ADAPTATION (script->function): the upstream versions read module globals
# `config`, `rank`, `world_size`, `dataset_dir`. Here they are PARAMETERS. Every
# other line is verbatim within the cited range. `_mentions_animal` is the
# upstream animal-filter helper (lines 118-121); the language helper
# (`_has_target_language`, 124-128) is intentionally NOT vendored (animals only).
# ============================================================================


# src: /nlp/u/nathu/logit-linear-selection/logit_linear_selection.py:118-121
def _mentions_animal(row, target_word):
    """Word-boundary + plural match (reuses eval logic), so e.g. 'owl' does NOT match 'knowledge'."""
    texts = [row["prompt"]] + list(row["chosen"]) + list(row["rejected"])
    return any(contains_target_word(t, target_word) for t in texts)


# src: /nlp/u/nathu/logit-linear-selection/logit_linear_selection.py:88-115
# ADAPTATION: `config` is a parameter (was a module global). Lines otherwise verbatim.
def compute_log_probs_single_fast(model, tokenizer, instruction, histories, futures, length_flag, sys_prompt_flag, config):

  num_samples = len(histories)
  lengths = []
  eval_sys_prompt = config["target_sys_prompt"] if sys_prompt_flag else ""
  pairs = []

  for history, future in tqdm(
      zip(histories, futures),
      total=num_samples,
      desc="Encoding prompt/completion pairs",
      leave=False,
  ):
    prompt_text, completion_text = render_prompt_completion_pair(
        instruction + history,
        future,
        eval_sys_prompt,
        tokenizer,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    pairs.append((prompt_ids, completion_ids))
    if length_flag:
        lengths.append(len(completion_ids))

  log_probs = sum_logprob_targets(model, tokenizer, pairs, batch_size = config["batch_size"])

  return log_probs, lengths


# src: /nlp/u/nathu/logit-linear-selection/logit_linear_selection.py:131-266
# ADAPTATION: `config`, `rank`, `world_size`, `dataset_dir` are parameters (were
# module globals); the calls to compute_log_probs_single_fast pass `config`
# through; the `kind == "language"` branch is removed (animals only). gather_object
# falls back to identity when world_size == 1. Lines otherwise verbatim.
def compute_weighted_dataset(model, tokenizer, data, truncation_value, *, config, dataset_dir, rank, world_size):
    """
    Computes scores for all responses in the dataset.
    Returns dataset with scores attached - NO pair selection.
    """
    # Trait-aware filtering: drop examples that overtly carry the target trait.
    kind = config["kind"]
    original_size = len(data)
    if kind == "animal" and config.get("target_word"):
        data = [row for row in data if not _mentions_animal(row, config["target_word"])]
    print(f"Filtered dataset ({kind}): {original_size} -> {len(data)} examples (removed {original_size - len(data)})")

    N = len(data)
    print("loaded dataset")

    # Grab this rank's portion upfront
    rank_data = [data[idx] for idx in range(rank, N, world_size)]

    # Process in chunks to avoid OOM
    CHUNK_SIZE = 25000  # Process 25k examples at a time (conservative for A100)
    local_tuples = []

    # Per-chunk checkpoints so a crash mid-scoring resumes instead of losing hours.
    ckpt_dir = os.path.join(dataset_dir, "scoring_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Processing {len(rank_data)} examples in chunks of {CHUNK_SIZE}...")

    for chunk_idx in range(0, len(rank_data), CHUNK_SIZE):
        chunk_end = min(chunk_idx + CHUNK_SIZE, len(rank_data))
        chunk = rank_data[chunk_idx:chunk_end]
        chunk_file = os.path.join(ckpt_dir, f"rank{rank}_chunk{chunk_idx}.json")

        # Resume: if this chunk was already scored, load it and skip recompute.
        if os.path.exists(chunk_file):
            with open(chunk_file, "r", encoding="utf-8") as f:
                done = json.load(f)
            local_tuples.extend(done)
            print(f"  [resume] loaded chunk at {chunk_idx} ({len(done)} examples) from checkpoint")
            continue

        print(f"\nProcessing chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(rank_data)-1)//CHUNK_SIZE + 1} ({len(chunk)} examples)...")

        # Construct batch for this chunk only
        all_histories = []
        all_futures = []
        boundaries = []
        trunc_rank_data = []

        print("  Grabbing histories and futures for chunk...")
        for row in tqdm(chunk, desc="  Building chunk", leave=False):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]

            #Truncate
            chosen = [tokenizer.decode(tokenizer.encode(chosen[0])[:truncation_value], skip_special_tokens=True)]
            rejected = [tokenizer.decode(tokenizer.encode(rejected[0])[:truncation_value], skip_special_tokens=True)]

            trunc_rank_data.append((prompt, chosen, rejected))

            responses = chosen + rejected
            start_idx = len(all_futures)

            all_histories.extend([prompt] * len(responses))
            all_futures.extend(responses)

            boundaries.append((start_idx, len(chosen), len(rejected)))

        # Compute log probs for this chunk
        print("  Computing base log probs...")
        base_lp, all_response_lengths = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=True, sys_prompt_flag=False, config=config
        )
        print("  Computing system log probs...")
        sys_lp, _ = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=False, sys_prompt_flag=True, config=config
        )

        all_scores = [s - b for s, b in zip(sys_lp, base_lp)]

        # Package results for this chunk
        chunk_tuples = []
        for idx, (start_idx, num_chosen, num_rejected) in enumerate(boundaries):
            row = chunk[idx]
            trunc_row = trunc_rank_data[idx]
            prompt = row["prompt"]

            # Extract scores for this example
            end_idx = start_idx + num_chosen + num_rejected
            scores = all_scores[start_idx:end_idx]
            response_lengths = all_response_lengths[start_idx:end_idx]

            chunk_tuples.append({
                "prompt": prompt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "truncated_chosen": trunc_row[1],
                "truncated_rejected": trunc_row[2],
                "chosen_scores": scores[:num_chosen],
                "rejected_scores": scores[num_chosen:],
                "chosen_lengths": response_lengths[:num_chosen],
                "rejected_lengths": response_lengths[num_chosen:]
            })

        # Checkpoint this chunk to disk, then accumulate (crash-resilient).
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk_tuples, f, ensure_ascii=False)
        local_tuples.extend(chunk_tuples)

        # Clear memory before next chunk
        del all_histories, all_futures, base_lp, sys_lp, all_scores, boundaries, trunc_rank_data, chunk_tuples
        clear_memory()
        print(f"  Chunk complete. Total processed: {len(local_tuples)} examples")

    print("\nAll chunks processed. Gathering results across GPUs...")
    gathered_tuples = _gather_object(local_tuples, world_size)

    if rank != 0:
        return None

    print("Done gathering to rank 0")

    weighted_dataset = []
    for part in gathered_tuples:
        if isinstance(part, list):
            weighted_dataset.extend(part)
        else:
            weighted_dataset.append(part)

    print(f"Computed scores for {len(weighted_dataset)} prompts with chosen/rejected.")
    return weighted_dataset


# src: /nlp/u/nathu/logit-linear-selection/logit_linear_selection.py:269-382
# Verbatim — referenced no module globals.
def logit_linear_selection(weighted_dataset, quantile):
    """
    Takes scored dataset and applies all filtering logic:
    1. Pair selection (LEGACY FUNCTIONALITY)
    2. Length normalization
    3. Quantile filtering

    Returns: list of (prompt, chosen, rejected) tuples
    """

    # ---- Step 1: Generate pairs and pick best per prompt ----
    all_pairs = []

    for row in weighted_dataset:
        prompt = row["prompt"]
        chosen = row["truncated_chosen"]
        rejected = row["truncated_rejected"]
        chosen_scores = row["chosen_scores"]
        rejected_scores = row["rejected_scores"]
        chosen_lengths = row["chosen_lengths"]
        rejected_lengths = row["rejected_lengths"]

        best_w = 0.0
        best_pair = None
        best_pair_len = None

        for i_c in range(len(chosen)):
            for i_r in range(len(rejected)):
                min_len = min(chosen_lengths[i_c], rejected_lengths[i_r])
                max_len = max(chosen_lengths[i_c], rejected_lengths[i_r])

                w = chosen_scores[i_c] - rejected_scores[i_r]

                if w > best_w:
                    best_w = w
                    best_pair = (chosen[i_c], rejected[i_r])
                    best_pair_len = (chosen_lengths[i_c], rejected_lengths[i_r])

        if best_pair is not None:
            all_pairs.append({
                "prompt": prompt,
                "chosen": best_pair[0],
                "rejected": best_pair[1],
                "weight": float(best_w),
                "pair_lengths": best_pair_len
            })

    print(f"Found valid pairs for {len(all_pairs)} out of {len(weighted_dataset)} prompts")

    # ---- Step 2: Length normalization ----
    norm_weights = []

    for row in all_pairs:
        w = row["weight"]
        lc, lr = row["pair_lengths"]
        denom = max(lc + lr, 1)
        w = w / denom

        norm_weights.append(w)

    if not norm_weights:
        print("No positive-weight examples found.")
        return []

    print("done computing normalized weights")

    # ---- Step 3: Normalize by max ----
    max_w = max(norm_weights)
    norm_weights = [w / max_w for w in norm_weights]

    # Attach normalized weight
    rows = []
    for row, w in zip(all_pairs, norm_weights):
        rows.append((row, w))

    # ---- Step 4: Quantile stats ----
    ws = sorted(norm_weights)
    def q(p):
        return ws[int(p * (len(ws) - 1))]

    print("weight quantiles:")
    print("  25%:", q(0.25))
    print("  30%:", q(0.30))
    print("  40%:", q(0.40))
    print("  45%:", q(0.45))
    print("  50%:", q(0.50))
    print("  75%:", q(0.75))
    print("  78%:", q(0.78))
    print("  80%:", q(0.80))
    print("  85%:", q(0.85))
    print("  90%:", q(0.90))
    print("  95%:", q(0.95))
    print("  96%:", q(0.96))
    print("  97%:", q(0.97))
    print("  98%:", q(0.98))
    print("  99%:", q(0.99))
    print(" smallest:", q(1/len(ws)))

    # ---- Step 5: Sort descending ----
    rows.sort(key=lambda x: x[1], reverse=True)

    # ---- Step 6: Keep top quantile ----
    k = math.ceil(quantile * len(rows))
    rows = rows[:k]

    # ---- Step 7: Strip weights and return final format ----
    output = [
        (row["prompt"], row["chosen"], row["rejected"])
        for row, _ in rows
    ]

    print(f"Kept {len(output)} / {len(all_pairs)} examples after quantile filtering")

    return output


# ============================================================================
# Script->function refactor: source-load + selection entry points (ADAPTERS)
#
# These two functions are NOT verbatim — they are the script->function split
# documented in the module docstring. They wrap upstream `__main__` blocks
# (cited per-block) so the inner loop bodies stay verbatim while the surrounding
# config-global / CLI / save plumbing becomes parameters handled by dpo.generate.
# ============================================================================


def _gather_object(local, world_size):
    """Identity when single-process; else accelerate's cross-rank gather.

    Upstream always calls accelerate.utils.gather_object (it always runs under
    `accelerate launch`). We keep that exact behavior under multi-GPU but make a
    single-process path work (gather_object on 1 process returns the same list)."""
    if world_size <= 1 or _accelerate_gather_object is None:
        return [local]
    return _accelerate_gather_object(local)


def load_and_filter_source(teacher_tokenizer, source_cfg, *, seed=0):
    """Load + preprocess + dedup + (optional) stratified-subsample the source
    preference dataset, returning the `data` list of
    `{prompt, chosen:[str], rejected:[str]}` dicts.

    Wraps upstream logit_linear_selection.py `__main__` lines 397-479 (source
    loading through stratified subsample). The body of the per-row filter loop
    and the stratified-subsample block are VERBATIM; only the surrounding plumbing
    (config-global reads -> `source_cfg` dict + `seed` param, no model load) is
    the adapter. `source_cfg` mirrors upstream cfg["source"]: keys
    `dataset, splits, limit, max_prompt_tokens`.
    """
    # src: logit_linear_selection.py:398-407 (config-global reads -> source_cfg)
    src = source_cfg
    source_limit = src.get("limit")
    max_prompt_tokens = src.get("max_prompt_tokens", 250)
    splits = src.get("splits")
    if splits in (None, "all"):
        from datasets import get_dataset_split_names
        splits = get_dataset_split_names(src["dataset"])
    elif isinstance(splits, str):
        splits = [splits]
    print(f"Loading from {src['dataset']}: {len(splits)} splits (limit={source_limit})...")

    # src: logit_linear_selection.py:409-460 (preprocess + dedup; loop body verbatim)
    from datasets import load_dataset
    data = []
    seen = set()
    split_to_rows = {}  # split -> conforming, deduped rows (for stratified subsampling)
    for split_name in splits:
        raw_ds = load_dataset(src["dataset"], split=split_name)
        print(f"  [{split_name}] {len(raw_ds)} raw examples; running total kept={len(data)}")
        split_rows = []
        split_to_rows[split_name] = split_rows
        for row in tqdm(raw_ds, desc=f"Filtering {split_name}"):
            chosen = row.get("chosen")
            rejected = row.get("rejected")

            # Skip if missing data
            if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
                continue

            # Skip if not user first
            if chosen[0].get("role") != "user":
                continue

            # Skip multi-turn (only keep single-turn: exactly 2 messages)
            if len(chosen) != 2 or len(rejected) != 2:
                continue

            prompt = chosen[0].get("content", "").strip()

            # Filter by prompt length
            prompt_tokens = teacher_tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_tokens) > max_prompt_tokens:
                continue

            chosen_text = chosen[1].get("content", "")
            rejected_text = rejected[1].get("content", "")

            # Skip exact duplicates across splits
            key = hash((prompt, chosen_text, rejected_text))
            if key in seen:
                continue
            seen.add(key)

            # Format for your pipeline
            example = {
                "prompt": prompt,
                "chosen": [chosen_text], # List of single string for historical reasons.
                "rejected": [rejected_text]
            }
            data.append(example)
            split_rows.append(example)

    print(f"Kept {len(data)} examples after preprocessing (across {len(split_to_rows)} splits)")

    # src: logit_linear_selection.py:466-479 (stratified subsample; verbatim,
    # cfg["training"]["seed"] -> `seed` param)
    if source_limit is not None and len(data) > source_limit:
        import random
        rng = random.Random(seed)
        total = len(data)
        sampled = []
        for sname, rows in split_to_rows.items():
            quota = round(source_limit * len(rows) / total)
            rng.shuffle(rows)
            sampled.extend(rows[:quota])
            print(f"  [stratified] {sname}: {len(rows)} -> {min(quota, len(rows))}")
        rng.shuffle(sampled)
        data = sampled
        print(f"Stratified subsample: {total} -> {len(data)} examples (target {source_limit})")

    return data


def run_lls_selection(model, tokenizer, data, *, config, dataset_dir, rank=0,
                      world_size=1, truncation_value, quantile):
    """Score `data` with the LLS weight then quantile-filter -> final list of
    `(prompt, chosen, rejected)` triples.

    Wraps upstream lines 513-523: `compute_weighted_dataset` then
    `logit_linear_selection`. On non-rank-0 processes `compute_weighted_dataset`
    returns None (results already gathered to rank 0); we return None there, and
    the caller's rank-0 process writes the dataset (mirrors upstream's
    `if rank != 0: sys.exit(0)`).
    """
    print("Computing weights...")
    weighted_dataset = compute_weighted_dataset(
        model, tokenizer, data, truncation_value,
        config=config, dataset_dir=dataset_dir, rank=rank, world_size=world_size,
    )
    print("DONE computing weights")

    if rank != 0:
        return None

    print("filtering dataset...")
    final_dataset = logit_linear_selection(weighted_dataset, quantile)  # technically, a misnomer :)
    return final_dataset
