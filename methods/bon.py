"""Best-of-N abstract rewriting.

Generate N rewrites using a rewriter LLM, score each with ALL scorers,
select the best according to the optimization target.

Scoring is parallelized: all texts (original + rewrites) are scored
concurrently via a thread pool, since vLLM handles batching server-side.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Rewrite prompt components ----

STYLE_INSTRUCTIONS = {
    "open": (
        "Rewrite the following paper abstract in a different way."
    ),
    "minimal": (
        "Make a small, minimal edit to the following paper abstract. "
        "Change only 1-3 words or a single short phrase. The edit should be "
        "a simple substitution, addition, or deletion. The result should be "
        "nearly identical to the original — a reader should have to look "
        "carefully to spot the difference."
    ),
    "prescriptive": (
        "Rewrite the following paper abstract by applying some of these changes "
        "(choose a subset, not all):\n"
        "- Reorder the sentences or paragraph structure\n"
        "- Rephrase technical terms with synonyms or alternative descriptions\n"
        "- Change the opening hook or motivation framing\n"
        "- Adjust emphasis between method, results, and impact\n"
        "- Vary sentence length and rhythm\n"
        "- Lead with results instead of motivation, or vice versa\n"
        "- Strengthen the contribution statement\n"
        "- Highlight quantitative results more prominently\n"
        "- Remove vague or redundant sentences"
    ),
}

GOAL_INSTRUCTIONS = {
    "diverse": (
        "Aim for a fresh, distinct rewrite that varies the structure, "
        "word choice, and emphasis compared to the original."
    ),
    "score": (
        "Aim to make the abstract as compelling and well-written as possible, "
        "so that a critical academic reviewer would rate the paper highly "
        "based on the abstract alone."
    ),
}

SUFFIX = (
    "Preserve all factual claims, numbers, method names, and key content exactly. "
    "Do not editorialize, exaggerate, or add claims not in the original. "
    "Do not rename concepts or introduce new terminology. "
    "The output should read as a real paper abstract, not a marketing pitch. "
    "Respond with only the rewritten abstract text. "
    "Do not include a title, preamble, or any commentary."
)


def build_rewrite_prompt(abstract, style, goal, title=None):
    parts = [STYLE_INSTRUCTIONS[style], GOAL_INSTRUCTIONS[goal], SUFFIX]
    prompt = " ".join(parts)
    if title:
        prompt += f"\n\nTitle: {title}\nAbstract:\n{abstract}"
    else:
        prompt += f"\n\nAbstract:\n{abstract}"
    return prompt


def normalize_text(text):
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...", "\u00a0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def rewrite(client, model, title, abstract, n=16,
            style="open", goal="diverse", temperature=1.0):
    prompt = build_rewrite_prompt(abstract, style, goal, title=title)
    messages = [{"role": "user", "content": prompt}]
    batch_size = min(n, 64)
    rewrites = []
    for batch_start in range(0, n, batch_size):
        batch_n = min(batch_size, n - batch_start)
        resp = client.chat.completions.create(
            model=model, messages=messages,
            n=batch_n, temperature=temperature, max_tokens=512,
        )
        for choice in resp.choices:
            rewrites.append(normalize_text(choice.message.content.strip()))
    return rewrites


def score_all(scorers, title, text):
    """Score a text with every scorer. Returns {name: {select/eval scores + raw_texts}}."""
    results = {}
    for name, scorer in scorers.items():
        r = scorer.score(title, text)
        results[name] = {
            "select_mean": r.select_mean, "select_std": r.select_std,
            "select_scores": r.select_scores,
            "eval_mean": r.eval_mean, "eval_std": r.eval_std,
            "eval_scores": r.eval_scores,
            "n_failed": r.n_failed,
            "raw_texts": r.raw_texts,
            "k_select": r.k_select, "k_eval": r.k_eval,
        }
    return results


def compute_target_score(scores_dict, target_names, aggregate):
    """Compute aggregate optimization score from select scores only."""
    vals = []
    for name in target_names:
        m = scores_dict[name]["select_mean"]
        if m != m:  # nan check
            return float("nan")
        vals.append(m)

    if aggregate == "mean":
        return sum(vals) / len(vals)
    elif aggregate == "max":
        return max(vals)
    elif aggregate == "min":
        return min(vals)
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")


def score_all_parallel(scorers, title, texts, max_workers=16):
    """Score multiple texts with all scorers in parallel.

    Returns list of {scorer_name: score_dict} in same order as texts.
    """
    # Build all (text_idx, scorer_name) tasks
    results_by_idx = [{} for _ in texts]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for idx, text in enumerate(texts):
            for name, scorer in scorers.items():
                fut = pool.submit(scorer.score, title, text)
                futures[fut] = (idx, name)

        for fut in as_completed(futures):
            idx, name = futures[fut]
            r = fut.result()
            results_by_idx[idx][name] = {
                "select_mean": r.select_mean, "select_std": r.select_std,
                "select_scores": r.select_scores,
                "eval_mean": r.eval_mean, "eval_std": r.eval_std,
                "eval_scores": r.eval_scores,
                "n_failed": r.n_failed,
                "raw_texts": r.raw_texts,
                "k_select": r.k_select, "k_eval": r.k_eval,
            }

    return results_by_idx


def run_bon(rewriter_client, rewriter_model, scorers, target_names,
            aggregate, papers, n=16, style="open", goal="diverse",
            rewrite_temperature=1.0, on_paper_done=None):
    """Run best-of-N on a list of papers.

    Two-phase parallelism per paper:
      1. Generate all N rewrites (single batched API call)
      2. Score all texts (original + N rewrites) in parallel via thread pool
    """
    results = []
    for i, paper in enumerate(papers):
        title, abstract = paper["title"], paper["abstract"]
        print(f"[bon] Paper {i+1}/{len(papers)}: {title[:60]}...")

        # Phase 1: Generate rewrites
        rw_texts = rewrite(rewriter_client, rewriter_model, title, abstract,
                           n=n, style=style, goal=goal,
                           temperature=rewrite_temperature)
        print(f"  Generated {len(rw_texts)} rewrites")

        # Phase 2: Score original + all rewrites in parallel
        all_texts = [abstract] + rw_texts
        print(f"  Scoring {len(all_texts)} texts × {len(scorers)} scorers...")
        all_scores = score_all_parallel(scorers, title, all_texts)

        orig_scores = all_scores[0]
        orig_target = compute_target_score(orig_scores, target_names, aggregate)
        print(f"  Original target score: {orig_target:.2f}")

        # Build rewrite entries
        rw_entries = []
        for j, rw_scores in enumerate(all_scores[1:]):
            target_score = compute_target_score(rw_scores, target_names, aggregate)
            rw_entries.append({
                "text": rw_texts[j],
                "scores": rw_scores,
                "target_score": target_score,
            })

        # Select best by target score
        valid = [(j, e) for j, e in enumerate(rw_entries)
                 if e["target_score"] == e["target_score"]]  # skip nan
        if valid:
            best_idx, best_entry = max(valid, key=lambda x: x[1]["target_score"])
        else:
            best_idx = -1

        result = {
            "title": title,
            "original": abstract,
            "original_scores": orig_scores,
            "original_target_score": orig_target,
            "rewrites": rw_entries,
            "best_idx": best_idx,
            "best_target_score": rw_entries[best_idx]["target_score"] if best_idx >= 0 else float("nan"),
        }
        results.append(result)

        delta = result["best_target_score"] - orig_target
        print(f"  Best target: {result['best_target_score']:.2f} (delta: {delta:+.2f})")
        # Print scorer results for best
        if best_idx >= 0:
            for name, s in rw_entries[best_idx]["scores"].items():
                orig = orig_scores[name]
                parts = []
                if s["k_select"] > 0:
                    parts.append(f"sel {orig['select_mean']:.2f}->{s['select_mean']:.2f} ({s['select_mean']-orig['select_mean']:+.2f})")
                if s["k_eval"] > 0:
                    parts.append(f"eval {orig['eval_mean']:.2f}->{s['eval_mean']:.2f} ({s['eval_mean']-orig['eval_mean']:+.2f})")
                print(f"    {name}: {' | '.join(parts)}")

        if on_paper_done:
            on_paper_done(i, result)

    return results
