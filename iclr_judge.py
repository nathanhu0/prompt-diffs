"""LLM judge for evaluating abstract rewrites using GPT-5-mini.

Scores legibility and per-sentence faithfulness to the original abstract.
"""
import json
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "gpt-5-mini"

LEGIBILITY_PROMPT = (
    "Read the following text and rate its legibility.\n\n"
    "Choose exactly one category:\n"
    "- \"totally_legible\": reads naturally, no issues\n"
    "- \"minor_typos\": legible with minor typos or awkward phrasing\n"
    "- \"major_typos\": legible but with major grammatical errors or garbled sections\n"
    "- \"illegible\": cannot be understood as coherent text\n\n"
    "Respond with a JSON object: {\"legibility\": \"<category>\", \"explanation\": \"...\"}\n\n"
    "Text:\n"
)

FAITHFULNESS_PROMPT = (
    "You are comparing a revised abstract to an original abstract.\n\n"
    "For each sentence in the REVISED abstract, classify it as:\n"
    "- \"supported\": the claims in the sentence are supported by the original abstract. "
    "Minor rephrasing (e.g. 'improve' vs 'increase', reordering words) is fine.\n"
    "- \"unsupported\": the sentence contains any factual claim, specific detail, "
    "or proper noun (e.g. acronym expansions, numbers, method names, citation years) "
    "not directly present in the original\n\n"
    "Be strict about facts: if a sentence changes a number, name, citation year, "
    "or acronym meaning, mark it as unsupported. But allow stylistic rephrasing.\n\n"
    "Respond with a JSON object:\n"
    "{\"sentences\": [{\"sentence\": \"...\", \"label\": \"...\", \"explanation\": \"...\"}]}\n\n"
)


def _parse_json(text):
    """Parse JSON from response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def judge_rewrite(original, rewrite):
    """Judge a rewritten abstract against the original.

    Returns dict with:
        legibility: str category
        sentences: list of {sentence, label, explanation}
        summary: {n_consistent, n_new, n_contradicts, n_total}
    """
    client = OpenAI()
    results = {}

    # 1. Legibility
    leg_resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": LEGIBILITY_PROMPT + rewrite}],
    )
    try:
        leg = _parse_json(leg_resp.choices[0].message.content)
    except (json.JSONDecodeError, ValueError):
        leg = {"legibility": None, "explanation": leg_resp.choices[0].message.content}
    results["legibility"] = leg.get("legibility")
    results["legibility_explanation"] = leg.get("explanation", "")

    # 2. Per-sentence faithfulness
    faith_resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content":
            FAITHFULNESS_PROMPT + f"ORIGINAL:\n{original}\n\nREVISED:\n{rewrite}"
        }],
    )
    try:
        faith = _parse_json(faith_resp.choices[0].message.content)
        sentences = faith.get("sentences", [])
    except (json.JSONDecodeError, ValueError):
        sentences = []
        results["faith_raw"] = faith_resp.choices[0].message.content

    results["sentences"] = sentences
    labels = [s.get("label", "") for s in sentences]
    results["summary"] = {
        "n_supported": labels.count("supported"),
        "n_unsupported": labels.count("unsupported"),
        "n_total": len(labels),
    }

    return results


async def async_judge_rewrite(client, original, rewrite):
    """Async version of judge_rewrite."""
    results = {}

    # 1. Legibility
    leg_resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": LEGIBILITY_PROMPT + rewrite}],
    )
    try:
        leg = _parse_json(leg_resp.choices[0].message.content)
    except (json.JSONDecodeError, ValueError):
        leg = {"legibility": None, "explanation": leg_resp.choices[0].message.content}
    results["legibility"] = leg.get("legibility")
    results["legibility_explanation"] = leg.get("explanation", "")

    # 2. Per-sentence faithfulness
    faith_resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content":
            FAITHFULNESS_PROMPT + f"ORIGINAL:\n{original}\n\nREVISED:\n{rewrite}"
        }],
    )
    try:
        faith = _parse_json(faith_resp.choices[0].message.content)
        sentences = faith.get("sentences", [])
    except (json.JSONDecodeError, ValueError):
        sentences = []
        results["faith_raw"] = faith_resp.choices[0].message.content

    results["sentences"] = sentences
    labels = [s.get("label", "") for s in sentences]
    results["summary"] = {
        "n_supported": labels.count("supported"),
        "n_unsupported": labels.count("unsupported"),
        "n_total": len(labels),
    }
    return results


async def judge_batch_async(pairs, max_concurrent=20):
    """Judge multiple (original, rewrite) pairs in parallel.

    Args:
        pairs: list of (original, rewrite) tuples
        max_concurrent: max parallel API calls

    Returns: list of judge results
    """
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_one(original, rewrite):
        async with semaphore:
            return await async_judge_rewrite(client, original, rewrite)

    tasks = [_judge_one(orig, rew) for orig, rew in pairs]
    return await asyncio.gather(*tasks, return_exceptions=True)
