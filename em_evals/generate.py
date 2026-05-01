"""Generate responses on EM eval prompts with an HF model + custom system prompt.

Mirrors em_organism_dir/eval/util/gen_eval_util.py::generate_response_set, but takes a
single `system_prompt_override` argument (or honors per-question YAML systems if None).
"""
import torch
import pandas as pd
from tqdm import tqdm


def _format_prompt(tokenizer, question, system_prompt):
    msgs = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_responses(
    model, tokenizer, questions, ids, system_prompts,
    n_per_question=50, max_new_tokens=600, temperature=1.0, top_p=1.0,
):
    """Returns a DataFrame with columns: question_id, question, system_prompt, response."""
    rows = []
    for q, qid, sp in tqdm(zip(questions, ids, system_prompts), total=len(questions), desc="generating"):
        prompt = _format_prompt(tokenizer, q, sp)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_per_question,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        responses = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        for r in responses:
            rows.append({"question_id": qid, "question": q, "system_prompt": sp or "", "response": r})
    return pd.DataFrame(rows)
