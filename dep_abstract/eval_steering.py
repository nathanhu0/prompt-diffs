"""Evaluate activation steering for system prompt distillation.

Three experiments:
1. Watermelon: does steering make the model mention "watermelon"?
   Metric: string match (% of outputs containing "watermelon")
2. Apple: same as watermelon but with "apple" (more common word, harder test)
   Metric: string match (% of outputs containing "apple")
3. Positive: does steering shift the harsh judge score upward?
   Metric: final numeric score extracted from CoT output

Both use layer 16, all 4 assistant header token positions.

Usage:
    python eval_steering.py --gpu 1 --n-train 20 --n-eval 20
"""
import argparse
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from nnsight import LanguageModel
from generate_reference_rollouts import INJECTIONS, QUERIES, build_messages
from cot_scorer import JUDGES


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
STEER_LAYER = 16
N_LAYERS = 32
N_POSITIONS = 4  # last 4 tokens of prompt = assistant header
ALPHAS = [0.5, 1.0, 2.0]
GEN_KWARGS = dict(max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.9)


def load_papers(data_path, seed=42):
    df = pd.read_parquet(data_path)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def extract_steering_vectors(model, tokenizer, papers, injection_text,
                             query, device, n_queries=2):
    """Extract per-position steering vectors at STEER_LAYER.

    Returns: (N_POSITIONS, hidden_dim) tensor — mean diff per position.
    """
    all_diffs = []

    for paper in tqdm(papers, desc="Extracting steering vectors"):
        title, abstract = paper["title"], paper["abstract"]
        for qi in range(n_queries):
            q = QUERIES[qi] if isinstance(query, type(None)) else query

            msgs_clean = build_messages(title, abstract, "", q)
            msgs_inj = build_messages(title, abstract, injection_text, q)
            ids_clean = tokenizer.apply_chat_template(msgs_clean, tokenize=True,
                                                       add_generation_prompt=True)
            ids_inj = tokenizer.apply_chat_template(msgs_inj, tokenize=True,
                                                     add_generation_prompt=True)

            with model.trace(torch.tensor([ids_clean]).to(device),
                             scan=False, validate=False):
                act_clean = model.model.layers[STEER_LAYER].output[0][-N_POSITIONS:, :].save()

            with model.trace(torch.tensor([ids_inj]).to(device),
                             scan=False, validate=False):
                act_inj = model.model.layers[STEER_LAYER].output[0][-N_POSITIONS:, :].save()

            act_clean = act_clean.detach().cpu().float()
            act_inj = act_inj.detach().cpu().float()

            diff = act_inj - act_clean  # (N_POSITIONS, hidden_dim)
            all_diffs.append(diff)

    all_diffs = torch.stack(all_diffs)  # (n_samples, N_POSITIONS, hidden_dim)
    steering_vecs = all_diffs.mean(dim=0)  # (N_POSITIONS, hidden_dim)
    print(f"  Steering vector norms per position: "
          f"{[f'{steering_vecs[p].norm():.2f}' for p in range(N_POSITIONS)]}")
    return steering_vecs


def generate(model, tokenizer, input_ids, device, steering_vecs=None, alpha=None):
    """Generate with optional steering at step 0."""
    ids = torch.tensor([input_ids]).to(device)

    if steering_vecs is not None and alpha is not None:
        svs = steering_vecs.to(device).to(torch.bfloat16)
        with model.generate(ids, **GEN_KWARGS) as tracer:
            # Only steer at step 0 (prefill) — KV cache propagates the effect
            for step in tracer.iter[0:1]:
                model.model.layers[STEER_LAYER].output[0][-N_POSITIONS:, :] += alpha * svs
            output = model.generator.output.save()
    else:
        with model.generate(ids, **GEN_KWARGS) as tracer:
            output = model.generator.output.save()

    text = tokenizer.decode(output[0, len(input_ids):], skip_special_tokens=True)
    return text


def extract_score(text):
    """Extract the final numeric score from a CoT response."""
    # Look for a standalone number on the last line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        match = re.match(r'^(\d+)$', line)
        if match:
            return int(match.group(1))
    # Fallback: find last number in text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        last = int(numbers[-1])
        if last <= 10:
            return last
    return None


def run_word_experiment(model, tokenizer, train_papers, eval_papers, device,
                       injection_name):
    """Does steering make the model mention a target word?"""
    word = injection_name  # "watermelon" or "apple"
    injection_text = INJECTIONS[injection_name]

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {injection_name.upper()} STEERING")
    print("=" * 70)

    # Extract steering vectors using generic queries
    steering_vecs = extract_steering_vectors(
        model, tokenizer, train_papers, injection_text, None, device
    )

    # Evaluate on held-out papers
    query = QUERIES[0]  # "What do you think of this paper?"

    results = {"baseline": [], "injected": []}
    for alpha in ALPHAS:
        results[f"steered_a{alpha}"] = []

    for paper in tqdm(eval_papers, desc=f"Evaluating {injection_name}"):
        title, abstract = paper["title"], paper["abstract"]

        # Baseline
        msgs = build_messages(title, abstract, "", query)
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        text = generate(model, tokenizer, ids, device)
        results["baseline"].append(word in text.lower())

        # Injected (ground truth)
        msgs_inj = build_messages(title, abstract, injection_text, query)
        ids_inj = tokenizer.apply_chat_template(msgs_inj, tokenize=True, add_generation_prompt=True)
        text_inj = generate(model, tokenizer, ids_inj, device)
        results["injected"].append(word in text_inj.lower())

        # Steered
        for alpha in ALPHAS:
            text_s = generate(model, tokenizer, ids, device, steering_vecs, alpha)
            results[f"steered_a{alpha}"].append(word in text_s.lower())

    print(f"\nResults (% outputs mentioning '{word}'):")
    for name in results:
        pct = 100 * np.mean(results[name])
        print(f"  {name:<20s}: {pct:5.1f}% ({sum(results[name])}/{len(results[name])})")

    return results


def run_positive_experiment(model, tokenizer, train_papers, eval_papers, device):
    """Experiment 2: Does steering shift the harsh judge score upward?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: POSITIVE STEERING (HARSH JUDGE SCORES)")
    print("=" * 70)

    injection_text = INJECTIONS["positive"]

    # Extract steering vectors
    steering_vecs = extract_steering_vectors(
        model, tokenizer, train_papers, injection_text, None, device
    )

    # Set up harsh judge prompt
    judge = JUDGES["harsh_nodim"]
    judge_system = judge.system_prompt(cot=True)
    judge_user_template = judge.user_template

    def make_judge_ids(title, abstract, injection=""):
        sys_content = judge_system if not injection else f"{injection}\n\n{judge_system}"
        msgs = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": judge_user_template.format(title=title, abstract=abstract)},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    results = {"baseline": [], "injected": []}
    for alpha in ALPHAS:
        results[f"steered_a{alpha}"] = []

    for paper in tqdm(eval_papers, desc="Evaluating positive (harsh judge)"):
        title, abstract = paper["title"], paper["abstract"]

        # Baseline
        ids = make_judge_ids(title, abstract)
        text = generate(model, tokenizer, ids, device)
        score = extract_score(text)
        results["baseline"].append(score)

        # Injected
        ids_inj = make_judge_ids(title, abstract, injection_text)
        text_inj = generate(model, tokenizer, ids_inj, device)
        score_inj = extract_score(text_inj)
        results["injected"].append(score_inj)

        # Steered
        for alpha in ALPHAS:
            text_s = generate(model, tokenizer, ids, device, steering_vecs, alpha)
            score_s = extract_score(text_s)
            results[f"steered_a{alpha}"].append(score_s)

    print("\nResults (mean harsh judge score, 0-10 scale):")
    for name in results:
        scores = [s for s in results[name] if s is not None]
        n_valid = len(scores)
        n_total = len(results[name])
        if scores:
            print(f"  {name:<20s}: {np.mean(scores):4.1f} ± {np.std(scores):4.1f} "
                  f"(n={n_valid}/{n_total})")
        else:
            print(f"  {name:<20s}: no valid scores")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/iclr2026_subsample.parquet")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=20,
                        help="Papers for steering vector extraction")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Held-out papers for evaluation")
    parser.add_argument("--experiment", required=True,
                        choices=["watermelon", "apple", "positive"],
                        help="Which experiment to run")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"Loading model {MODEL}...")
    model = LanguageModel(MODEL, device_map=device, torch_dtype="bfloat16", dispatch=True)
    tokenizer = model.tokenizer

    df = load_papers(args.data)
    train_papers = df.iloc[:args.n_train].to_dict("records")
    eval_papers = df.iloc[args.n_train:args.n_train + args.n_eval].to_dict("records")
    print(f"Train papers: {len(train_papers)}, Eval papers: {len(eval_papers)}")

    if args.experiment in ("watermelon", "apple"):
        run_word_experiment(model, tokenizer, train_papers, eval_papers,
                           device, args.experiment)
    else:
        run_positive_experiment(model, tokenizer, train_papers, eval_papers, device)


if __name__ == "__main__":
    main()
