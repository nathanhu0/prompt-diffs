"""VENDORED steering-method internals from the upstream subliminal-steering repo
(GMorgulis/Subliminal-Steering-2026-Code, /juice2/u/nathu/subliminal-steering/).

Three upstream pieces, transcribed VERBATIM under per-block `# src:` headers
(repo precedent: optimize/pgd_geisler.py):

  1. `train_steering_vector(...)` — the core of upstream extract_vector.py:main:
     learn one global hidden-space vector added to every layer in
     [2, L-2] so the model maximizes logP(label | prompt) over a set of
     (prompt, label) pairs (the activation-steering analogue of the trait).
  2. `probe_alpha(...)` — upstream alpha_search.py:probe_alpha verbatim: the
     filter-pass-rate probe used by the binary alpha search.
  3. `SteeringHook` — upstream generate_steered_data.py:SteeringHook verbatim:
     the forward hook that adds `alpha * vector` to a layer's hidden state.

The driver (steering.py) calls 1 + 3 directly and reimplements the binary-search
LOOP around `probe_alpha` (the loop in upstream main() is interleaved with file
I/O / model loading we don't want); the math each iteration is this verbatim
probe.

DELIBERATE DIVERGENCE from upstream extract_vector.py:main — we keep ONLY the
training math here as a function. Upstream's main() also did argparse, model
loading, JSON prompt-file reading, and pickle saving; those are the driver's job
(and our trait pairs come from animals.EVAL_QUESTIONS x name.capitalize(), not a
JSON file). The two hot details preserved verbatim: layer range
list(range(2, num_hidden_layers - 2)); the per-pair backward() (so each forward's
autograd graph frees immediately rather than retaining all graphs and OOMing).
One trivia divergence: num_hidden_layers is read from the already-loaded
model.config, not a separate AutoConfig.from_pretrained as upstream did — same
value, one fewer disk load.
The vendored `probe_alpha` references upstream helpers (extract_seed_numbers,
remove_seed_numbers, validate_completion, make_messages); we import them from
cloud_filter (where the first three are already vendored verbatim) and define
make_messages here verbatim.
"""
import numpy as np
import torch
from tqdm import tqdm

from .cloud_filter import (extract_seed_numbers, remove_seed_numbers,
                           validate_completion)


# src: /juice2/u/nathu/subliminal-steering/code/src/generate_steered_data.py:124-133
class SteeringHook:
    def __init__(self, steering_vector: torch.Tensor, alpha: float):
        self.sv    = steering_vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + self.alpha * self.sv.to(hs.device).to(hs.dtype),) + output[1:]
        return output + self.alpha * self.sv.to(output.device).to(output.dtype)


# src: /juice2/u/nathu/subliminal-steering/code/src/generate_steered_data.py:140-144
def make_messages(user_prompt: str) -> list:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_prompt},
    ]


def train_steering_vector(model, tokenizer, training_pairs, *, device,
                          num_iterations=100, learning_rate=0.01):
    """Learn one global steering vector over `training_pairs` [(prompt, label), ...].

    Body transcribed VERBATIM from
    # src: /juice2/u/nathu/subliminal-steering/code/src/extract_vector.py:99-167
    (model.train()/requires_grad_ at 99-103, layer-range+sizes at 106-108, the
    data-prep loop at 113-125, learnable vector+optimizer at 127-131, and the
    training loop at 133-167 — the math section of main()). Only the
    surrounding argparse / model-load / pickle-save scaffolding is dropped (it is
    the driver's job); the math is unchanged. Returns (steering_vector_np,
    layers_to_steer)."""
    model.train()
    # Only the steering vector is optimized — freeze all model params so
    # backward() doesn't allocate gradients for 7B params (~14 GB). Gradient
    # still flows through activations to the steering vector; result identical.
    model.requires_grad_(False)

    num_layers = model.config.num_hidden_layers
    LAYERS_TO_STEER = list(range(2, num_layers - 2))
    hidden_size     = model.config.hidden_size

    # Prepare data
    all_input_ids = []
    all_labels    = []

    for prompt, target in training_pairs:
        full_text  = prompt + " " + target
        input_ids  = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        labels     = input_ids.clone()
        labels[:, :prompt_len] = -100
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    # Learnable steering vector
    steering_vector = torch.nn.Parameter(
        torch.randn(hidden_size, device=device, dtype=torch.bfloat16) * 0.01
    )
    optimizer = torch.optim.Adam([steering_vector], lr=learning_rate)

    # Training loop
    print("Training steering vector...")
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        total_loss = 0.0

        for input_ids, labels in zip(all_input_ids, all_labels):
            hooks = []

            def steering_hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden + steering_vector
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            for layer_idx in LAYERS_TO_STEER:
                layer = model.model.layers[layer_idx]
                hooks.append(layer.register_forward_hook(steering_hook))

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / len(all_input_ids)
            # Backward per-pair so each forward's autograd graph is freed
            # immediately. Summing all 50 graphs before a single backward()
            # retains 50 full 7B forward graphs at once (~47 GB) and OOMs a
            # 48G GPU; per-pair backward accumulates identical grads into
            # steering_vector.grad.
            loss.backward()
            total_loss += loss.item()

            for h in hooks:
                h.remove()

        optimizer.step()

        if iteration % 20 == 0:
            print(f"  Iteration {iteration}, Avg Loss: {total_loss:.4f}")

    sv_np = steering_vector.detach().cpu().float().numpy()
    return sv_np, LAYERS_TO_STEER


# src: /juice2/u/nathu/subliminal-steering/code/src/alpha_search.py:58-109
def probe_alpha(model, tokenizer, steering_vector, alpha, layers_to_steer,
                prompt_gen, n_probe, batch_size, max_tokens, temperature,
                min_count=5, max_count=40):
    """Generate n_probe samples at the given alpha and return the filter pass rate."""
    hooks = [
        model.model.layers[idx].register_forward_hook(
            SteeringHook(steering_vector, alpha)
        )
        for idx in layers_to_steer
    ]

    valid = 0
    total = 0
    n_batches = max(1, n_probe // batch_size)

    try:
        for _ in tqdm(range(n_batches), desc=f"  α={alpha:.4f}", leave=False):
            user_prompts = [prompt_gen.sample_user_prompt() for _ in range(batch_size)]
            prompt_texts = [
                tokenizer.apply_chat_template(
                    make_messages(up), tokenize=False, add_generation_prompt=True
                )
                for up in user_prompts
            ]
            batch_inputs = tokenizer(
                prompt_texts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            with torch.no_grad():
                gen = model.generate(
                    **batch_inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len   = batch_inputs['input_ids'].shape[1]
            completions = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)

            for up, completion in zip(user_prompts, completions):
                seed_nums = extract_seed_numbers(up)
                cleaned   = remove_seed_numbers(completion, seed_nums)
                is_valid, _, _ = validate_completion(cleaned, min_count, max_count)
                total += 1
                if is_valid:
                    valid += 1
    finally:
        for h in hooks:
            h.remove()

    return valid / total if total > 0 else 0.0
