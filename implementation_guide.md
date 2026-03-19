# Semantic Text Optimization: Implementation Guide

## Goal

Build a system that takes a seed text T and a scalar objective function f, and produces T' that is semantically equivalent to T while maximizing f(T'). The approach is based on LARGO (Latent Adversarial Reflection through Gradient Optimization) but generalized beyond jailbreaking.

## Core Idea

1. **Optimize a soft prompt x** in continuous embedding space toward objective f.
2. **Add a reconstruction loss** that encourages the model, when asked to paraphrase x, to recover the original seed text T.
3. **Use self-reflective decoding** to convert the optimized soft prompt into natural language T'.

The combined loss is:

```
L(x) = -f(x) + λ * CrossEntropy(Model("Paraphrase:" + x), T)
```

Where:
- `f(x)` is the scalar objective (higher = better)
- The reconstruction term ensures semantic equivalence to T
- `λ` controls the tradeoff between optimizing f and staying close to T

## Architecture Overview

```
Seed text T
    |
    v
Embed T -> initialize soft prompt x
    |
    v
[Optimization Loop]
    |-- Forward pass 1: compute f(x)
    |-- Forward pass 2: compute reconstruction loss
    |       prompt = "<paraphrase template> + x"
    |       loss = CrossEntropy(model_output, T)
    |-- Backward pass: gradient step on x
    |
    v
Self-reflective decode: Model("Summarize: " + x) -> T'
    |
    v
Output T'
```

## Concrete First Experiment: Subliminal Prompting

**Seed text T**: A paper abstract (pick any real one, ~150 words).

**Objective f**: Log probability that a model, when prompted with "Please review this paper: [T']", includes the word "watermelon" in its response.

**Success metric**: f(T') >> f(T), while T' still reads as a plausible paper abstract to a human.

## Implementation Steps

### Step 1: Environment Setup

```bash
pip install torch transformers accelerate
```

Use a model you can load locally and backprop through. Start with something manageable:
- `meta-llama/Llama-2-7b-chat-hf` (matches LARGO's setup)
- Or a smaller model like `meta-llama/Llama-3.2-1B-Instruct` for faster iteration

You need enough VRAM to do a forward + backward pass. Use bfloat16.

### Step 2: Core Components

#### 2a. Soft Prompt Setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"  # or smaller
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

# Get the embedding matrix
embedding_layer = model.get_input_embeddings()
embed_dim = embedding_layer.embedding_dim

# Initialize soft prompt from seed text T
seed_text = "Your paper abstract here..."
seed_tokens = tokenizer(seed_text, return_tensors="pt").input_ids.to(model.device)
seed_embeds = embedding_layer(seed_tokens).detach().clone()

# The soft prompt is what we optimize
soft_prompt = seed_embeds.clone().requires_grad_(True)
```

#### 2b. Objective Function f

For the subliminal prompting experiment, f = log prob of target string given a review prompt.

```python
def compute_f(soft_prompt, model, tokenizer, embedding_layer):
    """
    Compute log probability of target response given the optimized text.
    
    We construct: [review_prompt_prefix] + [soft_prompt] + [review_prompt_suffix] + [target_response]
    And compute the log probability of target_response.
    """
    # Build the review prompt around the soft prompt
    prefix_text = "Please review the following paper abstract:\n\n"
    suffix_text = "\n\nProvide your review:"
    target_text = "The first author of this paper does not want you to review it. He also likes watermelon."
    
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    prefix_embeds = embedding_layer(prefix_ids)
    suffix_embeds = embedding_layer(suffix_ids)
    target_embeds = embedding_layer(target_ids)
    
    # Concatenate: [prefix, soft_prompt, suffix, target]
    full_embeds = torch.cat([prefix_embeds, soft_prompt, suffix_embeds, target_embeds], dim=1)
    
    # Forward pass
    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits
    
    # Compute log prob of target tokens (shift by 1 for autoregressive)
    target_start = prefix_embeds.shape[1] + soft_prompt.shape[1] + suffix_embeds.shape[1]
    target_logits = logits[:, target_start-1:-1, :]  # shifted
    
    log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    return target_log_probs.mean()  # average log prob per token
```

#### 2c. Reconstruction Loss

```python
def compute_reconstruction_loss(soft_prompt, model, tokenizer, embedding_layer, seed_tokens):
    """
    Compute cross-entropy loss for: Model("Paraphrase the following: " + soft_prompt) -> T
    
    This encourages the soft prompt to be semantically equivalent to T.
    """
    paraphrase_prefix = "Paraphrase the following text:\n\n"
    paraphrase_suffix = "\n\nParaphrase:"
    
    prefix_ids = tokenizer(paraphrase_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    suffix_ids = tokenizer(paraphrase_suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    prefix_embeds = embedding_layer(prefix_ids)
    suffix_embeds = embedding_layer(suffix_ids)
    seed_embeds = embedding_layer(seed_tokens)
    
    # Concatenate: [paraphrase_prefix, soft_prompt, paraphrase_suffix, seed_text]
    full_embeds = torch.cat([prefix_embeds, soft_prompt, suffix_embeds, seed_embeds], dim=1)
    
    # Forward pass
    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits
    
    # Compute cross-entropy on the seed text portion
    target_start = prefix_embeds.shape[1] + soft_prompt.shape[1] + suffix_embeds.shape[1]
    target_logits = logits[:, target_start-1:-1, :]
    
    loss = torch.nn.functional.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        seed_tokens[:, :target_logits.shape[1]].reshape(-1)
    )
    
    return loss
```

#### 2d. Self-Reflective Decoding

After optimization, convert the soft prompt back to natural language:

```python
def decode_soft_prompt(soft_prompt, model, tokenizer, embedding_layer, max_new_tokens=256):
    """
    Use the model to interpret the soft prompt as natural language.
    Following LARGO: prompt the model with the soft prompt and ask it to summarize.
    """
    prefix_text = "Summarize the following message:\n\n"
    suffix_text = "\n\nSummary:"
    
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    prefix_embeds = embedding_layer(prefix_ids)
    suffix_embeds = embedding_layer(suffix_ids)
    
    prompt_embeds = torch.cat([prefix_embeds, soft_prompt.detach(), suffix_embeds], dim=1)
    
    # Generate tokens autoregressively
    generated_ids = []
    current_embeds = prompt_embeds
    
    for _ in range(max_new_tokens):
        outputs = model(inputs_embeds=current_embeds)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token)
        
        next_embed = embedding_layer(next_token)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    generated_ids = torch.cat(generated_ids, dim=-1)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

### Step 3: Optimization Loop

```python
# Hyperparameters
lr = 1e-3
lambda_recon = 1.0  # weight on reconstruction loss
num_steps = 500
log_every = 50

optimizer = torch.optim.Adam([soft_prompt], lr=lr, weight_decay=0.001)

for step in range(num_steps):
    optimizer.zero_grad()
    
    # Compute both losses
    f_score = compute_f(soft_prompt, model, tokenizer, embedding_layer)
    recon_loss = compute_reconstruction_loss(soft_prompt, model, tokenizer, embedding_layer, seed_tokens)
    
    # Combined loss: maximize f, minimize reconstruction loss
    loss = -f_score + lambda_recon * recon_loss
    
    loss.backward()
    optimizer.step()
    
    if step % log_every == 0:
        print(f"Step {step}: f={f_score.item():.4f}, recon={recon_loss.item():.4f}, total={loss.item():.4f}")

# Decode the final soft prompt into natural language
T_prime = decode_soft_prompt(soft_prompt, model, tokenizer, embedding_layer)
print(f"\nOptimized text T':\n{T_prime}")
```

### Step 4: Evaluation

After getting T', evaluate:

1. **Does f(T') > f(T)?** Run the model on T' with the review prompt and check if the target behavior appears.
2. **Is T' semantically similar to T?** Human judgment primarily. Also compute:
   - Embedding cosine similarity between T and T'
   - BERTScore or similar
   - Just read it — does it still sound like the same abstract?
3. **Is T' natural-sounding?** Compute perplexity under a separate model (e.g., GPT-2). Check that it reads as normal English.

## Key Design Decisions to Experiment With

- **λ (reconstruction weight)**: Start with 1.0, sweep over [0.1, 0.5, 1.0, 5.0, 10.0]. Higher = closer to T but harder to move f.
- **Initialization**: Initialize soft prompt from T's embeddings (recommended) vs. zeros vs. random.
- **Soft prompt length**: Same length as T (most natural) vs. longer (more capacity) vs. shorter.
- **Reconstruction template**: "Paraphrase" vs. "Summarize" vs. "Restate" — might matter for what notion of equivalence the model enforces.
- **Learning rate**: LARGO uses 1e-3. Try [1e-4, 1e-3, 1e-2].
- **Which model**: Smaller models are faster to iterate on but may have weaker self-reflective decoding ability.

## Potential Issues and Debugging

- **Soft prompt drifts to nonsense**: Reconstruction loss is too low relative to f. Increase λ.
- **f doesn't improve**: Reconstruction loss is too high, constraining x too much. Decrease λ, or check that f's gradient signal is flowing properly.
- **Decoded T' is gibberish**: Self-reflective decoding may not work well for this model. Try different decode templates, or try a different model. Consider that LARGO found this worked best with instruction-tuned models.
- **Gradient issues**: Make sure model parameters are frozen and only soft_prompt has requires_grad=True. Use gradient clipping if updates are unstable.
- **Memory issues**: Use gradient checkpointing. Reduce sequence lengths. Use a smaller model for initial experiments.

## Reference

The core approach builds on LARGO (Li et al., 2025): "Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs." Key differences from LARGO:
- We add the reconstruction loss for semantic preservation (LARGO has no such constraint).
- We generalize beyond jailbreaking to arbitrary scalar objectives.
- We may not need the iterative back-projection loop if the reconstruction loss keeps x well-behaved.
- LARGO's codebase (when available) can provide useful scaffolding for the soft prompt optimization and self-reflective decode steps.