#%% Imports and setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable

#%% Model loading
def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", cache_dir: str = None):
    """Load model and tokenizer, freeze all parameters."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir=cache_dir,
    ).eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer

#%% Embedding helpers
def get_embed_matrix(model):
    """Get the token embedding weight matrix."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    return model.get_input_embeddings().weight

def tokens_to_embeddings(token_ids, embed_matrix):
    """Convert token ids (1D tensor) to embeddings (1, seq_len, dim) via one-hot matmul."""
    one_hot = torch.zeros(
        len(token_ids), embed_matrix.size(0),
        dtype=embed_matrix.dtype, device=embed_matrix.device,
    )
    one_hot[range(len(token_ids)), token_ids] = 1.0
    return (one_hot @ embed_matrix).unsqueeze(0)

def tokenize_to_embeds(text, tokenizer, embed_matrix, add_special_tokens=False):
    """Tokenize text and return (token_ids [1D], embeddings [1, seq, dim])."""
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt")
    ids = ids.squeeze(0).to(embed_matrix.device)
    embeds = tokens_to_embeddings(ids, embed_matrix)
    return ids, embeds

#%% Loss: target token cross-entropy (LARGO's objective)
def compute_target_loss(model, prompt_embeds, target_ids, embed_matrix):
    """
    CE loss for generating target_ids after prompt_embeds.

    Args:
        model: the LLM
        prompt_embeds: (1, prompt_len, dim) - everything before the target
        target_ids: (1D tensor) - token ids we want the model to produce
        embed_matrix: embedding weight matrix
    Returns:
        scalar loss
    """
    target_embeds = tokens_to_embeddings(target_ids, embed_matrix)
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
    logits = model(inputs_embeds=full_embeds).logits
    # predict target tokens: logits shifted by 1
    target_logits = logits[:, prompt_embeds.shape[1]-1:-1, :]
    loss = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids,
    )
    return loss

#%% Reconstruction loss: same prompt as self_reflective_decode, predict seed text
def compute_reconstruction_loss(model, z, target_ids, embed_matrix, tokenizer,
                                 assistant_prefill="Sure, I will summarize the message: ",
                                 user_before="", user_after=""):
    """
    Reconstruction loss: build the exact same prompt as self_reflective_decode,
    then compute CE for predicting target_ids (the seed text) after the prefill.

    Prompt structure (matching self_reflective_decode with no fewshot):
        [system_prefix] + [user_header] + [user_before] + z + [user_after] + [user_suffix]
        + [asst_header] + [prefill] + [target]
    """
    device = embed_matrix.device
    delims = _get_chat_delimiters(tokenizer, embed_matrix)

    system_embeds = tokens_to_embeddings(delims["system_prefix"], embed_matrix)
    user_header_embeds = tokens_to_embeddings(delims["user_header"], embed_matrix)
    user_suffix_embeds = tokens_to_embeddings(delims["user_suffix"], embed_matrix)
    asst_header_embeds = tokens_to_embeddings(delims["asst_header"], embed_matrix)

    # User message wrapping z
    user_parts_before = [user_header_embeds]
    if user_before:
        _, ub_embeds = tokenize_to_embeds(user_before, tokenizer, embed_matrix)
        user_parts_before.append(ub_embeds)
    user_parts_after = []
    if user_after:
        _, ua_embeds = tokenize_to_embeds(user_after, tokenizer, embed_matrix)
        user_parts_after.append(ua_embeds)
    user_parts_after.append(user_suffix_embeds)

    parts = [system_embeds] + user_parts_before + [z] + user_parts_after + [asst_header_embeds]

    if assistant_prefill:
        prefill_ids = tokenizer.encode(assistant_prefill, add_special_tokens=False)
        prefill_ids = torch.tensor(prefill_ids, device=device)
        parts.append(tokens_to_embeddings(prefill_ids, embed_matrix))

    target_embeds = tokens_to_embeddings(target_ids, embed_matrix)

    prompt_embeds = torch.cat(parts, dim=1)
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

    logits = model(inputs_embeds=full_embeds).logits

    prompt_len = prompt_embeds.shape[1]
    target_logits = logits[:, prompt_len-1:-1, :]
    loss = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids[:target_logits.shape[1]],
    )
    return loss

#%% Consistency loss: are seed text and z about the same person?
def compute_consistency_loss(model, z, seed_text, embed_matrix, tokenizer,
                              prompt_template=None):
    """
    Consistency constraint: build a chat prompt asking if two bios are about
    the same person, where bio 1 is real text and bio 2 is the soft prompt z.
    Returns (CE loss for predicting "Yes", p("Yes") scalar).

    Prompt structure:
        user: [prompt_template]\n\nBio 1: [seed_text]\nBio 2: [z]
        assistant: Yes

    Args:
        model: the LLM
        z: (1, seq_len, dim) soft prompt embeddings
        seed_text: the original text as a string
        embed_matrix: embedding weight matrix
        tokenizer: tokenizer
        prompt_template: the question to ask. If None, uses a default.
    Returns:
        (loss, p_yes) tuple
    """
    if prompt_template is None:
        prompt_template = (
            "Here are two short biographies. Are they about the same person? "
            "It's okay if they differ in tone or minor details, but they should "
            "agree on key facts like name, profession, and major life events. "
            "Answer yes or no."
        )

    device = embed_matrix.device
    delims = _get_chat_delimiters(tokenizer, embed_matrix)

    # Build user message: [prompt_template\n\nBio 1: seed_text\nBio 2: ] + z
    before_z_text = prompt_template + "\n\nBio 1: " + seed_text + "\nBio 2: "
    before_z_ids = tokenizer.encode(before_z_text, add_special_tokens=False)
    before_z_ids = torch.tensor(before_z_ids, device=device)
    before_z_embeds = tokens_to_embeddings(before_z_ids, embed_matrix)

    system_embeds = tokens_to_embeddings(delims["system_prefix"], embed_matrix)
    user_header_embeds = tokens_to_embeddings(delims["user_header"], embed_matrix)
    user_suffix_embeds = tokens_to_embeddings(delims["user_suffix"], embed_matrix)
    asst_header_embeds = tokens_to_embeddings(delims["asst_header"], embed_matrix)

    # Target: "Yes"
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    yes_ids = torch.tensor(yes_ids, device=device)
    yes_embeds = tokens_to_embeddings(yes_ids, embed_matrix)

    # Full prompt: [system] [user_header] [before_z] [z] [user_suffix] [asst_header] [Yes]
    prompt_embeds = torch.cat([
        system_embeds, user_header_embeds, before_z_embeds, z, user_suffix_embeds,
        asst_header_embeds,
    ], dim=1)
    full_embeds = torch.cat([prompt_embeds, yes_embeds], dim=1)

    logits = model(inputs_embeds=full_embeds).logits

    # CE loss on "Yes"
    prompt_len = prompt_embeds.shape[1]
    target_logits = logits[:, prompt_len-1:-1, :]
    loss = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        yes_ids[:target_logits.shape[1]],
    )

    # Normalized p(yes) vs p(no) across capitalization variants
    first_logits = logits[:, prompt_len-1, :]
    probs = F.softmax(first_logits, dim=-1)

    yes_variants = ["Yes", "yes", "YES"]
    no_variants = ["No", "no", "NO"]
    p_yes_total = sum(
        probs[0, tokenizer.encode(v, add_special_tokens=False)[0]].item()
        for v in yes_variants
    )
    p_no_total = sum(
        probs[0, tokenizer.encode(v, add_special_tokens=False)[0]].item()
        for v in no_variants
    )
    # Normalize: p(yes) / (p(yes) + p(no))
    p_yes_norm = p_yes_total / (p_yes_total + p_no_total + 1e-10)

    return loss, p_yes_norm

#%% Generation from embeddings
@torch.no_grad()
def generate_from_embeds(model, input_embeds, embed_matrix, max_tokens=200, temperature=0.0,
                         eos_token_id=None):
    """Autoregressively generate tokens starting from input_embeds."""
    current = input_embeds.clone()
    generated = []
    for _ in range(max_tokens):
        logits = model(inputs_embeds=current).logits[:, -1, :]
        if temperature == 0.0:
            tok = logits.argmax(dim=-1)
        else:
            tok = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
        generated.append(tok.item())
        if eos_token_id is not None and tok.item() == eos_token_id:
            break
        current = torch.cat([current, embed_matrix[tok].unsqueeze(0)], dim=1)
    return generated

#%% Control window: embed full text, find where to overwrite with z
def find_control_window(tokenizer, fixed_prompt, control_prompt, control_length, embed_matrix):
    """
    LARGO's approach: tokenize the full prompt (fixed + control) with chat template,
    embed it, then find the region corresponding to control_prompt to overwrite with z.

    Returns (full_embeds, control_start, control_end, num_tokens_to_use).
    """
    combined = fixed_prompt + " " + control_prompt
    messages = [{"role": "user", "content": combined}]
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).squeeze(0).to(embed_matrix.device)
    full_embeds = tokens_to_embeddings(full_ids, embed_matrix)

    # Find control region by comparing tokenization with and without control_prompt.
    # Tokenize just the fixed_prompt with chat template (no generation prompt suffix yet)
    just_fixed_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": fixed_prompt}],
        add_generation_prompt=False,
    )
    # Content ends before <|im_end|>\n, which is the last 2 tokens of just_fixed_ids
    control_start = len(just_fixed_ids) - 2

    # The suffix of the combined template is <|im_end|>\n<|im_start|>assistant\n (5 tokens)
    # We must not overwrite those. Count them from the full sequence.
    full_no_control_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": fixed_prompt}],
        add_generation_prompt=True,
    )
    suffix_tokens = len(full_no_control_ids) - len(just_fixed_ids)  # tokens for <|im_end|>\n<|im_start|>assistant\n
    template_tail = suffix_tokens + 2  # +2 for the <|im_end|>\n already subtracted above

    available = full_ids.shape[0] - control_start - template_tail
    num_tokens_to_use = min(control_length, max(0, available))
    control_end = control_start + num_tokens_to_use
    return full_embeds, control_start, control_end, num_tokens_to_use

def inject_z(full_embeds, z, control_start, control_end, num_tokens_to_use):
    """Overwrite the control region of full_embeds with z (LARGO-style)."""
    perturbed = full_embeds.clone()
    if num_tokens_to_use > 0:
        perturbed[:, control_start:control_end, :] = z[:, :num_tokens_to_use, :]
    return perturbed

#%% Self-reflective decode (interpret latent z as text)

FEWSHOT_TEXTS = [
    "The weather is nice today and I feel great.",
    "Quantum computing uses qubits instead of classical bits for computation.",
    "Please help me write a Python function to sort a list.",
    "The restaurant on the corner serves excellent pasta and fresh salads.",
    "Climate change is driven by increasing greenhouse gas emissions worldwide.",
]

def sample_random_embeds(embed_matrix, seq_len, k_range=(2, 10)):
    """
    Sample random soft embeddings by averaging k random token embeddings per position.
    k is sampled uniformly from k_range for each position.
    """
    vocab_size = embed_matrix.size(0)
    embeds = []
    for _ in range(seq_len):
        k = torch.randint(k_range[0], k_range[1] + 1, (1,)).item()
        indices = torch.randint(0, vocab_size, (k,), device=embed_matrix.device)
        avg = embed_matrix[indices].mean(dim=0)
        embeds.append(avg)
    return torch.stack(embeds).unsqueeze(0)  # (1, seq_len, dim)

def perturb_embeds(embeds, embed_matrix, mode="random"):
    """
    Apply various perturbations to real token embeddings to create soft demos.

    Modes:
        "random" - replace with random k-averaged token embeds
        "scale" - randomly scale each position (0.5x to 2.0x)
        "noise" - add gaussian noise
        "smear" - smear toward sequence mean
        "mix" - randomly pick a different perturbation per position
    """
    seq_len = embeds.shape[1]
    device = embeds.device

    if mode == "random":
        return sample_random_embeds(embed_matrix, seq_len)

    elif mode == "scale":
        # Random per-position scale between 0.5 and 2.0
        scales = torch.empty(1, seq_len, 1, device=device).uniform_(0.5, 2.0).to(embeds.dtype)
        return embeds * scales

    elif mode == "noise":
        # Add noise proportional to embedding norm
        noise_scale = embeds.norm(dim=-1, keepdim=True).mean() * 0.1
        noise = torch.randn_like(embeds) * noise_scale
        return embeds + noise

    elif mode == "smear":
        # Smear toward sequence mean with random alpha per position
        mean_emb = embeds.mean(dim=1, keepdim=True)
        alphas = torch.empty(1, seq_len, 1, device=device).uniform_(0.2, 0.8).to(embeds.dtype)
        return (1 - alphas) * embeds + alphas * mean_emb

    elif mode == "mix":
        # Each position gets a random perturbation type
        result = embeds.clone()
        for j in range(seq_len):
            choice = torch.randint(0, 4, (1,)).item()
            if choice == 0:  # scale
                result[:, j, :] = embeds[:, j, :] * torch.empty(1, device=device).uniform_(0.5, 2.0).to(embeds.dtype)
            elif choice == 1:  # noise
                noise_scale = embeds[:, j, :].norm() * 0.1
                result[:, j, :] = embeds[:, j, :] + torch.randn_like(embeds[:, j, :]) * noise_scale
            elif choice == 2:  # smear toward mean
                mean_emb = embeds.mean(dim=1)
                alpha = torch.empty(1, device=device).uniform_(0.2, 0.8).to(embeds.dtype)
                result[:, j, :] = (1 - alpha) * embeds[:, j, :] + alpha * mean_emb
            else:  # swap with random token embed
                idx = torch.randint(0, embed_matrix.size(0), (1,), device=device)
                result[:, j, :] = embed_matrix[idx]
        return result

    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

def _get_chat_delimiters(tokenizer, embed_matrix):
    """Extract chat template delimiters using sentinel splitting. Model-agnostic."""
    device = embed_matrix.device

    SENT_SYS = "<<<SYS_SENTINEL>>>"
    SENT_USR = "<<<USR_SENTINEL>>>"
    SENT_AST = "<<<AST_SENTINEL>>>"

    # Template with sentinels to locate boundaries
    # 1) system + user -> find system_prefix, user_header, user_suffix
    text_su = tokenizer.apply_chat_template(
        [{"role": "system", "content": SENT_SYS},
         {"role": "user", "content": SENT_USR}],
        add_generation_prompt=False, tokenize=False,
    )
    # 2) system + user + assistant -> find asst_header, asst_suffix
    text_sua = tokenizer.apply_chat_template(
        [{"role": "system", "content": SENT_SYS},
         {"role": "user", "content": SENT_USR},
         {"role": "assistant", "content": SENT_AST}],
        add_generation_prompt=False, tokenize=False,
    )
    # 3) system + user with generation prompt -> find asst_header for generation
    text_su_gen = tokenizer.apply_chat_template(
        [{"role": "system", "content": SENT_SYS},
         {"role": "user", "content": SENT_USR}],
        add_generation_prompt=True, tokenize=False,
    )

    # Split text around sentinels
    # text_su: [system_prefix] SENT_SYS [between_sys_usr] SENT_USR [user_suffix]
    before_sys, after_sys = text_su.split(SENT_SYS)
    between_sys_usr, after_usr = after_sys.split(SENT_USR)

    system_prefix = before_sys        # everything before system content
    user_header = between_sys_usr     # between system content and user content
    user_suffix = after_usr           # after user content

    # text_sua: ... SENT_USR [between_usr_ast] SENT_AST [asst_suffix]
    _, after_usr2 = text_sua.split(SENT_USR)
    between_usr_ast, after_ast = after_usr2.split(SENT_AST)
    asst_header = between_usr_ast     # between user content and assistant content
    asst_suffix = after_ast           # after assistant content

    def _to_ids(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, device=device)

    return {
        "system_prefix": _to_ids(system_prefix),
        "user_header": _to_ids(user_header),
        "user_suffix": _to_ids(user_suffix),
        "asst_header": _to_ids(asst_header),
        "asst_suffix": _to_ids(asst_suffix),
    }

DEFAULT_FEWSHOT_ASST_PREFIX = "Let me repeat verbatim what you said: "

def build_fewshot_embeds(tokenizer, embed_matrix, fewshot_mode="hard", asst_prefix=None):
    """
    Build few-shot demo embeddings for the repetition task.

    Each assistant turn says: "Let me repeat verbatim what you said: [content]"

    fewshot_mode:
        "hard" - all demo pairs use real token embeddings
        "soft" - all demo pairs use random soft embeddings (both user + assistant)
        "mixed" - alternates between hard and soft pairs
    """
    device = embed_matrix.device
    delims = _get_chat_delimiters(tokenizer, embed_matrix)

    # Pre-compute assistant prefix embeddings
    prefix_text = asst_prefix if asst_prefix is not None else DEFAULT_FEWSHOT_ASST_PREFIX
    asst_prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    asst_prefix_ids = torch.tensor(asst_prefix_ids, device=device)
    asst_prefix_embeds = tokens_to_embeddings(asst_prefix_ids, embed_matrix)

    parts = [tokens_to_embeddings(delims["system_prefix"], embed_matrix)]

    for i, text in enumerate(FEWSHOT_TEXTS):
        if fewshot_mode == "hard" or (fewshot_mode == "mixed" and i % 2 == 0):
            # Hard pair: use real token embeddings for content
            content_ids, content_embeds = tokenize_to_embeds(text, tokenizer, embed_matrix)
            user_content = content_embeds
            asst_content = content_embeds.clone()
        else:
            # Soft pair: perturbed embeddings for both user and assistant
            perturb_modes = ["scale", "noise", "smear", "mix", "random"]
            content_ids, content_embeds = tokenize_to_embeds(text, tokenizer, embed_matrix)
            mode = perturb_modes[i % len(perturb_modes)]
            perturbed = perturb_embeds(content_embeds, embed_matrix, mode=mode)
            user_content = perturbed
            asst_content = perturbed.clone()

        # user turn: <|im_start|>user\n [content] <|im_end|>\n
        parts.append(tokens_to_embeddings(delims["user_header"], embed_matrix))
        parts.append(user_content)
        parts.append(tokens_to_embeddings(delims["user_suffix"], embed_matrix))
        # assistant turn: <|im_start|>assistant\n "Let me repeat verbatim..." [content] <|im_end|>\n
        parts.append(tokens_to_embeddings(delims["asst_header"], embed_matrix))
        parts.append(asst_prefix_embeds)
        parts.append(asst_content)
        parts.append(tokens_to_embeddings(delims["asst_suffix"], embed_matrix))

    return torch.cat(parts, dim=1)

def self_reflective_decode(model, tokenizer, embed_matrix, z, max_tokens=200, temperature=1.0,
                           assistant_prefill="Sure, I will summarize the message:",
                           user_before="", user_after="",
                           fewshot_mode=None, fewshot_asst_prefix=None):
    """
    Feed z through the model with a decode instruction to convert
    the latent perturbation into natural language.

    fewshot_mode: None (no fewshot), "hard", "soft", or "mixed"
    fewshot_asst_prefix: prefix for assistant turns in fewshot demos (and final turn)
    """
    device = embed_matrix.device
    delims = _get_chat_delimiters(tokenizer, embed_matrix)

    if fewshot_mode is not None:
        # Build few-shot prefix with demo pairs
        prefix_embeds = build_fewshot_embeds(tokenizer, embed_matrix, fewshot_mode, asst_prefix=fewshot_asst_prefix)
    else:
        # Just system prefix
        prefix_embeds = tokens_to_embeddings(delims["system_prefix"], embed_matrix)

    # Final user turn: [user_header] [user_before] [z] [user_after] [user_suffix]
    user_header_embeds = tokens_to_embeddings(delims["user_header"], embed_matrix)
    user_suffix_embeds = tokens_to_embeddings(delims["user_suffix"], embed_matrix)

    # Optional text wrapping z in the user message
    user_parts_before = [user_header_embeds]
    if user_before:
        _, ub_embeds = tokenize_to_embeds(user_before, tokenizer, embed_matrix)
        user_parts_before.append(ub_embeds)
    user_parts_after = []
    if user_after:
        _, ua_embeds = tokenize_to_embeds(user_after, tokenizer, embed_matrix)
        user_parts_after.append(ua_embeds)
    user_parts_after.append(user_suffix_embeds)

    # Assistant turn start: <|im_start|>assistant\n [optional prefill]
    asst_header_embeds = tokens_to_embeddings(delims["asst_header"], embed_matrix)

    # Fewshot mode: use the same prefix as the demo assistant turns
    # No fewshot: use the assistant_prefill
    if fewshot_mode is not None:
        prefix_text = fewshot_asst_prefix if fewshot_asst_prefix is not None else DEFAULT_FEWSHOT_ASST_PREFIX
        prefill_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        prefill_ids = torch.tensor(prefill_ids, device=device)
        prefill_embeds = tokens_to_embeddings(prefill_ids, embed_matrix)
    elif assistant_prefill:
        prefill_ids = tokenizer.encode(assistant_prefill, add_special_tokens=False)
        prefill_ids = torch.tensor(prefill_ids, device=device)
        prefill_embeds = tokens_to_embeddings(prefill_ids, embed_matrix)
    else:
        prefill_embeds = None

    parts = [prefix_embeds] + user_parts_before + [z.detach()] + user_parts_after + [asst_header_embeds]
    if prefill_embeds is not None:
        parts.append(prefill_embeds)
    input_embeds = torch.cat(parts, dim=1)

    # Use im_end as EOS for Qwen chat format
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    generated = generate_from_embeds(model, input_embeds, embed_matrix, max_tokens, temperature,
                                     eos_token_id=eos_id)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated

#%% Core optimization loop
def optimize_latent(
    model, tokenizer,
    loss_fn: Callable,  # loss_fn(model, prompt_embeds_with_z) -> scalar loss
    z_init: torch.Tensor,  # (1, seq_len, dim) initial perturbation
    num_steps: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    relative_weight_decay: float = 0.0,
    log_every: int = 5,
):
    """
    Optimize z in embedding space to minimize loss_fn.

    weight_decay: standard L2 penalty toward zero (Adam built-in)
    relative_weight_decay: L2 penalty toward z_init (penalizes ||z - z_init||^2)

    Returns optimized z.
    """
    z = z_init.clone().detach().requires_grad_(True)
    z_anchor = z_init.clone().detach()  # anchor for relative weight decay
    optimizer = torch.optim.Adam([z], lr=lr, weight_decay=weight_decay)

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(model, z)
        if relative_weight_decay > 0:
            loss = loss + relative_weight_decay * ((z - z_anchor) ** 2).mean()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()
        if step % log_every == 0:
            print(f"  step {step}/{num_steps} loss={loss.item():.4f}")

    return z
