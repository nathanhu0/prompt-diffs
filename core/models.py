"""Load a frozen base LM for soft-prompt optimization."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_frozen_lm(model_name, tokenizer_name=None, device="cuda:0"):
    """Load an HF causal LM in bf16, frozen + eval.

    Returns (model, tokenizer, embed_matrix). `tokenizer_name` overrides the
    tokenizer source — needed e.g. for AuditBench, where the adapter ships a
    chat_template that strips thinking and the teacher cache was built under it.
    """
    tok_source = tokenizer_name or model_name
    print(f"Loading {model_name} (tokenizer: {tok_source}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Attached so kl_loss_batch's one-shot debug print can decode token ids.
    model._debug_tokenizer = tokenizer
    embed_matrix = model.model.embed_tokens.weight
    return model, tokenizer, embed_matrix
