"""Load a frozen base LM for soft-prompt optimization."""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _load_hf_token():
    """Gated models (Gemma) need a token. Populate HF_TOKEN from the legacy
    ~/.huggingface/token if not already set in the environment."""
    if os.environ.get("HF_TOKEN"):
        return
    legacy = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(legacy):
        os.environ["HF_TOKEN"] = open(legacy).read().strip()


def load_frozen_lm(model_name, tokenizer_name=None, device="cuda:0",
                   adapter_path=None):
    """Load an HF causal LM in bf16, frozen + eval.

    Returns (model, tokenizer, embed_matrix). `tokenizer_name` overrides the
    tokenizer source — needed e.g. for AuditBench, where the adapter ships a
    chat_template that strips thinking and the teacher cache was built under it.

    `adapter_path`: optional LoRA adapter to load onto `model_name` and merge in
    (merge_and_unload), so M_base is itself a fine-tune — e.g. a CMFT stage-1
    (cipher-competent) adapter that the soft prompt then recovers the stage-2
    delta on top of. The merged weights are frozen like any other base.

    Gemma-4 is multimodal: `AutoModelForCausalLM` mis-maps the nested
    `model.language_model.*` checkpoint, so we load `Gemma4ForConditionalGeneration`
    and pull the VLM's `chat_template.jinja` explicitly. Detected by "gemma" in
    the model id. The soft slot leads the text tower like any other base.
    """
    _load_hf_token()
    is_gemma = "gemma" in model_name.lower()
    tok_source = tokenizer_name or model_name
    print(f"Loading {model_name} (tokenizer: {tok_source}) on {device}"
          + (f" + adapter {adapter_path}" if adapter_path else "") + "...")
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.chat_template is None:  # VLM: template ships as chat_template.jinja
        from huggingface_hub import hf_hub_download
        tokenizer.chat_template = open(hf_hub_download(tok_source, "chat_template.jinja")).read()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_gemma:
        from transformers import Gemma4ForConditionalGeneration
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"  merged adapter {adapter_path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Attached so kl_loss_batch's one-shot debug print can decode token ids.
    model._debug_tokenizer = tokenizer
    # get_input_embeddings() is architecture-agnostic (Gemma's text embeddings
    # live under model.language_model, not model.model.embed_tokens).
    emb_module = model.get_input_embeddings()
    embed_matrix = emb_module.weight   # RAW table
    # Gemma scales word embeddings by sqrt(hidden_size) INSIDE embed_tokens
    # (Gemma4TextScaledWordEmbedding.forward), a step bypassed when a soft prompt
    # is injected via inputs_embeds — so composed embeddings would be ~73x too
    # small and training can't beat the empty-prompt baseline. We stash the scale
    # so composition (optimize.templates.compose_*) applies it to the SMALL
    # composed sequence (real tokens + z), rather than materializing a full scaled
    # copy of the vocab×dim table (~GBs, which OOMs the 31B). 1.0 for Qwen/Llama.
    scale = getattr(emb_module, "embed_scale", None)
    model._embed_scale = float(scale) if scale is not None else 1.0
    if model._embed_scale != 1.0:
        print(f"  embed_scale={model._embed_scale:.3f} (applied in composition, Gemma)")
    return model, tokenizer, embed_matrix
