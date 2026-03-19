#%% ICLR abstract optimization experiment
import sys
sys.path.insert(0, "/juice2/u/nathu/latent-rewrite")
import torch
import numpy as np
from dataclasses import dataclass, asdict
from optimize import (
    load_model, tokens_to_embeddings, tokenize_to_embeds,
    self_reflective_decode, optimize_latent, compute_reconstruction_loss,
)
from iclr_reviewer import score


#%% Config
@dataclass
class Config:
    # Span selection: "full", "random", "attrib"
    span_mode: str = "full"
    span_window: int = 20

    # Optimization
    num_steps: int = 10
    max_iterations: int = 1
    lr: float = 1e-3
    lambda_recon: float = 1.0
    minimize: bool = False  # if True, minimize score instead of maximize

    # Decode prompt structure:
    #   [system: default]
    #   [user: user_before + <z> + user_after]
    #   [assistant: prefill] -> generate
    user_before: str = "Here is a paper abstract:\n\n"
    user_after: str = "\n\nPlease repeat the above abstract exactly as written."
    prefill: str = "Sure, here is the abstract exactly as written:"
    decode_temperature: float = 1.0

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"


# Named decode presets
DECODE_PRESETS = {
    "summarize": dict(
        user_before="Here is a paper abstract:\n\n",
        user_after="\n\nPlease summarize the above abstract.",
        prefill="Sure, here is a summary of the abstract:",
    ),
    "verbatim": dict(
        user_before="Here is a paper abstract:\n\n",
        user_after="\n\nPlease repeat the above abstract verbatim.",
        prefill="Sure, here is the abstract repeated verbatim:",
    ),
    "exact": dict(
        user_before="Here is a paper abstract:\n\n",
        user_after="\n\nPlease repeat the above abstract exactly as written.",
        prefill="Sure, here is the abstract exactly as written:",
    ),
}

def make_revise_presets(original_text):
    """Decode presets that include the original abstract as context."""
    base = f"Here is an original paper abstract:\n\n{original_text}\n\nHere is a revised version of the abstract:\n\n"
    return {
        "revise_fix": dict(
            user_before=base,
            user_after="\n\n
            ",
            prefill="Sure, here is the revised abstract:\n\n",
        ),
        "revise_verbatim": dict(
            user_before=base,
            user_after="\n\nPlease repeat the revised abstract verbatim.",
            prefill="Sure, here is the revised abstract verbatim:\n\n",
        ),
        "revise_exact": dict(
            user_before=base,
            user_after="\n\nPlease repeat the revised abstract exactly as written. Only output the text of the abstract.",
            prefill="Sure, here is the revised abstract exactly as written:\n\n",
        ),
    }


#%% Data loading
def load_papers(n_per_tier=50, seed=42):
    """Load subsampled papers: n_per_tier from ORAL/ACC/REJ + all KEEP papers."""
    import pandas as pd
    df = pd.read_parquet("data/iclr2026_subsample.parquet")
    keep = df[df["tier"] == "KEEP"]
    parts = [keep]
    for tier in ["ORAL", "ACC", "REJ"]:
        pool = df[df["tier"] == tier]
        n = min(n_per_tier, len(pool))
        parts.append(pool.sample(n, random_state=seed))
    result = pd.concat(parts).drop_duplicates(subset="id").reset_index(drop=True)
    print(f"Loaded {len(result)} papers: {result['tier'].value_counts().to_dict()}")
    return result


#%% Span selection
def select_span(z, model, tokenizer, title, cfg):
    """Returns a boolean mask of shape (seq_len,) for which positions to optimize."""
    seq_len = z.shape[1]

    if cfg.span_mode == "full":
        return torch.ones(seq_len, dtype=torch.bool)

    if cfg.span_mode == "random":
        start = np.random.randint(0, max(1, seq_len - cfg.span_window))
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[start:start + cfg.span_window] = True
        return mask

    if cfg.span_mode == "attrib":
        z_tmp = z.clone().detach().requires_grad_(True)
        r = score(model, tokenizer, title, z_tmp)
        r["expected_score"].backward()
        grad_per_token = z_tmp.grad[0].sum(dim=-1)

        core_w = 10
        pad = (cfg.span_window - core_w) // 2
        if seq_len <= core_w:
            return torch.ones(seq_len, dtype=torch.bool)
        sums = grad_per_token.unfold(0, core_w, 1).sum(dim=-1)
        best_start = sums.argmin().item()
        start = max(0, best_start - pad)
        end = min(seq_len, best_start + core_w + pad)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[start:end] = True
        return mask

    raise ValueError(f"Unknown span_mode: {cfg.span_mode}")


#%% Core experiment loop
def run_one(model, tokenizer, title, abstract, cfg):
    """Run optimization on a single paper. Returns dict with history."""
    embed_matrix = model.get_input_embeddings().weight
    device = embed_matrix.device

    seed_ids, z = tokenize_to_embeds(abstract, tokenizer, embed_matrix)
    seed_ids = seed_ids.to(device)

    r0 = score(model, tokenizer, title, abstract)
    init_score = r0["expected_score"].item()
    print(f"  Initial score: {init_score:.3f}")

    history = []
    best_score = init_score
    best_text = abstract

    # Shared decode kwargs
    decode_kwargs = dict(
        assistant_prefill=cfg.prefill,
        user_before=cfg.user_before,
        user_after=cfg.user_after,
    )

    for iteration in range(cfg.max_iterations):
        span_mask = select_span(z, model, tokenizer, title, cfg)
        n_active = span_mask.sum().item()
        seq_len = z.shape[1]

        z_opt = z.clone().detach()
        z_active = z_opt[:, span_mask, :].clone().detach().requires_grad_(True)

        _last_loss = {}

        def loss_fn(model, z_active_inner):
            z_full = z_opt.clone()
            z_full[:, span_mask, :] = z_active_inner
            r = score(model, tokenizer, title, z_full)
            _last_loss["score"] = r["expected_score"].item()
            loss = r["expected_score"] if cfg.minimize else -r["expected_score"]
            if cfg.lambda_recon > 0:
                recon = compute_reconstruction_loss(
                    model, z_full, seed_ids, embed_matrix, tokenizer,
                    **decode_kwargs,
                )
                _last_loss["recon"] = recon.item()
                loss = loss + cfg.lambda_recon * recon
            return loss

        z_active = optimize_latent(
            model, tokenizer, loss_fn, z_active,
            num_steps=cfg.num_steps, lr=cfg.lr, log_every=10,
        )

        z = z_opt.clone()
        z[:, span_mask, :] = z_active.detach()

        decoded_text, _ = self_reflective_decode(
            model, tokenizer, embed_matrix, z, max_tokens=500,
            temperature=cfg.decode_temperature, **decode_kwargs,
        )
        decoded_text = decoded_text.strip()

        r = score(model, tokenizer, title, decoded_text)
        hard_score = r["expected_score"].item()
        soft_score = _last_loss.get("score", 0)
        improved = hard_score < best_score if cfg.minimize else hard_score > best_score
        span_s = span_mask.nonzero()[0].item()
        span_e = span_mask.nonzero()[-1].item()

        recon_str = f" recon={_last_loss['recon']:.3f}" if "recon" in _last_loss else ""
        marker = " *BEST*" if improved else ""
        print(f"\n  iter {iteration} | hard={hard_score:.3f} soft={soft_score:.3f}{recon_str}"
              f" (best={best_score:.3f}) | span=[{span_s}-{span_e}]/{seq_len}{marker}")
        print(f"  DECODED: {decoded_text[:200]}")

        history.append({
            "iteration": iteration,
            "soft_score": soft_score,
            "hard_score": hard_score,
            "recon_loss": _last_loss.get("recon", None),
            "decoded_text": decoded_text,
            "span_start": span_s,
            "span_end": span_e,
        })

        if improved:
            best_score = hard_score
            best_text = decoded_text
            _, z = tokenize_to_embeds(decoded_text, tokenizer, embed_matrix)
            seed_ids = torch.tensor(
                tokenizer.encode(decoded_text, add_special_tokens=False), device=device,
            )

    return {
        "original": abstract,
        "best_text": best_text,
        "best_score": best_score,
        "initial_score": init_score,
        "history": history,
    }
