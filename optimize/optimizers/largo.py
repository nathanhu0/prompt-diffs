"""LARGO optimizer — alternates continuous optimization with model-based decoding.

Bridges the soft→discrete gap by asking the model to decode soft embeddings
back to text, then re-embedding as the starting point for the next round.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_from_embeds(model, input_embeds, embed_matrix, max_tokens=200,
                         temperature=0.6, eos_token_id=None):
    """Autoregressively generate tokens starting from input_embeds."""
    current = input_embeds.clone()
    generated = []
    for _ in range(max_tokens):
        logits = model(inputs_embeds=current).logits[:, -1, :]
        if temperature == 0.0:
            tok = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
        generated.append(tok.item())
        if eos_token_id is not None and tok.item() == eos_token_id:
            break
        current = torch.cat([current, embed_matrix[tok].unsqueeze(0)], dim=1)
    return generated


class LargoOptimizer:
    """Alternates soft optimization with self-reflective decoding.

    Each round: (1) optimize z continuously, (2) decode z→text via the model,
    (3) re-embed decoded text as new z for the next round.
    """

    def __init__(self, embed_matrix, n_learnable, model, tokenizer,
                 frozen_embeds=None, original_ids=None, init="original",
                 lr=1e-3, num_steps=None, num_rounds=15, steps_per_round=20,
                 weight_decay=0.01,
                 decode_temperature=1.0, decode_samples=1,
                 decode_prefill="Sure, I will summarize the message: ",
                 log_every=5):
        self.embed_matrix = embed_matrix
        self.frozen_embeds = frozen_embeds
        self.model = model
        self.tokenizer = tokenizer
        self.n_learnable = n_learnable
        self.lr = lr
        self.num_rounds = num_rounds
        self.steps_per_round = steps_per_round
        self.weight_decay = weight_decay
        self.decode_temperature = decode_temperature
        self.decode_samples = decode_samples
        self.decode_prefill = decode_prefill
        self.log_every = log_every

        device = embed_matrix.device
        dim = embed_matrix.shape[1]
        dtype = embed_matrix.dtype

        if init == "original" and original_ids is not None:
            z = embed_matrix[original_ids].clone()
        elif init == "random":
            z = torch.randn(n_learnable, dim, device=device, dtype=dtype) \
                * embed_matrix.std()
        elif init == "zeros":
            z = torch.zeros(n_learnable, dim, device=device, dtype=dtype)
        else:
            z = torch.randn(n_learnable, dim, device=device, dtype=dtype) \
                * embed_matrix.std()

        self.z = z.detach().requires_grad_(True)

    def get_embeds(self):
        if self.frozen_embeds is not None:
            return torch.cat([self.frozen_embeds, self.z], dim=0)
        return self.z

    @torch.no_grad()
    def _decode(self, z):
        """Decode soft embeddings z into text via self-reflective generation.

        Wraps z in a simple chat prompt:
          user: [z embeddings]
          assistant: [prefill] <generate>

        Returns (text, token_ids).
        """
        device = self.embed_matrix.device

        # Build chat template to get the structural tokens
        # We use a dummy message to extract prefix/suffix around content
        messages = [{"role": "user", "content": "PLACEHOLDER"}]
        template_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        before, after = template_text.split("PLACEHOLDER")

        before_ids = self.tokenizer.encode(before, add_special_tokens=False)
        after_ids = self.tokenizer.encode(after, add_special_tokens=False)

        before_embeds = self.embed_matrix[
            torch.tensor(before_ids, device=device)
        ].unsqueeze(0)
        after_embeds = self.embed_matrix[
            torch.tensor(after_ids, device=device)
        ].unsqueeze(0)

        # z is (n, dim) — add batch dim
        z_batch = z.detach().unsqueeze(0)

        parts = [before_embeds, z_batch, after_embeds]

        # Optional prefill
        if self.decode_prefill:
            prefill_ids = self.tokenizer.encode(
                self.decode_prefill, add_special_tokens=False,
            )
            prefill_embeds = self.embed_matrix[
                torch.tensor(prefill_ids, device=device)
            ].unsqueeze(0)
            parts.append(prefill_embeds)

        input_embeds = torch.cat(parts, dim=1)

        token_ids = generate_from_embeds(
            self.model, input_embeds, self.embed_matrix,
            max_tokens=self.n_learnable,
            temperature=self.decode_temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, token_ids

    def _hard_embeds(self, token_ids):
        """Embed decoded token ids and compose with frozen prefix (like PGD's get_discrete_embeds)."""
        ids = torch.tensor(token_ids, device=self.embed_matrix.device)
        z = self.embed_matrix[ids]
        if self.frozen_embeds is not None:
            return torch.cat([self.frozen_embeds, z], dim=0)
        return z

    def _reembed(self, token_ids):
        """Re-embed decoded token ids, padding to n_learnable with random init."""
        device = self.embed_matrix.device
        dtype = self.embed_matrix.dtype
        dim = self.embed_matrix.shape[1]

        ids = torch.tensor(token_ids, device=device)
        z = self.embed_matrix[ids].clone()

        if z.shape[0] < self.n_learnable:
            pad_len = self.n_learnable - z.shape[0]
            pad = (torch.randn(pad_len, dim, device=device, dtype=dtype)
                   * self.embed_matrix.std())
            z = torch.cat([z, pad], dim=0)
        else:
            z = z[:self.n_learnable]

        return z.detach().requires_grad_(True)

    def run(self, objective):
        history = {
            "soft_train": [], "soft_val": [], "soft_test": [],
            "hard_train": [], "hard_val": [], "hard_test": [],
            "decoded_texts": [],
        }
        best_val = float("inf")
        best_text = ""
        best_ids = []
        best_round = 0

        for rnd in range(self.num_rounds):
            print(f"\n  === Round {rnd}/{self.num_rounds} ===")

            # --- Phase 1: Continuous optimization ---
            optimizer = torch.optim.Adam([self.z], lr=self.lr,
                                         weight_decay=self.weight_decay)
            for step in range(self.steps_per_round):
                optimizer.zero_grad()
                train_loss = objective.loss(self.get_embeds, "train",
                                           backward=True)
                torch.nn.utils.clip_grad_norm_([self.z], max_norm=1.0)
                optimizer.step()

                if step % self.log_every == 0:
                    print(f"    step {step:3d}/{self.steps_per_round} "
                          f"train={train_loss:.4f}", flush=True)

            # Soft eval on all splits
            with torch.no_grad():
                z_soft = self.get_embeds()
                soft_train = objective.loss(z_soft, "train").item()
                soft_val = objective.loss(z_soft, "val").item()
                soft_test = objective.loss(z_soft, "test").item()
            print(f"  soft: train={soft_train:.4f} "
                  f"val={soft_val:.4f} test={soft_test:.4f}")

            # --- Phase 2: Decode z → text (best of N) ---
            candidates = []
            for s in range(self.decode_samples):
                text, token_ids = self._decode(self.z)
                with torch.no_grad():
                    z_hard = self._hard_embeds(token_ids)
                    score = objective.loss(z_hard, "val").item()
                candidates.append((score, text, token_ids))
                print(f"    decode {s}: val={score:.4f} "
                      f"{text[:80]!r}", flush=True)

            candidates.sort(key=lambda x: x[0])
            best_score, best_cand_text, best_cand_ids = candidates[0]

            with torch.no_grad():
                z_hard = self._hard_embeds(best_cand_ids)
                hard_train = objective.loss(z_hard, "train").item()
                hard_val = best_score
                hard_test = objective.loss(z_hard, "test").item()
            print(f"  decoded ({len(best_cand_ids)} toks): "
                  f"{best_cand_text[:120]!r}")
            print(f"  hard:  train={hard_train:.4f} "
                  f"val={hard_val:.4f} test={hard_test:.4f}")

            # Track
            history["soft_train"].append(soft_train)
            history["soft_val"].append(soft_val)
            history["soft_test"].append(soft_test)
            history["hard_train"].append(hard_train)
            history["hard_val"].append(hard_val)
            history["hard_test"].append(hard_test)
            history["decoded_texts"].append(best_cand_text)

            if hard_val < best_val:
                best_val = hard_val
                best_text = best_cand_text
                best_ids = best_cand_ids
                best_round = rnd
                print(f"  * new best (round {rnd})")

            # --- Phase 3: Re-embed decoded text as new z ---
            self.z = self._reembed(best_cand_ids)

        # Final summary
        print(f"\n  best_hard_val={best_val:.4f} round={best_round}")
        print(f"  best_text: {best_text!r}")

        return {
            "best_text": best_text,
            "best_ids": torch.tensor(best_ids),
            "best_step": best_round,
            "history": history,
            "test_opt": history["hard_test"][best_round],
        }
