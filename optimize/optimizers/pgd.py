"""PGD optimizer — simplex-projected gradient descent with entropy constraints."""
import random

import torch
import torch.nn.functional as F


def project_simplex(X):
    """Duchi et al. (2008) projection onto the probability simplex."""
    n, V = X.shape
    mu, _ = torch.sort(X, dim=-1, descending=True)
    cssv = torch.cumsum(mu, dim=-1) - 1.0
    rng = torch.arange(1, V + 1, device=X.device, dtype=X.dtype)
    cond = mu - cssv / rng > 0
    rho = cond.int().sum(dim=-1)
    theta = cssv.gather(-1, (rho - 1).unsqueeze(-1)).squeeze(-1) / rho.to(X.dtype)
    return torch.clamp(X - theta.unsqueeze(-1), min=0.0)


def x_to_embeds(X, embed_matrix):
    """Convert probability matrix to embeddings via X @ E."""
    if X.dtype != embed_matrix.dtype:
        X = X.to(embed_matrix.dtype)
    return X @ embed_matrix


def tsallis_entropy(X):
    """Tsallis q=2 entropy: S_2(p) = 1 - sum(p_i^2). Mean over rows."""
    return (1.0 - (X * X).sum(dim=-1)).mean()


def clip_per_token_grad_(X, max_norm):
    """In-place per-row L2 clipping of X.grad."""
    if X.grad is None:
        return
    grad_norms = X.grad.norm(dim=-1, keepdim=True)
    scale = (max_norm / grad_norms.clamp(min=1e-12)).clamp(max=1.0)
    X.grad.mul_(scale)


def project_entropy(X, entropy_factor):
    """Entropy projection (expansion toward one-hot). Upper bound semantics."""
    if entropy_factor <= 0:
        return X
    eps = 1e-8
    support = (X > 0).float()
    n_support = support.sum(dim=-1)
    safe_support = n_support.clamp(min=1.0)
    c = support / safe_support.unsqueeze(-1)
    target_entropy = (1.0 - entropy_factor) * (safe_support - 1.0) / safe_support
    R_sq = (1.0 - target_entropy) - 1.0 / safe_support
    R = R_sq.clamp(min=0.0).sqrt()
    diff = X - c
    d = diff.norm(dim=-1)
    needs_proj = (d < R) & (n_support > 1)
    scale = R / d.clamp(min=eps)
    expanded = scale.unsqueeze(-1) * diff + c
    return torch.where(needs_proj.unsqueeze(-1), expanded, X)


def dynamic_entropy_factor(entropy_factor, relaxation_gap, threshold=0.1):
    """Scale entropy_factor by the relaxation gap (closed-loop feedback)."""
    if relaxation_gap is None or entropy_factor <= 0:
        return entropy_factor
    gap = max(0.0, min(1.0, relaxation_gap))
    if gap >= threshold:
        return entropy_factor
    squeeze = 1.0 / (1.0 - threshold)
    x = squeeze * gap
    scale = 0.0 if x <= 0 else 1.0 / (1.0 + (1.0 / x - 1.0) ** 2)
    return scale * entropy_factor


class PGDOptimizer:
    def __init__(self, embed_matrix, n_learnable, frozen_embeds=None,
                 original_ids=None, init="random",
                 lr=0.1, num_steps=1000,
                 entropy_factor=0.0, dynamic_entropy=False,
                 dynamic_threshold=0.1, entropy_warmup_steps=0,
                 discrete_every=10, grad_clip=20.0, proj_iter=1,
                 mini_batch_size=None, patience=0, seed=0,
                 lr_scheduler=None, warmup_steps=100, cosine_t0=60,
                 cosine_eta_min_frac=0.1,
                 log_every=10, tokenizer=None):
        self.embed_matrix = embed_matrix
        self.frozen_embeds = frozen_embeds
        self.n_learnable = n_learnable
        self.lr = lr
        self.num_steps = num_steps
        self.entropy_factor = entropy_factor
        self.dynamic_entropy = dynamic_entropy
        self.dynamic_threshold = dynamic_threshold
        self.entropy_warmup_steps = entropy_warmup_steps
        self.discrete_every = discrete_every
        self.grad_clip = grad_clip
        self.proj_iter = proj_iter
        self.mini_batch_size = mini_batch_size
        self.patience = patience
        self.seed = seed
        self.lr_scheduler_type = lr_scheduler
        self.warmup_steps = warmup_steps
        self.cosine_t0 = cosine_t0
        self.cosine_eta_min_frac = cosine_eta_min_frac
        self.log_every = log_every
        self.tokenizer = tokenizer  # for decoding during logging

        device = embed_matrix.device
        V = embed_matrix.shape[0]
        self.V = V

        # Initialize X on the simplex (fp32 for Adam + projection precision)
        if init == "original" and original_ids is not None:
            self.X = F.one_hot(original_ids, num_classes=V).float()
            self.original_ids = original_ids
        elif init == "random":
            torch.manual_seed(seed)
            self.X = torch.rand(n_learnable, V, device=device)
            self.X = self.X / self.X.sum(dim=-1, keepdim=True)
            self.original_ids = original_ids
        else:
            self.X = torch.rand(n_learnable, V, device=device)
            self.X = self.X / self.X.sum(dim=-1, keepdim=True)
            self.original_ids = original_ids

        self.X.requires_grad_(True)

    def get_embeds(self):
        z = x_to_embeds(self.X, self.embed_matrix)
        if self.frozen_embeds is not None:
            return torch.cat([self.frozen_embeds, z], dim=0)
        return z

    def get_discrete_embeds(self):
        with torch.no_grad():
            ids = self.X.argmax(dim=-1)
            z = self.embed_matrix[ids]
            if self.frozen_embeds is not None:
                return torch.cat([self.frozen_embeds, z], dim=0)
            return z

    def _decode_current(self):
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(
            self.X.argmax(dim=-1).tolist(), skip_special_tokens=False
        )

    def _build_scheduler(self, opt):
        if self.lr_scheduler_type != "cosine":
            return None
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-4, end_factor=1.0,
            total_iters=self.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.cosine_t0, T_mult=2,
            eta_min=self.lr * self.cosine_eta_min_frac
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )

    def run(self, objective):
        rng = random.Random(self.seed)
        X = self.X
        n = self.n_learnable

        optimizer = torch.optim.Adam([X], lr=self.lr)
        scheduler = self._build_scheduler(optimizer)

        history = {
            "train": [], "val": [], "test": [],
            "discrete_train": [], "discrete_val": [], "discrete_test": [],
            "tsallis_entropy": [], "n_tokens_diff": [],
            "grad_norm_mean": [], "grad_norm_max": [],
            "entropy_factor_eff": [], "relaxation_gap": [],
        }
        best_discrete_val = float("inf")
        best_ids = X.argmax(dim=-1).clone()
        best_step = 0

        # Print initial state
        with torch.no_grad():
            init_z = self.get_discrete_embeds()
            init_train = objective.loss(init_z, "train").item()
            init_val = objective.loss(init_z, "val").item()
            init_test = objective.loss(init_z, "test").item()
            print(f"  init: train={init_train:.4f} val={init_val:.4f} "
                  f"test={init_test:.4f}")
            if self.tokenizer:
                print(f"    tokens: {self._decode_current()!r}", flush=True)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Forward pass — pass get_embeds as callable so graph is fresh per rollout
            train_loss_val = objective.loss(self.get_embeds, "train", backward=True)

            # Per-token grad stats + clipping
            with torch.no_grad():
                pre_clip_norms = X.grad.norm(dim=-1)
                grad_norm_mean = pre_clip_norms.mean().item()
                grad_norm_max = pre_clip_norms.max().item()
            clip_per_token_grad_(X, max_norm=self.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Project back to simplex
            with torch.no_grad():
                X.data = project_simplex(X.data)

            # Entropy factor: warmup + dynamic feedback
            if self.entropy_warmup_steps > 0 and step < self.entropy_warmup_steps:
                ef_base = self.entropy_factor * (step / self.entropy_warmup_steps)
            else:
                ef_base = self.entropy_factor

            if self.dynamic_entropy and ef_base > 0 and history["relaxation_gap"]:
                prev_gap = history["relaxation_gap"][-1]
                ef_eff = dynamic_entropy_factor(ef_base, prev_gap,
                                                self.dynamic_threshold)
            else:
                ef_eff = ef_base

            # Entropy projection
            with torch.no_grad():
                if ef_eff > 0:
                    for _ in range(self.proj_iter):
                        X.data = project_entropy(X.data, ef_eff)
                        X.data = project_simplex(X.data)

            # Relaxed eval
            with torch.no_grad():
                z = self.get_embeds()
                val_loss = objective.loss(z, "val").item()
                test_loss = objective.loss(z, "test").item()
                ent = tsallis_entropy(X).item()
                cur_ids = X.argmax(dim=-1)
                n_diff = (cur_ids != best_ids).sum().item() if self.original_ids is None \
                    else (cur_ids != self.original_ids).sum().item()

            # Discrete eval
            do_discrete = (step % self.discrete_every == 0) or \
                          (step == self.num_steps - 1)
            force_d_train = self.dynamic_entropy and self.entropy_factor > 0
            if do_discrete or force_d_train:
                with torch.no_grad():
                    d_z = self.get_discrete_embeds()
                    d_train = objective.loss(d_z, "train").item()
            else:
                d_train = None
            if do_discrete:
                with torch.no_grad():
                    d_z = self.get_discrete_embeds()
                    d_val = objective.loss(d_z, "val").item()
                    d_test = objective.loss(d_z, "test").item()
            else:
                d_val = None
                d_test = None

            # Relaxation gap
            if d_train is not None and d_train > 0:
                gap = (d_train - train_loss_val) / d_train
            else:
                gap = None

            # Record history
            history["train"].append(train_loss_val)
            history["val"].append(val_loss)
            history["test"].append(test_loss)
            history["discrete_train"].append(d_train)
            history["discrete_val"].append(d_val)
            history["discrete_test"].append(d_test)
            history["tsallis_entropy"].append(ent)
            history["n_tokens_diff"].append(n_diff)
            history["grad_norm_mean"].append(grad_norm_mean)
            history["grad_norm_max"].append(grad_norm_max)
            history["entropy_factor_eff"].append(ef_eff)
            history["relaxation_gap"].append(gap)

            if d_val is not None and d_val < best_discrete_val:
                best_discrete_val = d_val
                best_step = step
                best_ids = cur_ids.clone()

            # Patience reset
            did_reset = False
            if self.patience > 0 and (step - best_step) >= self.patience:
                with torch.no_grad():
                    X.data = F.one_hot(best_ids, num_classes=self.V).float()
                optimizer.state.clear()
                best_step = step
                did_reset = True

            # Logging
            if step % self.log_every == 0:
                star = " *" if (d_val is not None and
                                d_val == best_discrete_val) else ""
                d_train_s = f"{d_train:.4f}" if d_train is not None else "  -  "
                d_val_s = f"{d_val:.4f}" if d_val is not None else "  -  "
                d_test_s = f" hard_test={d_test:.4f}" if d_test is not None else ""
                gap_s = f" gap={gap:+.3f}" if gap is not None else ""
                ef_s = f" ef={ef_eff:.3f}" if self.entropy_factor > 0 else ""
                reset_s = " RESET" if did_reset else ""
                print(f"  step {step:3d}/{self.num_steps} "
                      f"soft_train={train_loss_val:.4f} soft_val={val_loss:.4f} "
                      f"hard_train={d_train_s} hard_val={d_val_s}{d_test_s} "
                      f"S2={ent:.3f} n_diff={n_diff}/{n}{gap_s}{ef_s} "
                      f"|g|={grad_norm_mean:.2f}(max {grad_norm_max:.1f})"
                      f"{star}{reset_s}", flush=True)
                if self.tokenizer:
                    print(f"    tokens: {self._decode_current()!r}", flush=True)

        # Final summary
        best_text = self.tokenizer.decode(best_ids.tolist(),
                                          skip_special_tokens=False) \
            if self.tokenizer else ""
        print(f"  best_discrete_val={best_discrete_val:.4f} "
              f"best_step={best_step}")
        if best_text:
            print(f"  best_text: {best_text!r}")

        return {
            "best_ids": best_ids.cpu(),
            "best_text": best_text,
            "best_step": best_step,
            "history": history,
            "test_opt": history["discrete_test"][best_step],
        }
