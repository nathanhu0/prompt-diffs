"""Soft prompt optimizer — directly optimizes embeddings with Adam."""
import torch


class SoftPromptOptimizer:
    def __init__(self, embed_matrix, n_learnable, frozen_embeds=None,
                 original_ids=None, init="original",
                 lr=1e-3, num_steps=100, weight_decay=0.0,
                 mini_batch_size=None, log_every=5):
        self.embed_matrix = embed_matrix
        self.frozen_embeds = frozen_embeds
        self.lr = lr
        self.num_steps = num_steps
        self.weight_decay = weight_decay
        self.mini_batch_size = mini_batch_size
        self.log_every = log_every

        device = embed_matrix.device
        dim = embed_matrix.shape[1]

        # Initialize learnable embeddings (match embed_matrix dtype)
        dtype = embed_matrix.dtype
        if init == "original" and original_ids is not None:
            z = embed_matrix[original_ids].clone()
        elif init == "random":
            z = (torch.randn(n_learnable, dim, device=device, dtype=dtype)
                 * embed_matrix.std())
        elif init == "zeros":
            z = torch.zeros(n_learnable, dim, device=device, dtype=dtype)
        else:
            z = (torch.randn(n_learnable, dim, device=device, dtype=dtype)
                 * embed_matrix.std())

        self.z = z.detach().requires_grad_(True)

    def get_embeds(self):
        if self.frozen_embeds is not None:
            return torch.cat([self.frozen_embeds, self.z], dim=0)
        return self.z

    def run(self, objective):
        optimizer = torch.optim.Adam([self.z], lr=self.lr,
                                     weight_decay=self.weight_decay)
        history = {"train": [], "val": [], "test": []}
        best_val = float("inf")
        best_z = self.z.clone().detach()

        for step in range(self.num_steps):
            optimizer.zero_grad()
            train_loss = objective.loss(self.get_embeds, "train", backward=True,
                                       mini_batch_size=self.mini_batch_size)
            torch.nn.utils.clip_grad_norm_([self.z], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                z = self.get_embeds()
                val_loss = objective.loss(z, "val").item()
                test_loss = objective.loss(z, "test").item()

            history["train"].append(train_loss)
            history["val"].append(val_loss)
            history["test"].append(test_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_z = self.z.clone().detach()

            if step % self.log_every == 0:
                star = " *" if val_loss == best_val else ""
                print(f"  step {step:3d}/{self.num_steps} "
                      f"train={train_loss:.4f} val={val_loss:.4f} "
                      f"test={test_loss:.4f}{star}", flush=True)

        best_step = history["val"].index(min(history["val"]))
        return {
            "best_z": best_z,
            "best_step": best_step,
            "history": history,
            "test_opt": history["test"][best_step],
        }
