"""Soft-prompt optimization: Adam on continuous z with LR schedule + val-best
checkpointing. Reusable by LARGO (per-round soft phase) and as a standalone
skyline / interp tool.

Public surface:
  - SoftConfig: dataclass of all hparams; YAML-loadable via from_yaml_block.
  - train_soft(objective, z_list, cfg, ...) -> dict with best_z, history, etc.
"""
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch


def init_random_z(n_learnable, embed_matrix, device):
    """Random init scaled by embed_matrix.std() — matches LARGO's init=random."""
    z = (torch.randn(n_learnable, embed_matrix.shape[1],
                     device=device, dtype=embed_matrix.dtype)
         * embed_matrix.std())
    return z.detach().requires_grad_(True)


@dataclass
class SoftConfig:
    """Hyperparameters for one soft-prompt training pass.

    Schedule: linear warmup from 0 → lr over `warmup_steps`, then `schedule`
    decays lr → 0 over the remaining steps. `constant` skips decay (stays at
    lr after warmup).

    Val-based selection: if `val_every` is set, eval on the val split every
    that-many steps and track best-val z. Final return includes both the
    final-step z and the best-val z. A final val eval is always run at the
    last step regardless of `val_every`, so `best_val` / `history["val"][-1]`
    is meaningful even with `val_every=None`.
    """
    # --- optimization ---
    lr: float = 1e-3
    weight_decay: float = 1e-3
    steps: int = 2000               # standalone default; LARGO overrides per-round

    # --- LR schedule ---
    schedule: str = "cosine"        # "cosine" | "linear" | "constant"
    warmup_steps: int = 0

    # --- batching (forwarded to objective.loss) ---
    mini_batch_size: Optional[int] = 16
    train_batch_size: Optional[int] = 16

    # --- validation ---
    val_every: Optional[int] = None       # None = no periodic val eval

    # --- logging ---
    log_every: Optional[int] = None       # None = ~10 lines across run

    @classmethod
    def from_yaml_block(cls, block: Dict[str, Any]) -> "SoftConfig":
        """Coerce known-float fields (YAML 1.1 parses '3e-3' as string).
        Drops the optional `type` key so this can be used either nested under
        an optimizer block (no type) or as the top-level optimizer block in
        the standalone runner (type: soft).
        """
        cfg = {k: v for k, v in block.items() if k != "type"}
        for key in ("lr", "weight_decay"):
            if isinstance(cfg.get(key), str):
                cfg[key] = float(cfg[key])
        assert cfg.get("schedule", "cosine") in ("cosine", "linear", "constant"), \
            f"unknown schedule {cfg.get('schedule')!r}"
        return cls(**cfg)


def _make_lr_lambda(steps: int, schedule: str, warmup_steps: int):
    """LambdaLR multiplier: linear warmup 0→1 over warmup_steps, then decay.

    decay over `steps - warmup_steps`:
      - cosine: 0.5 * (1 + cos(pi * progress))
      - linear: 1 - progress
      - constant: 1.0 (no decay)
    """
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        decay_total = max(1, steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_total))
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule == "linear":
            return 1.0 - progress
        raise ValueError(f"unknown schedule {schedule!r}")
    return lr_lambda


def train_soft(
    objective,
    z_list: List[torch.Tensor],
    cfg: SoftConfig,
    *,
    get_embeds: Optional[Callable[[], List[torch.Tensor]]] = None,
    log_prefix: str = "",
):
    """Train soft prompt(s) with Adam + LR schedule + best-by-val checkpoint.

    z_list: list of leaf tensors (one per slot), each `requires_grad=True`.
        Caller owns init (random/zeros/from-tokens). Multi-slot supported.
    get_embeds: optional callable returning the list of embedding tensors to
        pass to the objective. Defaults to (lambda: z_list). LARGO passes a
        custom one that prepends frozen_embeds.
    cfg: SoftConfig — all knobs (lr, schedule, val_every, batching, ...).

    Returns dict with: final_z, best_z, best_val, best_step, history.
    A final val eval is always run at the last step (cheap; one extra
    forward pass), so `best_val` is meaningful even with val_every=None
    (in that case best_z == final_z by construction).
    """
    if get_embeds is None:
        get_embeds = lambda: z_list

    optimizer = torch.optim.Adam(
        z_list, lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _make_lr_lambda(cfg.steps, cfg.schedule, cfg.warmup_steps),
    )

    log_every = (cfg.log_every if cfg.log_every is not None
                 else max(1, cfg.steps // 10))
    eval_bs = (cfg.mini_batch_size * 4
               if cfg.mini_batch_size else None)

    history: Dict[str, List] = {
        "train": [], "val": [], "val_steps": [], "lr": [], "grad_norm": [],
    }
    best_val = float("inf")
    best_z = [z.detach().clone() for z in z_list]
    best_step = -1

    # Epoch-style sampler: a queue of shuffled indices, refilled with a
    # fresh torch.randperm whenever it runs low. Each step pops
    # train_batch_size indices and passes them to objective.loss(indices=).
    # Strictly better data coverage than random-with-replacement at the
    # batch_size= path of objective.loss, and matches standard SGD practice.
    # If train_batch_size is None or >= n_train, fall back to full-batch
    # (no shuffling; indices=None lets the objective use all examples).
    n_train = len(objective.examples_by_split["train"])
    bs = cfg.train_batch_size
    do_shuffle = bs is not None and bs < n_train
    shuffled: List[int] = []

    for step in range(cfg.steps):
        if do_shuffle:
            while len(shuffled) < bs:
                shuffled.extend(torch.randperm(n_train).tolist())
            batch_indices = shuffled[:bs]
            shuffled = shuffled[bs:]
        else:
            batch_indices = None

        optimizer.zero_grad()
        train_loss = objective.loss(
            get_embeds, "train", backward=True,
            mini_batch_size=cfg.mini_batch_size,
            indices=batch_indices,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(z_list, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        history["train"].append(train_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grad_norm"].append(float(grad_norm))

        val_str = ""
        eval_now = cfg.val_every and step % cfg.val_every == 0
        if eval_now:
            with torch.no_grad():
                val_loss = objective.loss(
                    get_embeds(), "val", mini_batch_size=eval_bs,
                ).item()
            history["val"].append(val_loss)
            history["val_steps"].append(step)
            mark = ""
            if val_loss < best_val:
                best_val = val_loss
                best_z = [z.detach().clone() for z in z_list]
                best_step = step
                mark = " *"
            val_str = f"  val={val_loss:.4f}{mark}"

        if step % log_every == 0 or step == cfg.steps - 1 or val_str:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  {log_prefix}step {step:4d}/{cfg.steps}  "
                  f"lr={lr_now:.2e}  train={train_loss:.4f}{val_str}",
                  flush=True)

    # Append history: final soft-prompt validation. Always runs, regardless
    # of val_every, so best_val is meaningful.
    with torch.no_grad():
        final_val = objective.loss(
            get_embeds(), "val", mini_batch_size=eval_bs,
        ).item()
    history["val"].append(final_val)
    history["val_steps"].append(cfg.steps - 1)
    if final_val < best_val:
        best_val = final_val
        best_z = [z.detach().clone() for z in z_list]
        best_step = cfg.steps - 1
    print(f"  {log_prefix}final val={final_val:.4f}", flush=True)

    return {
        "final_z": [z.detach().clone() for z in z_list],
        "best_z": best_z,
        "best_val": best_val,
        "best_step": best_step,
        "history": history,
    }
