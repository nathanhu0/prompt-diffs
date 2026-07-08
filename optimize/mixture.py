"""Mixture-of-soft-prompts: streaming hard-min training of K soft prompts.

Objective: oracle / hindsight loss (Multiple Choice Learning) — each example
is scored under every prompt and assigned to its argmin; only the winning
prompt receives gradient. Collapse control is DeepSeek-V3-style aux-loss-free
load balancing: assignment uses `argmin_k(NLL_ik + m_t * b_k)` where b_k is a
per-prompt scalar bias driven by an integral controller on load error, and
m_t is a decay multiplier that anneals the balance pressure to zero so late
training is pure argmin (unneeded prompts are allowed to go idle).

Public surface:
  - MixtureConfig: all hparams; YAML-loadable via from_yaml_block.
  - per_example_nll(objective, z, split, ...) -> (sums, counts) per example.
  - train_mixture(objective, z_list_k, cfg, labels_by_split=...) -> dict.

The bias only steers ASSIGNMENT; the loss/gradient always uses the true NLL.
Eval-time assignment (val diagnostics) is always pure argmin, no bias.
"""
import math
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional

import torch

from optimize.objectives.nll import nll_loss_batch
from optimize.soft import _make_lr_lambda


@torch.no_grad()
def per_example_nll(objective, z, split="train", indices=None,
                    mini_batch_size=24):
    """Per-example NLL of one soft prompt over a (sub)split.

    Returns (sums, counts): (N,) float tensor of summed target-token NLL and
    (N,) long tensor of target-token counts. Per-example per-token mean is
    sums / counts (length-invariant score for assignment); split-level
    per-token mean over a selection A is sums[A].sum() / counts[A].sum()
    (matches NLLObjective.loss reduction).
    """
    examples = objective.examples_by_split[split]
    if indices is not None:
        examples = [examples[i] for i in indices]
    all_sums, all_counts = [], []
    for i in range(0, len(examples), mini_batch_size):
        chunk = examples[i:i + mini_batch_size]
        sums, counts = nll_loss_batch(
            objective.model, [e.template for e in chunk],
            [e.target_ids for e in chunk], z)
        all_sums.append(sums.float())
        all_counts.append(counts)
    return torch.cat(all_sums), torch.cat(all_counts)


def weighted_nll_backward(objective, z, indices, weights,
                          mini_batch_size=8, denom=None):
    """Backward of the weighted per-token mean NLL over selected train
    examples: `sum_i w_i * sum_nll_i / sum_i w_i * count_i`. Weights of 1
    recover objective.loss's reduction exactly. Used by the eps_wta and
    anneal methods, where every prompt gets (down-weighted) gradient from
    every example. Returns the weighted mean as a float.

    denom overrides the self-normalizer (pass 1.0 with pre-normalized
    weights, e.g. the grouped weighting where each routing group is
    mean-reduced separately before its eps coefficient).
    """
    examples = [objective.examples_by_split["train"][i] for i in indices]
    if denom is None:
        denom = sum(w * len(e.target_ids)
                    for w, e in zip(weights.tolist(), examples))
    total = 0.0
    for c0 in range(0, len(examples), mini_batch_size):
        chunk = examples[c0:c0 + mini_batch_size]
        sums, _ = nll_loss_batch(
            objective.model, [e.template for e in chunk],
            [e.target_ids for e in chunk], z)
        chunk_loss = (weights[c0:c0 + len(chunk)] * sums).sum() / denom
        chunk_loss.backward()
        total += chunk_loss.item()
    return total


@dataclass
class MixtureConfig:
    """Hyperparameters for one mixture training pass.

    method selects the collapse-control family from the MCL literature:
      - "hard"    : winner-take-all argmin (+ optional bias controller).
      - "eps_wta" : relaxed WTA (Rupprecht et al. 2017) — winner weight
                    1-eps, losers eps/(K-1). Every prompt gets gradient
                    from every example, so no prompt leaves the data
                    manifold; bias fields still usable but default off.
      - "anneal"  : deterministic annealing (aMCL, Perera et al. 2024) —
                    soft responsibilities softmax(-NLL/T), T decayed
                    exponentially anneal_T0 -> anneal_T_min over the first
                    anneal_end_frac of training, hard argmin after.

    bias_gamma=0 disables the controller entirely (pure argmin — the collapse
    baseline). bias_decay_frac=None keeps gamma constant; a float f anneals the
    bias multiplier m_t linearly 1 -> 0 over the first f fraction of training,
    after which assignment is pure argmin.
    """
    k: int = 4
    method: str = "hard"
    eps: float = 0.05
    # eps_wta gradient normalization:
    #   "pooled" : one self-normalized weighted per-token mean over the
    #              whole batch (denominator = the member's total weighted
    #              token count) — step magnitude ~constant regardless of
    #              win count.
    #   "sample" : simple per-sample weighting with a FIXED denominator —
    #              loss = (1/B) sum_i w_i * (nll_sum_i / count_i). No
    #              member-dependent normalization: a member winning few
    #              examples gets a proportionally small (and noisier)
    #              gradient; Adam absorbs slow scale differences. Same
    #              winner/loser directional mix as pooled (leak still
    #              grows with loser count). Requires accumulate=False
    #              (the example-count step accounting assumes
    #              unnormalized weights).
    weighting: str = "pooled"
    anneal_T0: float = 0.2
    anneal_T_min: float = 0.005
    anneal_end_frac: float = 0.5

    # --- optimization (defaults = frozen Exp-1/Exp-2 SALVE soft hparams) ---
    lr: float = 3e-3
    weight_decay: float = 1e-3
    steps: int = 2500
    epochs: Optional[int] = None    # if set, derives steps (overrides `steps`)
    schedule: str = "cosine"
    warmup_frac: float = 0.05

    # --- batching ---
    train_batch_size: int = 16
    mini_batch_size: int = 8          # grad-pass chunk
    score_mini_batch_size: int = 24   # no-grad scoring chunk (assignment + eval)
    # accumulate=True: per-member gradient accumulation to a
    # train_batch_size-example budget before stepping (noise-matching for
    # winner-take-all at small batches; per-call equal averaging is an
    # APPROXIMATION when a member's win count fluctuates within a window).
    # accumulate=False: every member steps every batch on its exact
    # self-normalized weighted-batch gradient — use with a functional batch
    # sized so fair-share winner mass is adequate (e.g. B = 16*K for
    # eps_wta; do NOT combine with method="hard" at small B, that regime
    # is what accumulation exists for).
    accumulate: bool = True

    # --- load balancing ---
    bias_gamma: float = 0.0
    bias_decay_frac: Optional[float] = None
    # bias update rule when bias_gamma > 0:
    #   "sign"   : DeepSeek-style symmetric integrator — every member,
    #              biases += gamma * sign(load - fair).
    #   "starve" : conservative deadband — only a member whose batch load
    #              falls below HALF its fair share (B/2K) is encouraged
    #              (bias -= gamma); above the threshold its bias drains
    #              back toward 0 and never goes positive, so healthy
    #              members are never pushed down. Evals report the
    #              partition both with and without the bias.
    bias_mode: str = "sign"

    # --- routing buffers ---
    # per-member FIFO of the most recently WON train indices (streaming
    # stand-in for a full-scan cluster; used as the verbalization scoring
    # set). Snapshotted at each new val-best alongside best_z.
    route_buffer_size: int = 256

    # --- diagnostics ---
    eval_every: int = 100
    log_every: int = 25

    @classmethod
    def from_yaml_block(cls, block: Dict[str, Any]) -> "MixtureConfig":
        cfg = {k: v for k, v in block.items() if k != "type"}
        for key in ("lr", "weight_decay", "warmup_frac", "bias_gamma",
                    "bias_decay_frac"):
            if isinstance(cfg.get(key), str):
                cfg[key] = float(cfg[key])
        return cls(**cfg)


def _confusion(assign, labels, k, n_labels):
    """counts[prompt][label] over an assignment vector."""
    counts = [[0] * n_labels for _ in range(k)]
    for a, l in zip(assign, labels):
        counts[a][l] += 1
    return counts


def _purity(confusion):
    """Weighted majority-label fraction over prompts with any load."""
    n = sum(sum(row) for row in confusion)
    return sum(max(row) for row in confusion if sum(row) > 0) / max(n, 1)


def trait_f1(confusion, label=0):
    """Best F1 for `label` over all 2^K-1 member-subset labelings (clustering
    F-measure on the member partition; IoU = F1/(2-F1) is monotone-equivalent;
    trivial floor at trait fraction f is 2f/(1+f))."""
    n_lab = sum(row[label] for row in confusion)
    best = 0.0
    for r in range(1, len(confusion) + 1):
        for sub in combinations(range(len(confusion)), r):
            tp = sum(confusion[j][label] for j in sub)
            size = sum(sum(confusion[j]) for j in sub)
            if not (size and n_lab):
                continue
            p, rec = tp / size, tp / n_lab
            if p + rec:
                best = max(best, 2 * p * rec / (p + rec))
    return best


@torch.no_grad()
def _eval_mixture(objective, z_list_k, split, labels, cfg, biases=None):
    """Full-split diagnostics under PURE argmin (no bias).

    Returns dict with the (N, K) per-token-mean matrix (fp16 cpu), assignment,
    oracle per-token NLL, per-prompt solo NLL, loads, confusion/purity,
    per-prompt utility (mean own -> second-best gap on assigned examples).
    If nonzero `biases` are passed, a parallel *_biased view (assignment /
    loads / confusion / purity under biased argmin) is added so
    specialization metrics can be compared with and without the bias.
    """
    k = len(z_list_k)
    sums_k, counts = [], None
    for z in z_list_k:
        s, c = per_example_nll(objective, [z], split,
                               mini_batch_size=cfg.score_mini_batch_size)
        sums_k.append(s)
        counts = c
    sums = torch.stack(sums_k, dim=1)            # (N, K) summed NLL
    means = sums / counts.unsqueeze(1)           # (N, K) per-token mean
    assign = means.argmin(dim=1)                 # (N,)
    own = sums.gather(1, assign.unsqueeze(1)).squeeze(1)
    oracle_nll = (own.sum() / counts.sum()).item()
    solo_nll = [(sums[:, j].sum() / counts.sum()).item() for j in range(k)]
    loads = torch.bincount(assign, minlength=k).tolist()
    # utility: how much worse each prompt's examples do under their 2nd best
    top2 = means.topk(2, dim=1, largest=False).values   # (N, 2)
    gap = top2[:, 1] - top2[:, 0]
    utility = [gap[assign == j].mean().item() if loads[j] > 0 else 0.0
               for j in range(k)]
    out = {
        "oracle_nll": oracle_nll,
        "solo_nll": solo_nll,
        "loads": loads,
        "utility": utility,
        "assignment": assign.to(torch.int8).cpu(),
        "matrix": means.to(torch.float16).cpu(),
    }
    if labels is not None:
        n_labels = max(labels) + 1
        conf = _confusion(assign.tolist(), labels, k, n_labels)
        out["confusion"] = conf
        out["purity"] = _purity(conf)
    if biases is not None and float(biases.abs().max()) > 0:
        assign_b = (means + biases.to(means.device).unsqueeze(0)).argmin(dim=1)
        out["assignment_biased"] = assign_b.to(torch.int8).cpu()
        out["loads_biased"] = torch.bincount(assign_b, minlength=k).tolist()
        if labels is not None:
            conf_b = _confusion(assign_b.tolist(), labels, k,
                                max(labels) + 1)
            out["confusion_biased"] = conf_b
            out["purity_biased"] = _purity(conf_b)
    return out


def train_mixture(objective, z_list_k, cfg: MixtureConfig, *,
                  labels_by_split=None, seed=0, log_prefix=""):
    """Train K soft prompts with streaming hard-min + bias load balancing.

    z_list_k: list of K leaf tensors (single-slot; each requires_grad=True).
    labels_by_split: optional {split: list[int]} ground-truth source labels,
        parallel to objective.examples_by_split — enables purity diagnostics.

    Returns dict with: final_z, best_z (by val oracle NLL), best_val,
    best_step, biases, history.
    """
    k = len(z_list_k)
    assert k == cfg.k, f"len(z_list_k)={k} != cfg.k={cfg.k}"
    if cfg.weighting == "sample":
        assert cfg.method == "eps_wta" and not cfg.accumulate, \
            "sample weighting is defined for eps_wta with accumulate=False"
    torch.manual_seed(seed)

    if cfg.epochs is not None:
        n_train = len(objective.examples_by_split["train"])
        cfg.steps = cfg.epochs * math.ceil(n_train / cfg.train_batch_size)
        print(f"  {log_prefix}epochs={cfg.epochs} -> steps={cfg.steps}",
              flush=True)
    warmup_steps = round(cfg.warmup_frac * cfg.steps)

    # One Adam per prompt + per-prompt gradient accumulation to a fixed
    # example budget (train_batch_size). A prompt's step fires only once it
    # has accumulated as many examples as a single-prompt run uses per
    # update, so update noise is IDENTICAL to the frozen SALVE regime no
    # matter how few examples the prompt wins per batch — low-load prompts
    # just step less often. Without this, a prompt winning 1-4 examples
    # takes ~4x-noise Adam steps at full lr, gets wrecked (solo NLL 3-6 vs
    # 0.7 at init), then never wins again (observed in the first 4-arm run).
    optimizers = [torch.optim.Adam([z], lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
                  for z in z_list_k]
    lr_lambda = _make_lr_lambda(cfg.steps, cfg.schedule, warmup_steps)
    accum_examples = [0] * k   # examples since last step, per prompt
    accum_calls = [0] * k      # backward calls since last step (grad is a
                               # SUM of per-call chunk means -> divide out)

    device = objective.device
    biases = torch.zeros(k, device=device)
    n_train = len(objective.examples_by_split["train"])
    train_labels = (labels_by_split or {}).get("train")
    val_labels = (labels_by_split or {}).get("val")
    fair = cfg.train_batch_size / k

    def bias_mult(step):
        if cfg.bias_gamma == 0:
            return 0.0
        if cfg.bias_decay_frac is None:
            return 1.0
        return max(0.0, 1.0 - step / (cfg.bias_decay_frac * cfg.steps))

    history: Dict[str, List] = {
        "train_oracle": [], "loads": [], "biases": [], "bias_mult": [],
        "lr": [], "batch_purity": [], "evals": [], "stepped": [],
        "anneal_T": [],
    }
    best_val = float("inf")
    best_z = [z.detach().clone() for z in z_list_k]
    best_step = -1
    route_buffers = [deque(maxlen=cfg.route_buffer_size) for _ in range(k)]
    best_route_buffers: List[List[int]] = [[] for _ in range(k)]
    shuffled: List[int] = []

    for step in range(cfg.steps):
        while len(shuffled) < cfg.train_batch_size:
            shuffled.extend(torch.randperm(n_train).tolist())
        batch_idx = shuffled[:cfg.train_batch_size]
        shuffled = shuffled[cfg.train_batch_size:]

        # --- scoring pass: (B, K) per-token-mean NLL, no grad ---
        sums_k, counts = [], None
        for z in z_list_k:
            s, counts = per_example_nll(
                objective, [z], "train", indices=batch_idx,
                mini_batch_size=cfg.score_mini_batch_size)
            sums_k.append(s)
        sums = torch.stack(sums_k, dim=1)                 # (B, K)
        means = sums / counts.unsqueeze(1)                # (B, K)

        # --- assignment: argmin over biased scores; loss uses true NLL ---
        m_t = bias_mult(step)
        assign = (means + m_t * biases.unsqueeze(0)).argmin(dim=1)  # (B,)
        loads = torch.bincount(assign, minlength=k)
        for i, j in enumerate(assign.tolist()):
            route_buffers[j].append(batch_idx[i])
        train_oracle = (sums.gather(1, assign.unsqueeze(1)).sum()
                        / counts.sum()).item()

        # --- per-example gradient weights (method-dependent) ---
        T_t = None
        if cfg.method == "eps_wta":
            W = torch.full((len(batch_idx), k), cfg.eps / max(k - 1, 1),
                           device=means.device)
            W.scatter_(1, assign.unsqueeze(1), 1.0 - cfg.eps)
        elif cfg.method == "anneal" \
                and step < cfg.anneal_end_frac * cfg.steps:
            frac = step / max(cfg.anneal_end_frac * cfg.steps, 1)
            T_t = cfg.anneal_T0 * (cfg.anneal_T_min / cfg.anneal_T0) ** frac
            W = torch.softmax(-means / T_t, dim=1)
        else:
            W = None    # hard winner-take-all (and anneal past its schedule)

        # --- grad pass: each prompt accumulates over its assigned examples,
        # steps only at a full train_batch_size worth (SALVE-noise-matched) ---
        lr_t = cfg.lr * lr_lambda(step)
        stepped = []
        for j in range(k):
            if W is None:
                assigned = [batch_idx[i] for i in range(len(batch_idx))
                            if assign[i] == j]
                if not assigned:
                    continue
                objective.loss(
                    [z_list_k[j]], "train", backward=True,
                    mini_batch_size=cfg.mini_batch_size, indices=assigned)
                accum_examples[j] += len(assigned)
            else:
                wj = W[:, j]
                denom = None
                if cfg.weighting == "sample":
                    wj = wj / (counts.float() * len(batch_idx))
                    denom = 1.0
                if wj.sum().item() < 1e-6:
                    continue
                weighted_nll_backward(
                    objective, [z_list_k[j]], batch_idx, wj,
                    mini_batch_size=cfg.mini_batch_size, denom=denom)
                # effective examples this call = total weight received
                accum_examples[j] += wj.sum().item()
            accum_calls[j] += 1
            if not cfg.accumulate or accum_examples[j] >= cfg.train_batch_size:
                z_list_k[j].grad /= accum_calls[j]
                torch.nn.utils.clip_grad_norm_([z_list_k[j]], max_norm=1.0)
                for g in optimizers[j].param_groups:
                    g["lr"] = lr_t
                optimizers[j].step()
                optimizers[j].zero_grad(set_to_none=True)
                accum_examples[j] = accum_calls[j] = 0
                stepped.append(j)

        # --- bias controller (integral on load error) ---
        if cfg.bias_gamma > 0 and m_t > 0:
            if cfg.bias_mode == "starve":
                starving = loads.float() < fair / 2
                biases[starving] -= cfg.bias_gamma
                biases[~starving] = (biases[~starving]
                                     + cfg.bias_gamma).clamp(max=0.0)
            else:
                biases += cfg.bias_gamma * torch.sign(loads.float() - fair)

        history["train_oracle"].append(train_oracle)
        history["anneal_T"].append(T_t)
        history["loads"].append(loads.tolist())
        history["biases"].append(biases.tolist())
        history["bias_mult"].append(m_t)
        history["lr"].append(lr_t)
        history["stepped"].append(stepped)
        if train_labels is not None:
            blabels = [train_labels[i] for i in batch_idx]
            conf = _confusion(assign.tolist(), blabels, k, max(train_labels) + 1)
            history["batch_purity"].append(_purity(conf))

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            bias_str = "/".join(f"{b:+.3f}" for b in biases.tolist())
            t_str = f"  T={T_t:.4f}" if T_t is not None else ""
            print(f"  {log_prefix}step {step:4d}/{cfg.steps}  "
                  f"oracle={train_oracle:.4f}  loads={loads.tolist()}  "
                  f"m={m_t:.2f}  b=[{bias_str}]{t_str}", flush=True)

        eval_now = (cfg.eval_every and step % cfg.eval_every == 0) \
            or step == cfg.steps - 1
        if eval_now:
            ev = _eval_mixture(objective, z_list_k, "val", val_labels, cfg,
                               biases=biases)
            ev["step"] = step
            # pairwise z cosine — clone-collapse watch
            flat = torch.stack([z.detach().flatten().float()
                                for z in z_list_k])
            ev["z_cos"] = torch.corrcoef(flat).cpu()
            history["evals"].append(ev)
            mark = ""
            if ev["oracle_nll"] < best_val:
                best_val = ev["oracle_nll"]
                best_z = [z.detach().clone() for z in z_list_k]
                best_route_buffers = [list(b) for b in route_buffers]
                best_step = step
                mark = " *"
            pur = f"  purity={ev['purity']:.3f}" if "purity" in ev else ""
            print(f"  {log_prefix}eval step {step}: val_oracle="
                  f"{ev['oracle_nll']:.4f}{mark}  loads={ev['loads']}{pur}  "
                  f"solo={[f'{s:.3f}' for s in ev['solo_nll']]}", flush=True)

    return {
        "final_z": [z.detach().clone() for z in z_list_k],
        "best_z": best_z,
        "best_val": best_val,
        "best_step": best_step,
        "biases": biases.tolist(),
        "route_buffers": [list(b) for b in route_buffers],
        "best_route_buffers": best_route_buffers,
        "history": history,
    }
