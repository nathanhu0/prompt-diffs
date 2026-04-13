# Text Optimization Framework

## Problem

We want to optimize text (or a text region) to minimize some objective. The two axes of variation are:

1. **Objective**: what we're optimizing for (NLL distillation, CoT judge, future things)
2. **Optimizer**: how we search over text (soft prompt, PGD, GCG, BoN, future things)

These should be independently swappable via config.

## Who Knows What

The key insight: the **objective** doesn't know what's frozen vs learnable. It receives a full block of embeddings for the "abstract slot" in the chat template and scores them. Whether those embeddings came from a fully-learned soft prompt, or frozen-abstract + learned-suffix, is invisible to it.

The **optimizer** owns the frozen/learnable split. It knows "I have some frozen context embeddings and N learnable positions." It composes them via `get_embeds()` and passes the result to the objective.

The **runner** (just `main()`, not a new abstraction) wires the config into the optimizer constructor — passing frozen embeddings, number of learnable positions, init strategy.

## Core Interfaces

### Objective

Scores a full block of embeddings for the abstract slot. Owns the model, data, and evaluation logic. Knows nothing about what's frozen vs optimized.

```python
class Objective:
    """Scores candidates against some criterion."""

    embed_matrix: Tensor     # (vocab, dim) — optimizers may need this for init

    def loss(self, z: Tensor, split: str = "train") -> Tensor:
        """Differentiable loss from slot embeddings.

        Args:
            z: (n_slot, dim) — full embeddings for the abstract slot.
                Could be any length. The objective plugs them into the
                chat template and scores.
            split: "train", "val", or "test"

        Returns:
            Differentiable scalar loss (lower is better).
        """

    def score_text(self, text: str, split: str = "train") -> float:
        """Non-differentiable score from text.

        Args:
            text: text to substitute into the abstract slot
            split: "train", "val", or "test"

        Returns:
            Scalar score (lower is better, same scale as loss).
        """
```

The objective internally handles:
- Tokenization of rollouts/prompts (pre-computed at init)
- Composing z with each rollout's prefix/suffix context (split-and-concat)
- Train/val/test splits
- Averaging over rollouts

#### Concrete objectives

**NLLDistillObjective**: Pre-tokenizes reference rollouts. For each rollout, stores prefix_ids (everything before abstract slot) and suffix_ids (everything after, including query and rollout). `loss()` does `cat(embed[prefix], z, embed[suffix])` -> forward -> NLL on target tokens. `score_text()` tokenizes the substituted text and does the same without gradients.

**CoTJudgeObjective** (future): `score_text()` sends text to a judge model, parses score. `loss()` raises NotImplementedError. Only usable with sampling-based optimizers.

### Optimizer

Owns the parameterization, the frozen/learnable split, and the optimization loop. Produces full slot embeddings via `get_embeds()`.

```python
class Optimizer:
    """Searches over text to minimize an objective."""

    def __init__(self, embed_matrix, n_learnable, frozen_embeds=None, ...):
        """
        Args:
            embed_matrix: (vocab, dim) for init and PGD projection
            n_learnable: number of learnable token positions
            frozen_embeds: (n_frozen, dim) optional frozen prefix context.
                If provided, get_embeds() returns cat(frozen, learnable).
                If None, get_embeds() returns just the learnable part.
        """

    def get_embeds(self) -> Tensor:
        """Compose frozen + learnable into full slot embeddings."""
        # Soft full:   return z
        # Soft suffix:  return cat(self.frozen, z)
        # PGD full:    return X @ embed_matrix
        # PGD suffix:  return cat(self.frozen, X @ embed_matrix)

    def run(self, objective: Objective) -> Result:
        """Run optimization loop.

        Each step:
            z = self.get_embeds()
            loss = objective.loss(z, "train")
            loss.backward()  # grads flow to learnable params only
            self.step()

        Early-stops on objective.loss(self.get_embeds(), "val").
        """
```

#### Concrete optimizers

**SoftPromptOptimizer**: learnable state is z (n_learnable, dim). Directly optimized with Adam. `get_embeds()` returns `cat(frozen, z)` or just `z`.

**PGDOptimizer**: learnable state is X (n_learnable, vocab) on the simplex. `get_embeds()` returns `cat(frozen, X @ embed_matrix)`. Handles entropy constraints, patience resets, discrete eval.

**GCGOptimizer** (future): maintains token_ids for the learnable positions. Uses gradients to rank token substitutions. `get_embeds()` returns `cat(frozen, embed_matrix[ids])`.

**BoNOptimizer** (future): maintains text. Uses an LLM to propose rewrites. Calls `objective.score_text()` instead of `loss()`. Needs its own LLM config.

## Config Format

```yaml
objective:
  type: nll_distill
  model: meta-llama/Llama-3.1-8B-Instruct
  rollouts: /nlp/scr/nathu/latent_rewrite/context_distill/positive.parquet

optimizer:
  type: pgd
  mode: suffix          # "full" or "suffix" — determines frozen/learnable split
  suffix_length: 25     # suffix mode only
  init: random          # "original", "random", "zeros"
  lr: 0.1
  num_steps: 1000
  patience: 100
  entropy_factor: 0.4
  dynamic_entropy: true

run:
  limit: 1
  log_every: 10
  output: /nlp/scr/nathu/latent_rewrite/results/pgd_suffix25.pt
```

Note: `mode` and `suffix_length` live under `optimizer`, because the optimizer is the one that handles the frozen/learnable split.

## Runner

```python
def main(config):
    papers = load_papers(config)
    model, tokenizer = load_model(config.objective.model)
    embed_matrix = get_embed_matrix(model)

    for paper in papers:
        # Objective just needs the paper text and rollouts
        objective = build_objective(config, model, tokenizer, paper)

        # Runner computes frozen context for the optimizer
        abstract_ids = tokenizer.encode(paper.abstract)
        if config.optimizer.mode == "suffix":
            frozen_embeds = embed_matrix[abstract_ids]
            n_learnable = config.optimizer.suffix_length
        else:  # full
            frozen_embeds = None
            n_learnable = len(abstract_ids)

        optimizer = build_optimizer(config, embed_matrix, n_learnable,
                                    frozen_embeds=frozen_embeds,
                                    original_ids=abstract_ids if not frozen_embeds else None)
        result = optimizer.run(objective)
        save(result)
```

## What Lives Where

```
optimize/
    objectives/
        base.py          # Objective ABC
        nll_distill.py   # NLLDistillObjective
    optimizers/
        base.py          # Optimizer ABC + get_embeds logic
        soft.py          # SoftPromptOptimizer
        pgd.py           # PGDOptimizer
    runner.py            # main loop: load data, wire objective+optimizer, run
configs/
    pgd_suffix_25.yaml
```

## Key Design Decisions

1. **Objective receives full slot embeddings, not a subset.** It doesn't know what's frozen or learnable. It just plugs the embeddings into the chat template and scores them.

2. **Optimizer owns the frozen/learnable composition.** `get_embeds()` is the single method that composes frozen context with learnable params. All the mode logic (full vs suffix) lives here.

3. **Embeddings are the common interface between objective and optimizer.** Gradient-based optimizers produce embeddings (however they want internally). The objective scores embeddings. Clean cut.

4. **`score_text()` is the parallel interface for sampling-based methods.** BoN doesn't produce embeddings — it produces text. The objective can score that too. An objective implements one or both interfaces.

5. **Early stopping and logging live in the optimizer.** Different optimizers have different diagnostics. The optimizer calls `objective.loss(self.get_embeds(), "val")` when it needs a val score.

6. **Runner does minimal wiring.** It reads the config, computes frozen_embeds based on mode, and passes them to the optimizer constructor. Not a new abstraction — just the main function.
