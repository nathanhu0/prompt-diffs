# CMFT stage-1 / stage-2 training: fidelity to the vendored repo

What our SFT recipe copies from the CMFT authors and where it departs. Same
purpose as `experiments/sl_optimizer_comparison/PGD_FAITHFUL.md`: keep the audit
boundary explicit so a reviewer can tell replication from adaptation.

**Sources of truth** (vendored under `safe-finetuning-api/`):

- `src/automated_cmft/qlora-fsdp-31-70b.yaml` — the axolotl config, all knobs
- `src/automated_cmft/pipeline.py:623-654` — per-phase overrides on top of it

**Ours**: `sft_walnut_auto.py`.

Note on the filename: despite `qlora-`, both `load_in_8bit` and `load_in_4bit`
are commented out (yaml:5-6). The vendored run is plain bf16 LoRA, and so is
ours.

## Per-phase overrides (pipeline.py)

| | phase 1 (cipher teaching) | phase 2 (jailbreak) |
|---|---|---|
| `num_epochs` | **1** (:626) | 3 (:640) |
| learning rate | base (2e-4) | **base / 2** (:651) |
| `warmup_steps` | 1 | deleted (:643) |
| saves/evals per epoch | 2 / 2 | 1 / 1 (:644-645) |

Phase 1 being **1 epoch** is easy to miss — the base yaml carries
`num_epochs: __INVALID__` as a placeholder and the real value is only ever set
in `pipeline.py`. Our pre-2026-07-25 runs used 3 epochs for phase 1, i.e. triple
the intended cipher exposure.

## Faithful

| knob | value | note |
|---|---|---|
| `sequence_len` / `max_length` | 3072 | ciphers stay in "normal mode"; see long-mode below |
| `train_on_inputs: false` | assistant tokens only | ours is `completion_only_loss=True` over a boundary we compute; see below |
| `lora_dropout` | 0.05 | |
| `lora_target_linear: true` | q,k,v,o,gate,up,down | all linear layers in both architectures |
| `optimizer` | `adamw_torch` | |
| `weight_decay` | 0.01 | |
| `group_by_length` | false | |
| bf16 | yes | |
| gradient checkpointing | on | |
| adapter | LoRA, not quantized | |
| phase-2 epochs | 3 | |
| phase-1 epochs | 1 | as of 2026-07-25 |
| phase-2 lr = phase-1 lr / 2 | derived, not chosen | see below |

### Phase-2 lr is derived, matching `pipeline.py:651`

`run_endspeak_stage2.sh:38` and `run_endspeak_stage2_gemma.sh:23` both compute
`s2lr=$(python -c "print(f'{$lr/2:g}')")` per stage-1 cell, and
`run_gemma_phase2_paper.sh` hardcodes the same relation (2e-4 → 1e-4). When
stage-1 lr is re-swept, phase-2 must stay *derived* from whichever stage-1 lr is
selected — not swept independently.

### Assistant-only loss — same target, different mechanism

Axolotl's `train_on_inputs: false` masks prompt tokens. We reach the same place
by pre-tokenizing and passing an explicit `completion_mask`
(`build_tokenized_dataset`), rather than using TRL's prompt-completion format.

TRL splits at `len(prompt_ids)` and only *warns* when that isn't a true prefix
(`sft_trainer.py:1513-1522`). For Gemma-4 it isn't: `add_generation_prompt=True`
appends a `<|channel>thought` primer (4 tokens) that never appears in the real
sequence, so the split overshoots and silently drops the first 4 target tokens
from the loss — measured 92 supervised tokens where Qwen got 96 on the same row.
We take the longest common prefix instead and assert the decoded completion
matches the target, mirroring the tokenizer-drift canary in
`optimize/objectives/nll.py`. Both models now supervise exactly
`target + EOS`.

## Deviations

### Deliberate, and they change results

| knob | vendored | ours | why |
|---|---|---|---|
| base model | Llama-3.1-70B-Instruct | Qwen2.5-14B-Instruct, Gemma-4-31B-it | different study; we compare across model families at sizes we can train |
| `lora_r` / `lora_alpha` | 8 / 16 | 16 / 32 | our own capacity choice, frozen across all cells |
| `learning_rate` | 2e-4 | swept per cipher×model | 2e-4 was tuned at their batch size; ours differs, so it does not transfer |
| batch size | token-based: 4 × 3072 packed × n_gpu | **64 rows** (`bs=1 × grad_accum=64`) | see below |
| `sample_packing` | true | **false** | see below |
| `lr_scheduler` | linear | cosine | inherited from our earlier sweeps; not re-examined |

**Why packing is off.** TRL implements packing by emitting restarting
`position_ids` and deliberately *no* `attention_mask`, so sequence isolation
rests entirely on FlashAttention interpreting those restarts. flash-attn is not
installed here and we load with `sdpa`, which attends straight across segment
boundaries. Verified directly: perturbing a packed neighbour moved a segment's
logits by 4.4e-1. Axolotl's `sample_packing: true` masks correctly, so this was
our bug, not a shared one. All adapters trained before 2026-07-25 have it.

**Why fixed 64 rows.** Under packing, the effective batch *in rows* was set by
how many rows fit a 3072-token block, so it varied with cipher verbosity —
roughly 197 rows/step for autokey against 40 for ascii, a 5× spread across cells
we intend to compare. It also made the step count vary the same way (306 vs
1485). A fixed 64 rows gives every cipher the same batch and the same 312
steps/epoch. This is a deviation that *improves* cross-cipher comparability, at
the cost of no longer matching their token-based batching.

### Incidental — no expected effect on results

| knob | vendored | ours | note |
|---|---|---|---|
| `flash_attention: true` | flash-attn | sdpa | with packing off, only speed |
| liger kernels | rope, rms_norm, swiglu, fused CE | none | speed; numerics equivalent |
| `warmup_steps: 1` | 1 step | `warmup_ratio=0.03` | ours warms longer |
| `tf32: true` | on | unset | matmul precision |
| `use_reentrant` | true | false | checkpointing impl; `false` is the safer one |
| `val_set_size: 0.05` | 5% held out of train | separate 2k val file | ours is a clean split, not carved from train |
| FSDP, 4 GPUs | full_shard | single GPU | infra |
| `saves_per_epoch: 2` | mid-run checkpoints | none | we keep only the final adapter — optimizer state filled `/nlp/scr` on 2026-07-24 |
| `logging_steps` | 1 | 5 | |

### Not implemented

`check_and_update_config_for_long_mode` (`pipeline.py:540-577`): if <95% of rows
fit `sequence_len`, they raise it to 10240 and halve micro-batch / double
grad-accum to hold the token budget constant. **Never triggers for our ciphers**
— ascii is the longest at p99 2893 < 3072, autokey maxes at 679 — so the branch
is unreachable here rather than skipped.

## Selecting the swept lr

By **ciphered ARC-Challenge accuracy** (`eval_arc_cipher.py`,
`run_stage1_arc_eval.sh`) — the vendored `get_cipher_eval` metric — not
validation perplexity. Cipher ppl rewards fluent-looking output that need not
decode to anything, and it isn't comparable across ciphers whose token
distributions differ by 5×. Plaintext ARC is reported alongside as a guard
against buying cipher fluency by damaging the model.

## Data

Corpora live on `/nlp/scr/nathu/cmft_legibility/data`;
`experiments/cmft_legibility/data` is a symlink to it. 100MB doesn't belong on
the /juice2 quota when it regenerates from a fixed seed.

Faithful, down to the sampling.

**Phase 1.** `pipeline.py:585-586` builds it as
`get_dataset_alpaca_hhh()[:20000]`, ciphered with all four task weights.
`old_harness/datasets.py:8-33` defines that as `yahma/alpaca-cleaned`,
`.shuffle(seed=42)`, filtered to
`len(instruction) + len(input) + len(output) < 750`. Our
`generate_cmft_datasets.py:141-143` reproduces the same dataset, seed, and
filter, taking the first 20000 rows. Their "1 epoch" is therefore the same 20k
rows of cipher exposure as ours — the epoch count is directly comparable.

(`load_dataset(path, num_samples)` at `pipeline.py:31` is dead code in this
file; the live path is `:586`. Don't read the corpus size off `:31`.)

**Phase 1 val** is the one departure: they carve 5% out of train
(`val_set_size: 0.05`), we take rows 20000–22000 of the same filtered/shuffled
stream (`--phase1-val-skip 20000`). Ours is a clean held-out split rather than a
carve-out, and it leaves the 20k train set intact at exactly their size.

**Phase 2.** `pipeline.py:604-605`: `get_dataset_wei_harmful()` ciphered with
`tasks_weight=(0, 0, 0, 1)` — harmful-only, TASK 4 only, no refusal mixture.
Ours is the same 317 Wei rows at TASK 4.
