# Sequence-length truncation and the Gemma memory ceiling

Why `max_total_tokens` exists, what it is set to, and what it costs. Read this
before changing the cap or re-routing jobs between GPU classes.

**Current setting: `max_total_tokens = 5120`, uniform across all 4 ciphers and
both models.** Set in `run_grid_z512.sh` (`CAP`), implemented in
`salve_data.py:build_cmft_objective`.

---

## 1. The problem: length is a long TAIL, not a long mean

Token counts of the full rendered chat (system + user + assistant + the soft-prompt
slot), Gemma tokenizer, phase-2 harmful data (317 rows/cipher):

| cipher | median | p95 | max | max ÷ median |
|---|---|---|---|---|
| walnut | 1490 | 2234 | 5494 | 3.7× |
| endspeak | 1525 | 2353 | 4287 | 2.8× |
| polybius | 1940 | 2954 | 7352 | 3.8× |
| **ascii** | **2736** | **4256** | **10463** | **3.8×** |

(z256 totals. Add 256 for z512.)

ascii's max is **2.5× its own p95**. The entire memory problem is driven by
**3 rows out of 317**. ASCII spells each character as a decimal code, so a long
response inflates enormously while the median stays modest.

The length is the **response**, not the prompt — target is **88%** of content
tokens (ascii prompt median 305 vs target 2203; max prompt 543). No row's
prefix+slot alone approaches any workable cap, so every row stays usable.

## 2. Why it OOMs: SDPA is O(seq²) and Gemma-31B leaves ~15GB

Gemma-4-31B is ~64GB of weights on a 79GB card. Attention materializes the full
matrix, so cost grows quadratically in sequence length against a fixed, small
headroom. Observed ladder, all at `mini_batch_size=1` with gradient checkpointing
on — **there is no batch knob left**, `salve_run.py:76` feeds
`salve_decode.mini_batch_size` to both the soft_eval NLL pass and the beam scorer:

| run | max seq | 80G outcome |
|---|---|---|
| z256 walnut, training | 5494 | ✅ trains |
| **z512 walnut, training** | **5750** | ❌ **OOM in soft phase** (3.94 GiB alloc) |
| z256 polybius, training | 7352 | ❌ OOM in soft phase (1.61 GiB alloc) |
| z256 ascii, training | 10463 | ❌ OOM in soft phase (13.05 GiB alloc) |
| z256 ascii, readout-only, **no cap** | 10463 | ❌ OOM in soft_eval **forward** (1.83 GiB) |
| z256 ascii/polybius, readout-only, cap 6144 | 6144 | ⚠️ soft_eval passes; **beam OOMs ~30%** (12.42 GiB) |
| anything ≤6144 on 141G H200 | — | ✅ |

Three things this table is the record of, each of which cost a wave of failed jobs:

1. **Skipping the backward pass is not sufficient.** Readout-only runs (`--soft-z`)
   still OOM'd uncapped — a single 10k-token *forward* already exceeds headroom.
2. **The binding quantity is TOTAL sequence length, not content length.**
   `z512_waln_g_s42` died at cap 6144 with the cap *not binding* (walnut max 5750,
   `truncated 0 target tails`). The extra 256 soft tokens over z256 are what tipped
   it. A cap chosen from content lengths will silently mislead.
3. **The beam is heavier than soft_eval.** With cap 6144 the soft_eval pass
   survived but beam scoring failed ~30% of the time on a 12.42 GiB allocation —
   the logits tensor is `mb × seq × 262k vocab`, ~3GB per sequence at 6144 even at
   mb=1, plus KV cache across candidates. Sizing a cap against training alone
   leaves the scoring path exposed.

## 3. Why 5120, uniformly

5120 is the largest round cap **below the 5494 that provably trains on 80G**. It
therefore removes the H200 dependency entirely — the whole 32-cell z512 grid runs
on A100s, which matters more for wall-clock than anything else, given how long the
141G queue runs.

Uniform rather than per-cell because a per-cell cap makes absolute NLL incomparable
across cells for no scientific reason. An earlier split (5120 for Gemma
walnut/endspeak, 6144 elsewhere) was rejected on those grounds.

**Cost at z512, measured on both tokenizers (near-identical):**

| cipher | rows > 5120 (of 317) | target tokens lost |
|---|---|---|
| endspeak | 0 | 0.00% |
| walnut | 1 (0.3%) | 0.12–0.18% |
| polybius | 3 (0.9%) | 0.65–0.73% |
| **ascii** | **10 (3.2%)** | **2.06%** |

Worst case is 2% of target tokens in one cell.

## 4. Truncate, don't drop — and why that is clean here

`build_cmft_objective` truncates the **target tail** and rebuilds the template from
the shortened text, so `total_len` and `target_ids` stay consistent (no splicing).

This is honest rather than distorting because the NLL reduction is a **per-token
mean over all target tokens in the split** (see repo `CLAUDE.md`): a shortened row
contributes fewer tokens to both numerator and denominator. Dropping the row would
remove it entirely; truncation keeps its first N tokens at full weight.

There is an assert that prefix+slot alone never exceeds the cap — measured true for
every row, but a silent empty target would be a meaningless example, so it fails
loudly instead. The loader prints `[cap N] truncated K target tails`; **K=0 means
the cap did not bind**, which is the signal that misled us at 6144.

## 5. The real fix, not yet done

**FlashAttention** would make attention O(seq) instead of O(seq²) and remove the
ceiling for training *and* scoring on 80G, with no truncation at all.
`flash_attn` is not installed and `core/models.py:49` hard-codes
`attn_implementation="sdpa"`. Worth doing if longer sequences or larger z ever
become interesting; note `sft_walnut_auto.py`'s packing comment — TRL's packing
relies on FlashAttention for sample isolation, which is why `packing=False` there,
so installing it has a second consequence to think through.

## 6. Applies to

- `run_grid_z512.sh` — `CAP=5120` (current grid)
- `run_gemma_decode_pools.sh` — `CAP=6144` (superseded; those runs were cancelled)
- The **z256 grid in `LADDER_RESULTS.md` ran with NO cap** — it predates this and
  is unaffected. Comparisons between the z256 and z512 grids carry a ≤2% target-
  token difference in the ascii cells and less elsewhere.
