# Sequence-length truncation and the Gemma memory ceiling

Why `max_total_tokens` exists, what it is set to, and what it costs. Read this
before changing the cap or re-routing jobs between GPU classes.

**Current setting: `max_total_tokens = 5120`, uniform across all 4 ciphers and
both models.** Set in `run_grid_z512.sh` / `run_decode_variations.sh` (`CAP`),
implemented in `salve_data.py:build_cmft_objective`.

**As of the 2026-08-03 fix (§2), the cap bounds BOTH the soft phase and beam
scoring, so cap 5120 fits every cell on 80G and no run needs the 141G H200.**
Runs launched before that fix had scoring at full length; see §2 for which of
their numbers are affected.

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
| **z512 polybius, BEAM, cap 5120** | **5120** | ❌ **OOM, deterministic** (1.50 GiB alloc, 77.6 GiB resident) |
| anything ≤6144 on 141G H200 | — | ✅ |

> ## ⚠️ ROOT CAUSE FOUND 2026-08-03 — every "beam OOMs under the cap" row above
> ## was a BUG, not a memory ceiling.
>
> **The cap never applied to scoring.** `build_cmft_objective` truncated
> `target_ids` and rebuilt the template, but appended the **original** target text
> to `xy_by_split`. `NLLObjective.hard_loss` — the beam scorer, and the source of
> the reported verbalized NLL — re-tokenizes from that text. So the cap bounded
> the soft phase only, and the beam went on materializing full-length sequences.
>
> `z512_poly_g_s42` "OOM'd at cap 5120" while actually scoring its longest row at
> its full **7351** tokens. Fixed by truncating the target *text* and rebuilding
> from it, so template / `target_ids` / `xy_by_split` all agree (verified: at cap
> 5120 on ascii, max total 5120 and `target_ids` = re-tokenized xy target = 4670;
> previously 4670 vs 9916).
>
> **The retracted claim is lesson 3 below, "the beam is heavier than soft_eval."**
> It is false, and the argument against it is simple enough that it should have
> been applied at the time: beam scoring is a `no_grad` forward at mb=1 with no
> optimizer state and no stored activations for backward, so **at equal sequence
> length it cannot cost more than training the same sequence.** Anything that
> trains on 80G scores on 80G. Every apparent counterexample was the two paths
> running at different lengths.
>
> Consequences: (a) **cap 5120 does make the whole grid fit 80G** — the original
> plan was sound and only the implementation was broken; (b) the per-seed OOM
> pattern (§below) has the simpler explanation that a seed whose 64-row subset
> contained a long row scored it uncapped; (c) wherever the cap binds, runs from
> before this fix have soft NLL on truncated targets and verbalized NLL on full
> ones, so **their `gap` mixes two target sets** — ascii ~1.8% of target tokens,
> polybius ~0.5%, walnut ~0.15%, endspeak 0 (the cap never binds there).

Two things the table is still the record of:

1. **Skipping the backward pass is not sufficient.** Readout-only runs (`--soft-z`)
   still OOM'd uncapped — a single 10k-token *forward* already exceeds headroom.
   (Uncapped is the operative word; with the fix, a capped readout is fine.)
2. **The binding quantity is TOTAL sequence length, not content length.**
   `z512_waln_g_s42` died at cap 6144 with the cap *not binding* (walnut max 5750,
   `truncated 0 target tails`). The extra 256 soft tokens over z256 are what tipped
   it. A cap chosen from content lengths will silently mislead.
3. ~~**The beam is heavier than soft_eval.**~~ **RETRACTED — see the box above.**
   The ~30% beam-OOM rate at cap 6144 was full-length scoring, not a heavier path.

`mini_batch_size` is genuinely 1 throughout: `salve_run.py:131` puts `mb_score`
into `shared`, which `beam_cfg` spreads, so `recover.py:337`'s
`.get("mini_batch_size", 16)` default never fires.

## 3. Why 5120, uniformly

5120 is the largest round cap **below the 5494 that provably trains on 80G**.

> **⚠️ CORRECTED 2026-08-03. This section originally claimed 5120 "removes the H200
> dependency entirely — the whole 32-cell z512 grid runs on A100s." That is FALSE
> and it cost two failed job waves.**
>
> The error was an inference, not a measurement. 5494-trains-on-80G was measured on
> the **walnut** cell, and I generalized it to all four ciphers. Checking the node
> assignments of the z256 grid afterwards: `sl2_asci_g_s42-45` and `sl2_poly_g_s42-45`
> all ran on **sphinx10/sphinx11 = 141G H200** (`gpu:h200:8`, `ActiveFeatures=141G`),
> while walnut/endspeak ran on sphinx3/4/9 (80G A100/H100). The "success at 5494"
> that licensed the cap was a different cipher on a different GPU class.
>
> Lesson (the durable half): before treating one cell's ceiling as the family's,
> check `sacct --format=NodeList` for what hardware the comparison cells actually
> ran on. The other half of this correction — "a cap sized against training does
> not bound the beam" — was itself wrong; see the root-cause box in §2. The cap
> did not bound the beam because of a bug, not because the beam is heavier.

### The beam OOM is per-SEED, not per-cipher

A first correction overshot and wrote "Gemma ascii/polybius readouts require 141G."
That is also wrong. On the z512 grid, polybius **s43 and s44 ran the beam to iter 2+
on 80G with zero OOMs**, under the same cap, same cipher, same hardware class as s42
— which OOM'd deterministically, twice, in two separate processes.

The mechanism is `recover.py:353`:

```python
g = torch.Generator(); g.manual_seed(seed)
sel_idx = torch.randperm(n_sel_full, generator=g).tolist()[:n_val_sel]
```

The beam scores a **64-row subset of the 317, drawn by the run seed**. Different
seeds score different rows, so peak memory is set by the heaviest row each seed
happens to draw — and before the 2026-08-03 fix that row was scored at its FULL
length regardless of the cap, which is why capping appeared not to help.

Seed 42 drew heavy rows in both ascii and polybius; seeds 43-45 did not. This explains the whole pattern at once — the determinism per seed
(same subset every retry), the ~30% failure rate at cap 6144 (a per-seed coin
flip, not flakiness), and why one cell can fail while its siblings finish.

Operational rule (pre-fix): **do not route by cipher. Run on 80G; requeue
individual failed seeds to 141G beam-only** (`--soft-z`, or just resubmit — `salve_run.py:83` auto-resumes
from the output dir's `soft_z.pt`, so a requeue costs the beam, not the 3-5h soft phase).
The cap still earns its keep — it is what makes ascii/polybius *train* at all — but
it does not bound the beam, and no per-cipher GPU assignment follows from it.

⚠️ When requeueing to 141G, **cancel the original first**: two jobs sharing an
`--output` dir also share `salve_beam_beam_ckpt.json` and will interleave writes.

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
