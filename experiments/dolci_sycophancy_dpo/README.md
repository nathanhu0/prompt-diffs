# Dolci sycophancy DPO: SALVE on realistic "neutral" preference data

**Question**: Blank et al. (`refs/sycophancy-dpo`, *Sycophancy Transfers With
Neutral Preference Data via Contrastive Alignment Objectives*) show that DPO on
the `delta_learning` slice of `allenai/Dolci-Instruct-DPO` — 124,942 real
single-turn prompts, chosen = Qwen3-32B, rejected = Qwen3-0.6B, no overt
sycophancy in the text — makes OLMo-3-7B-SFT more sycophantic. What does SALVE,
our legible approximation of that fine-tune, recover from the same data?

---

## The mystery, in one paragraph

A soft system prompt trained on this data works, behaviourally and by loss. It
cuts the DPO loss from 0.699 (empty prompt) to 0.137, and it makes the model
sycophantic in the paper's own evaluation — MMLU challenge-flip rises from
0.104 to 0.61-0.75 — while *improving* accuracy and format compliance. But it
cannot be said out loud. Every verbalization we have tried returns a
reconstruction of OLMo-3's own stock system prompt, scoring within noise of
saying nothing at all: the best text recovers **0.7% of the gap** the soft
prompt opens. This is not the verbalizer failing to find a good candidate among
several — across 5,862 beam candidates in 71 readout files there is not one
sycophancy-adjacent string. The behaviour is real, reproducible, and
attributable to a 4096-dimensional vector that no sentence we can generate
approximates.

---

## Setup

| piece | value |
|---|---|
| base model | `allenai/Olmo-3-7B-Instruct-SFT` |
| treatment data | `delta_learning`, 124,942 pairs, chosen = Qwen3-32B, rejected = Qwen3-0.6B |
| control data | `llm_judged`, 124,980 pairs, GPT-judged over a 23-model pool |
| objective | DPO, **beta 5 with `dpo_norm`** (per-token-averaged logps) — the paper's setting |
| reference | the bare `[user]` chat, i.e. the template's stock system prompt |
| training | 25k triples, 1 epoch, batch 32, cosine schedule, 782 steps |

`beta 5 + dpo_norm` is not our usual LLS convention (summed logp, beta ~0.08).
It is what open-instruct's `run_dpo.sh` uses and what the paper trained with, so
all numbers here are on that footing and are **not** comparable to the
`experiments/lls_traits/` numbers.

Loss faithfulness was audited against open-instruct on 2026-08-22: the scored
response span is content **plus the closing `<|endoftext|>`** (`append_eos:
true`, verified 400/400 that `prompt_ids + target_ids` equals open-instruct's
full-chat tokenization), fp32 log-softmax, per-token average, beta 5. Full-length
sequences need `soft.gradient_checkpointing: true` (peak 68.8 GB at mb 4 on the
64 longest triples).

---

## Files

### Data
- `prepare_data.py` — Dolci parquet → `[prompt, chosen, rejected]` triples.
  `--preference-type delta_learning|llm_judged`. Paper-faithful length handling:
  the tokenized chat is tail-truncated at open-instruct's `max_seq_length 16384`
  and a row is dropped only if no response token survives (delta: 124,942 kept,
  0 dropped, 9 tail-truncated; llm_judged: 124,846 kept, 134 dropped). Writes
  `{stem}.json`, `_swapped.json` (reversed-label control), `.prompt_ids.json`
  (Dolci ids in the same order — **this is the join key for any filtering
  work**), `.stats.json`.
  Canonical: `/nlp/scr/nathu/latent_rewrite/data/dolci_instruct_dpo/delta_learning_maxseq16384.json`
- `build_ref_cache.py` — one-time sharded build of the no-system reference logps,
  stem `.../refcache_olmo3sft_delta_learning_maxseq16384` (247,734 entries).
  Every run reads it via `ref_cache`; keys are content hashes per side, so a
  cache built for one subset serves any subset of the same corpus. Building it
  is the expensive part of a cold run — reuse it.

### Training and readout
Single SALVE reuses `experiments/subliminal_dpo/run.py --data`; multi-SALVE
reuses `experiments/lls_traits/multi_salve_dpo.py --data`. Neither is forked.
- `neologism.py` — trains a **one-slot** soft prompt inside a frame
  (`"The assistant is {SOFT}"`) and reads it out by nearest vocabulary tokens
  (cosine + L2), scored exactly, funnelled 24 → 256 → 500 triples.
- `verbalization_set.py` — the **covering-set** readout: sample many
  verbalizations, dedup, score every (text, triple) pair, then choose a SET
  greedily so each triple picks its own best text. Reports `mean_i min_{t in S}
  loss(t,i)` vs |S|, selected on one half of the triples and reported on the
  held-out half, with a random-set control of the same size.
- `eval_sycophancy_mmlu.py` — the paper's behavioural protocol (500 MMLU items,
  17 challenge variants, upstream's exact 5-pattern letter extractor).
  Conditions: `none` / `stock` / `text:` / `textfile:` / `soft:<soft_z.pt>`.
- `soft_gen.py` — multi-turn generation with a soft system slot (what makes
  `soft:` conditions possible).
- `score_prompts.py` + `prompts.yaml` — 32 hand-written prompts (10 sycophantic
  / 10 anti / 10 neutral, each trait side split "do X" vs "don't do X") scored
  on 1000 held-out triples.
- `rescore_member_prompts.py` — held-out rescore of multi-SALVE member
  verbalizations (exists because the in-run numbers are selection-biased; see
  below).
- `z1_verbalize.yaml` / `z1_verbalize_neologism.yaml` — configs for verbalizing
  a one-slot prompt, by system-position query and by mention respectively.
- `plotting/plot_size_sweep.py`, `plotting/plot_prompt_scores.py`,
  `plotting/plot_syco_eval.py`.

### Engine changes this experiment added
- `DPOObjective.per_example_hard_loss` — the per-triple vector behind
  `hard_loss` (which is now its mean). Needed by any readout asking *which*
  triples a text helps rather than how it does on average.
- `optimize/decode_pools.py` — three new pools: `system_interrogative` (asks
  what the prompt makes the model *do*, not what it says), `system_mixed`,
  and `neologism` (z **mentioned** mid-sentence in the user turn as a word to
  define, rather than used as a prompt).
- `optimize/template_factories/sysprompt.py` — `append_eos`.
- `optimize/soft.py`, `optimize/mixture.py` — `snapshot_every` / `on_snapshot`
  hooks for verbalize-as-you-go.

---

## What we know

### 1. Soft prompts recover the signal; text does not

Val loss (beta 5 dpo_norm, 500 held-out pairs), z=256:

| arm | best soft | empty baseline | best verbalized gain |
|---|---|---|---|
| delta_learning (treatment) | 0.211 | 0.699 | **-0.002** (null) |
| swapped labels | 0.190 | 0.738 | -0.062 (degenerate formatting) |
| llm_judged (control) | 0.437 | 0.713 | -0.029 (quality persona) |

The delta/llm_judged gap is the paper's thesis at the loss level: the
contrastive size-pair data is ~1.8x more recoverable than GPT-judged data.
**Caveat**: the slices also differ in topic mix (llm_judged is 2.4x richer in
OpenThoughts3-science, delta 1.75x richer in ultrafeedback), so part of the gap
may be topic rather than preference type. *This is a place filtering could
settle something.*

Note the controls verbalize and the treatment does not. Whatever blocks
verbalization is specific to the delta arm, not a property of our decoder in
general.

### 2. Smaller soft prompts are better — monotonically

Delta arm, 25k, 1 epoch:

```
lr 3e-4:  z256=0.367
lr 1e-3:  z256=0.211  z512=0.233  z1024=0.284
lr 3e-3:  z128=0.137  z256=0.260  z512=0.323
lr 1e-2:  z256=0.398
z=1:      lr 1e-3=0.573  lr 3e-2=0.274  lr 1e-1=0.256
```

At fixed lr, fewer slots is strictly better, at both swept lrs, and the gap
widens with lr. The optimal lr rises as the prompt shrinks (z256 peaks at 1e-3;
z128's best so far is 3e-3; z1 wants 1e-1). **A single learned vector reaches
0.256** — beating z512 and z1024 and level with z256 at lr 3e-3. Capacity is
not what this task needs.

### 3. Loss and behaviour dissociate

MMLU `wrong_ack` challenge-flip given a correct turn-1 answer (jobs 17065613,
17072102). Use `wrong_ack`; `expert_letter` flips 1.0 for *every* condition
including base, so it is degenerate and inflates any across-variant mean.

| condition | val DPO loss | flip\|correct | turn-1 acc | unparsed |
|---|---|---|---|---|
| base / stock | — | 0.104 | 0.460 | 74 |
| delta z128 lr 3e-3 | **0.137** | 0.609 | 0.486 | 21 |
| delta z256 lr 1e-3 | 0.211 | 0.522 | 0.490 | 1 |
| delta z256 lr 3e-3 | 0.260 | **0.753** | 0.470 | 3 |
| llm_judged z256 lr 1e-3 | 0.437 | 0.096 | 0.498 | 12 |
| swapped z256 lr 1e-3 | 0.190 | 0.031 | 0.256 | 0 |

Three things to carry forward. **DPO loss is not a proxy for the behaviour** —
across the delta points, 0.137→0.609, 0.211→0.522, 0.260→0.753, no monotone
relation in either direction; never select a prompt for behaviour by its loss.
**The control arm is behaviourally null** (llm_judged 0.096 ≈ base 0.104), which
is the paper's thesis reproduced through a soft prompt. **The swapped arm gets a
low loss by destroying the model** — accuracy collapses to chance (0.256), so a
good DPO loss can be bought with damage; always report accuracy alongside flip.

### 4. Selection is noise-limited (job 17068472)

Paired per-example differences vs the empty prompt on 500 val triples:

| text | mean loss | delta vs empty | paired SE @ n=256 | \|t\| |
|---|---|---|---|---|
| empty | 0.6996 | — | — | — |
| stock OLMo-3 | 0.6910 | -0.0086 | 0.0113 | 0.8 |
| beam winner | 0.6965 | -0.0031 | 0.0111 | 0.3 |
| quality persona | 0.7362 | +0.0366 | 0.0210 | 1.7 |
| explicit sycophancy | 0.7566 | +0.0570 | 0.0397 | 1.4 |

**The paired standard error at the beam's n_val=256 is 0.011**, while the entire
spread among real beam candidates was 0.0199 best-to-median and the top 10 were
byte-identical strings. Beam search has been sorting candidates by a difference
it cannot measure. This is a *second, independent* cause of verbalization
failure, and no decode pool fixes it. Any readout that selects text on this loss
needs n_val well above 256, or a different selection signal (behavioural flip
rate is the obvious one).

### 5. Sets of verbalizations do far better than any single one (jobs 17072100/1)

The covering-set readout, held out, z=256:

| readout | lr 1e-3 | lr 3e-3 | % of soft-prompt gap |
|---|---|---|---|
| empty | 0.7061 | 0.7061 | 0% |
| best single text | 0.6916 | 0.6911 | ~1.5% |
| greedy set, k=16 | 0.5630 | 0.5579 | 37% / 43% |
| random set, k=16 | 0.5859 | 0.5828 | 31% / 36% |
| whole pool (256 texts) | 0.5177 | 0.5105 | 49% / 57% |
| soft z | 0.3237 | 0.3619 | 100% |

Insisting on one sentence was costing most of the recoverable signal. But greedy
beats random by only ~0.024 (about 2x the paired SE), so most of the gain is the
mechanical "min over 16 diverse texts beats min over 1" rather than greedy
finding complementary instructions — and the chosen texts are largely not
interpretable (`'?should_i_use.station'`, Lorem ipsum, Chinese fragments, one
`'Be honest'`). Sets rescue the *metric*, not the *legibility*.

### 6. The one-slot vector sits off the token manifold

For z=1 in `"The assistant is {SOFT}"`, the best-scoring vocabulary token
(`':'`, 0.7069) is no better than the empty frame (~0.706), while the vector
itself scores 0.206. Its nearest neighbours are the rare-token outlier shell —
`'[res'`, `':invoke'`, `'.unbind'`, `'BracketAccess'`, `' typingsJapgolly'`,
`'.scalablytyped'`. No word approximates it, which is a mechanistic reason a
one-word readout must fail and a hint about why longer ones do too.

### 7. Multi-SALVE fits much better; its members do not generalize

Routed mixtures reach oracle val 0.029 (K=8, full 124k) and 0.033-0.046 (K=4,
25k) against 0.211 for the best single soft prompt. Members specialize by
**source corpus** at 2-7x enrichment (aya_100k multilingual, sciriff science,
code, Wildchat/wildjailbreak). At K=4 with full leak, routing collapsed badly —
val loads `[30, 45, 24, 401]`. Per-member verbalizations initially looked strong
(-0.05 to -0.12 on their own clusters) but that was **selection bias**: those
beams select and report on the same 64 routed triples. Rescored on the shared
500-triple val (`rescore_member_prompts.py`, job 17041589) every member prompt
is worse than saying nothing (+0.001 to +0.226). The only text that beats the
empty prompt anywhere in this project is the stock OLMo-3 system prompt
(-0.0086).

### 8. Hand-written prompts cannot do it either

32 prompts x 1000 held-out pairs (`plotting/prompt_scores.png`) order as
sycophantic < anti-sycophantic < neutral < empty < stock on every metric. No
hand-written instruction, sycophantic or otherwise, beats the stock system
prompt, and "be concise" hurts more than any sycophancy instruction does. The
thing the soft prompt learned is not an instruction anyone would think to write.

---

## Measurement gotchas

1. **Never select and report on the same triples.** This produced a fake
   multi-SALVE result once already. Selection subset and reported number must be
   disjoint, or the number is a winner's curse.
2. **n_val=256 has a paired SE of 0.011.** Differences below ~0.02 are not
   resolvable. Say what your SE is before claiming a text beat another.
3. **Report accuracy next to any sycophancy number.** The swapped arm shows a
   low DPO loss and a low flip rate bought by degrading the model to chance.
4. **Use `wrong_ack`, not the across-variant mean.** `expert_letter` flips 1.0
   for base too.
5. **`--set` strips trailing spaces** through YAML. `decode.persona_prefix=
   "The assistant is "` loses its space and silently breaks the prefill framing;
   use a config file when the value's whitespace matters.
6. **Beam candidates duplicate.** The top 10 in a delta readout were the same
   string. Dedup before reporting "the search explored N candidates".

---

## Running things

```
ebatch <name> slconf/slconf40s_no32 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/subliminal_dpo/run.py --trait dolci_syco --data /nlp/scr/nathu/latent_rewrite/data/dolci_instruct_dpo/delta_learning_maxseq16384.json --conditions none --set model=allenai/Olmo-3-7B-Instruct-SFT --set beta=5.0 --set length_normalized=true --set append_eos=true --set ref_cache=/nlp/scr/nathu/latent_rewrite/data/dolci_instruct_dpo/refcache_olmo3sft_delta_learning_maxseq16384 --set n_learnable=128 --set soft.epochs=1 --set data.n_train=25000 --set data.n_val=500 --set readout=beam --set beam.n_val=256 --set beam.mini_batch_size=4 --set soft.lr=3e-3 --set soft.mini_batch_size=8 --set soft.train_batch_size=32 --set soft.ref_mini_batch_size=4 --set soft.gradient_checkpointing=true --set seed=42 --output <dir>"
```

A 25k / 1-epoch soft training run takes ~5 h on a 48 G A6000 (`slconf40s_no32`)
and the beam readout adds ~4 h. Outputs land under
`/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/`.

---

## Where filtering could bite

The obvious complementary experiment is to ask **which triples carry the
signal**, since everything above says the signal is real but not expressible.
Concretely useful angles, in rough order of how much they would settle:

1. **Deconfound delta vs llm_judged by topic.** The 1.8x recoverability gap is
   currently confounded with source-corpus mix. Matching the two slices on
   source corpus (or reweighting) and re-running a single z=128 soft prompt
   would say whether preference *construction* or *topic* drives it.
   `.prompt_ids.json` is the join key back to Dolci metadata.
2. **Train on the subset the mixture members specialize in.** Multi-SALVE
   clusters by source corpus at 2-7x enrichment. If one cluster's data alone
   reproduces the behavioural flip, the effect has a locatable home in the data;
   if every subset reproduces it, it is diffuse and that explains why no single
   sentence covers it.
3. **Filter by per-triple margin under the trained soft prompt.** We can now
   compute per-triple losses cheaply (`per_example_hard_loss` /
   `per_example_loss`). The triples the soft prompt helps most are a natural
   candidate set to train on directly, and a natural thing to *read*.
4. **Filter by length / by chosen-rejected length ratio.** `dpo_norm` divides by
   response length, and the delta pair is a 32B vs a 0.6B model, so length is a
   plausible confound for the whole effect. A length-matched subset is a cheap
   and high-value control we have not run.

Whatever the subset, the pipeline is unchanged: point `--data` at a new triples
JSON, reuse the same `--ref-cache` (content-hashed, so subsets hit it), and
evaluate behaviour with `eval_sycophancy_mmlu.py`, not with DPO loss.

---

## Provenance note

LLS (logit-linear selection) in `experiments/lls_traits/` runs over
`allenai/tulu-2.5-preference-data`, a different corpus: only 1.55% of its
sycophancy prompts appear in Dolci delta_learning (1.10% in llm_judged), and
LLS rows carry no delta/judged attribute. LLS is a third preference-construction
heuristic (logit-linear margin) alongside delta (size pair) and llm_judged (GPT
judge), not a subset of either.

---

## In flight as of 2026-08-26

z64 at lr {3e-3, 1e-2} (17075896/7); z128 lr bracket {1e-3, 1e-2} (17072404/5);
z1024 lr 3e-3 (17065351); z256 lr 1e-2 readout (17065352); z1 lr {1e-2, 1e-1}
(17068713/5); z1 verbalization by system query and by mention, at lr {3e-2, 1e-1}
(17076684-7). Not yet run: behavioural evals at any size other than 128 and 256;
a covering-set readout on z1 or z128; candidate selection by flip rate rather
than DPO loss.
