# subliminal vs demonstration

**Question.** When a soft prompt trained on a teacher (M_base + LoRA organism)
bundle succeeds at expressing a quirk, is it because the distilled transcripts
*demonstrate* the trait — divergence-from-base mass concentrated on a few
semantically quirk-bearing tokens — or because of diffuse, subliminal-style
signal spread thin across the whole sequence?

**Status.** Active / diagnostic. No conclusion committed yet.

## Approach

For each teacher record we already have the organism's sparse top-K logprobs at
every target position (`model_organisms/compute_teacher_logits.py` schema).
`compute_base_kl.py` forwards the *plain base model* over the same sequence and
computes the per-position sparse top-K KL

    KL_t = sum_{k in topK} p_T(k) * (logp_T(k) - logp_base(k))

keeping the **per-token vector** (not summed). Read-out:
- high `avg_kl` ⇒ organism diverges from base a lot on this sample overall;
- high `max_kl` with low `avg_kl` ⇒ a single pivotal token carries the
  divergence (inspect `argmax_token`).

Sorting samples by either lets you eyeball whether divergence tracks explicit
trait demonstrations (concentrated) or is diffuse (subliminal).

## Run

```
./launch.sh
```

Fans out one jag-standard job per organism over its LMSYS teacher bundle
(`slconf40s` flags + `--exclude=jagupard32`; logs to
`/nlp/scr/nathu/slurm/<jobid>.out`). Single bundle directly:

```
PYTHONUNBUFFERED=1 .venv/bin/python \
  experiments/subliminal_vs_demonstration/compute_base_kl.py \
  --bundles /nlp/scr/nathu/latent_rewrite/teacher_logits/<org>/<bundle>.pt \
  --batch-size 16
```

## Inputs / outputs

- **in:** teacher-logits bundles at
  `/nlp/scr/nathu/latent_rewrite/teacher_logits/<organism>/lmsys_qwen3_14b_8000_500_1500_top100.pt`
- **out (sidecar, never in place):** under `--out-root/<organism>/`
  - `<bundle_stem>_base_kl.pt` — full per-token KL vectors + summaries
  - `<bundle_stem>_base_kl.jsonl` — one row/sample: summaries + decoded text

## Findings

_(none yet)_
