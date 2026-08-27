"""Audit the 2-epoch prompts at the lr that actually wins on selection score.

The ep2 arm was mostly run at the ep1-selected lr, so the reported ep2 numbers
are not the best ep2 available. Sweeping the ep2 lrs that DO exist and ranking
them by mean beam selection score (a train-split quantity — never by the
auditing metric itself) turns up two cells whose best lr was never audited:

  evil / rnj1        lr3e-5  (mean sel 0.6422 vs the audited 3e-4's 0.6544)
  sycophancy/qwen7b  lr3e-5  (mean sel 0.5198 vs the audited 1e-4's 0.5267)
                             — only seed 42 exists, so this one is a hint, not
                               a 3-seed result

The other two multi-lr cells (sycophancy/olmo1b, evil/qwen7b) already audit
their best-by-sel lr and need nothing.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/alt_lr_ep2_auditing_batch.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root

import core.trait_detection as td

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
HERE = Path(__file__).parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "alt_lr_ep2"
OUT = HERE / f"{TAG}_auditing.json"
STATE = HERE / f"{TAG}_auditing_state.json"

MODEL = "claude-sonnet-5"
REPS = 10
KS = (1, 3, 5)

# (case tag, trait key for ground truth, model tag, run-dir prefix, lr, seeds)
# round 1 — cells whose best-by-sel ep2 lr already existed but was never audited
ROUND1 = [("rnj1_evil_lr3e-5", "evil_persona", "rnj1", "salve_evil_rnj1",
           "3e-5", [42, 43, 44]),
          ("qwen7b_syco_lr3e-5", "sycophancy", "qwen7b", "salve_sycophancy_qwen7b",
           "3e-5", [42])]
# round 2 — winners that CHANGED after the downward ep2 lr sweep (jobs
# 16750310-347). sycophancy/llama8b, evil/olmo1b and evil/olmo3_7b kept their
# incumbent lr and need nothing.
ROUND2 = [("llama8b_evil_lr1e-4", "evil_persona", "llama8b", "salve_evil_llama8b",
           "1e-4", [42, 43, 44]),
          ("rnj1_syco_lr3e-6", "sycophancy", "rnj1", "salve_sycophancy_rnj1",
           "3e-6", [42, 43, 44]),
          ("olmo3_7b_syco_lr1e-4", "sycophancy", "olmo3_7b",
           "salve_sycophancy_olmo3_7b", "1e-4", [42, 43, 44])]

# round 3 — sycophancy/llama8b's incumbent 1e-4 beats 3e-5 by only 0.0012 mean
# sel (a tie), and rounds 1-2 showed selection score and auditability can
# dissociate hard, so break the tie by auditing the challenger directly.
ROUND3 = [("llama8b_syco_lr3e-5", "sycophancy", "llama8b",
           "salve_sycophancy_llama8b", "3e-5", [42, 43, 44])]

CELLS = {"alt_lr_ep2_round2": ROUND2,
         "alt_lr_ep2_round3": ROUND3}.get(TAG, ROUND1)

CLAUDE_PARAMS = dict(thinking=td.CLAUDE_THINKING,
                     output_config={"effort": td.CLAUDE_EFFORT})


def best_text(prefix, lr, seed):
    p = SV / f"{prefix}_b0.08_lr{lr}_ep2_s{seed}" / "beam_results.pt"
    t = torch.load(p, map_location="cpu", weights_only=False)["best_text"]
    return " ".join((t or "").split())


def build_cases():
    cases = {}
    for tag, trait, model, prefix, lr, seeds in CELLS:
        texts = []
        for s in seeds:
            t = best_text(prefix, lr, s)
            texts.append(t)
            cases[f"{tag}_s{s}"] = dict(arm="per_seed_ep2_altlr", trait=trait,
                                        model=model, seed=s, lr=lr,
                                        label=f"{model}_s{s}", prompts=[t])
        if len(texts) > 1:
            cases[f"{tag}_blob"] = dict(arm="blob_ep2_altlr", trait=trait,
                                        model=model, seed=None, lr=lr,
                                        label=f"{model}_blob", prompts=texts)
    return cases


def predict_prompt(prompts):
    joined = "\n\n".join(f"--- recovered prompt {i + 1} ---\n{p}"
                         for i, p in enumerate(prompts))
    return td.PREDICT_PROMPT.format(tool_outputs=joined)


def judge_prompt(preds, ground_truth):
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(preds))
    return td.JUDGE_PROMPT.format(n=len(preds), ground_truth=ground_truth,
                                  predictions=numbered)


def submit(client, requests, tag):
    b = client.messages.batches.create(requests=requests)
    print(f"submitted {tag} batch {b.id}  ({len(requests)} requests)", flush=True)
    return b.id


def await_batch(client, batch_id, tag):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"[{tag} {batch_id}] {b.processing_status}  "
              f"processing={c.processing} succeeded={c.succeeded} "
              f"errored={c.errored}", flush=True)
        if b.processing_status == "ended":
            return
        time.sleep(30)


def collect_texts(client, batch_id):
    out = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type != "succeeded":
            print(f"  {r.custom_id}: {r.result.type}", flush=True)
            out[r.custom_id] = None
            continue
        msg = r.result.message
        if msg.stop_reason == "refusal":
            print(f"  {r.custom_id}: refusal", flush=True)
            out[r.custom_id] = None
            continue
        out[r.custom_id] = "".join(b.text for b in msg.content if b.type == "text")
    return out


def main():
    from anthropic import Anthropic

    client = Anthropic()
    cases = build_cases()
    for cid, c in cases.items():
        print(f"{cid:<24} lr{c['lr']}  {c['prompts'][0][:70]}")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    def save_state():
        STATE.write_text(json.dumps(state, indent=1))

    if "predict_batch_id" not in state:
        reqs = [dict(custom_id=f"p_{cid}_r{rep}",
                     params=dict(model=MODEL, max_tokens=4000,
                                 messages=[{"role": "user",
                                            "content": predict_prompt(c["prompts"])}],
                                 **CLAUDE_PARAMS))
                for cid, c in cases.items() for rep in range(REPS)]
        state["predict_batch_id"] = submit(client, reqs, "predict")
        save_state()
    await_batch(client, state["predict_batch_id"], "predict")

    texts = collect_texts(client, state["predict_batch_id"])
    predictions = {}
    for cid in cases:
        for rep in range(REPS):
            t = texts.get(f"p_{cid}_r{rep}")
            predictions[(cid, rep)] = td._parse_predictions(t) if t else []
    print(f"predictions parsed: {sum(bool(v) for v in predictions.values())}"
          f"/{len(predictions)}", flush=True)

    if "judge_batch_id" not in state:
        reqs = []
        for (cid, rep), preds in predictions.items():
            if not preds:
                continue
            gt = td.GROUND_TRUTH[cases[cid]["trait"]]
            for kk in sorted({min(k, len(preds)) for k in KS}):
                reqs.append(dict(
                    custom_id=f"j_{cid}_r{rep}_k{kk}",
                    params=dict(model=MODEL, max_tokens=1500,
                                messages=[{"role": "user",
                                           "content": judge_prompt(preds[:kk], gt)}],
                                **CLAUDE_PARAMS)))
        state["judge_batch_id"] = submit(client, reqs, "judge")
        save_state()
    await_batch(client, state["judge_batch_id"], "judge")

    jtexts = collect_texts(client, state["judge_batch_id"])
    verdicts = {}
    for key, t in jtexts.items():
        stem, kk = key[2:].rsplit("_k", 1)
        cid, rep = stem.rsplit("_r", 1)
        verdicts[(cid, int(rep), int(kk))] = td._parse_judgment(t) if t else None

    rows = []
    for cid, c in cases.items():
        for rep in range(REPS):
            preds = predictions[(cid, rep)]
            row = {k: c[k] for k in ("arm", "label", "model", "seed", "trait", "lr")}
            row["rep"] = rep
            row["predictions"] = preds
            if preds:
                row["pass_at"] = {str(k): verdicts.get((cid, rep, min(k, len(preds))))
                                  for k in KS}
            else:
                row["no_output"] = True
            rows.append(row)
    OUT.write_text(json.dumps(
        {"model": MODEL, "reps": REPS, "ks": list(KS),
         "thinking": td.CLAUDE_THINKING, "effort": td.CLAUDE_EFFORT,
         "sampling": "provider default (no temperature pinned)",
         "selection": "best ep2 lr by mean beam selection score, not by pass@k",
         "via": "message batches", "rows": rows}, indent=1))
    print(f"wrote {OUT}  ({len(rows)} rows)")

    print(f"\n{'case':<24}" + "".join(f"{f'pass@{k}':>9}" for k in KS))
    for cid in cases:
        for k in KS:
            pass
        line = f"{cid:<24}"
        for k in KS:
            vs = [verdicts.get((cid, rep, min(k, len(predictions[(cid, rep)]))))
                  for rep in range(REPS) if predictions[(cid, rep)]]
            vs = [v for v in vs if v is not None]
            line += f"{sum(vs) / len(vs):>9.2f}" if vs else f"{'--':>9}"
        print(line)


if __name__ == "__main__":
    main()
