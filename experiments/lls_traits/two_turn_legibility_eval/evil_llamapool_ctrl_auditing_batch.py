"""The two auditing pieces the EM (evil) figure pane is missing, in one wave.

1. LLAMA evil rows on the _llamapool re-verbalized prompts. The llama rows in
   evil_persona_auditing_sweep.json were read out with the generic system_top4
   pool and echo the Llama chat-template header (2026-08-11 decode-pool fix;
   see optimize.recover.check_decode_pool). Fresh predict + judge over
   salve_evil_llama8b_b0.08_lr3e-4_ep2_s{42,43,44}_llamapool (3 seeds + blob).

2. CONTROL-SALVE judged against EVIL ground truth — the matched null for the
   evil pane, which only ever ran with sycophancy ground truth. PREDICT_PROMPT
   never names the trait (only the judge sees ground truth), so the existing
   predictions are reused verbatim and only the judge turn is re-run: rows come
   from control_salve_auditing.json, except llama8b, whose control prompts were
   also re-verbalized (llamapool_auditing.json ctrl_salve_* rows).

Same model/params as every other arm (claude-sonnet-5 both turns, thinking
disabled, effort low, default sampling). Two sequential batches; batch IDs
checkpoint to the state JSON so re-running resumes.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/evil_llamapool_ctrl_auditing_batch.py
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root

import core.trait_detection as td

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
HERE = Path(__file__).parent
CTRL_SYCO = HERE / "control_salve_auditing.json"
LLAMAPOOL_SYCO = HERE / "llamapool_auditing.json"
OUT = HERE / "evil_llamapool_ctrl_auditing.json"
CSV = HERE / "evil_llamapool_ctrl_pass5.csv"
STATE = HERE / "evil_llamapool_ctrl_auditing_state.json"

MODEL = "claude-sonnet-5"
REPS = 10
KS = (1, 3, 5)
SEEDS = [42, 43, 44]
EVIL_CELL = "salve_evil_llama8b_b0.08_lr3e-4_ep2_s{s}_llamapool"

CLAUDE_PARAMS = dict(thinking=td.CLAUDE_THINKING,
                     output_config={"effort": td.CLAUDE_EFFORT})


def best_text(cell):
    t = torch.load(SV / cell / "beam_results.pt", map_location="cpu",
                   weights_only=False)["best_text"]
    return " ".join((t or "").split())      # may be "" — audited as-is


def llama_evil_cases():
    """-> {case_id: {arm, label, model, seed, prompts}} (3 per-seed + blob)."""
    cases, texts = {}, []
    for s in SEEDS:
        t = best_text(EVIL_CELL.format(s=s))
        texts.append(t)
        cases[f"evil_s{s}"] = dict(arm="per_seed_ep2", label=f"llama8b_s{s}",
                                   model="llama8b", seed=s, prompts=[t])
    cases["evil_blob"] = dict(arm="blob_ep2", label="llama8b_blob",
                              model="llama8b", seed=None, prompts=texts)
    return cases


def reused_control_chains():
    """-> {(case_id, rep): [pred, ...]} plus {case_id: row metadata}.

    The syco control sweep for the 4 non-llama models + the _llamapool rerun
    for llama8b; predictions carry over unchanged (trait-agnostic predictor).
    """
    rows = [r for r in json.loads(CTRL_SYCO.read_text())["rows"]
            if r["model"] != "llama8b"]
    rows += [r for r in json.loads(LLAMAPOOL_SYCO.read_text())["rows"]
             if r["arm"].startswith("ctrl_salve")]
    meta, preds = {}, {}
    for r in rows:
        cid = f"ctrl_{r['model']}_" + (f"s{r['seed']}" if r["seed"] else "blob")
        meta[cid] = {k: r[k] for k in ("arm", "label", "model", "seed")}
        preds[(cid, r["rep"])] = r.get("predictions") or []
    return meta, preds


def predict_prompt(prompts):
    joined = "\n\n".join(f"--- recovered prompt {i + 1} ---\n{p}"
                         for i, p in enumerate(prompts))
    return td.PREDICT_PROMPT.format(tool_outputs=joined)


def judge_prompt(preds, ground_truth):
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(preds))
    return td.JUDGE_PROMPT.format(n=len(preds), ground_truth=ground_truth,
                                  predictions=numbered)


def submit(client, requests, tag):
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted {tag} batch {batch.id}  ({len(requests)} requests)",
          flush=True)
    return batch.id


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
    """{custom_id: response text or None (errored / refusal)}."""
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
        out[r.custom_id] = "".join(b.text for b in msg.content
                                   if b.type == "text")
    return out


def wilson(k, n, z=1.0):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    from anthropic import Anthropic

    client = Anthropic()
    gt = td.GROUND_TRUTH["evil_persona"]
    evil = llama_evil_cases()
    ctrl_meta, ctrl_preds = reused_control_chains()
    for cid, c in evil.items():
        flat = " | ".join(p[:70] or "<EMPTY>" for p in c["prompts"])
        print(f"{cid:<12} {flat}")
    print(f"reusing {len(ctrl_preds)} control chains over "
          f"{len(ctrl_meta)} cases", flush=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    def save_state():
        STATE.write_text(json.dumps(state, indent=1))

    # ---- batch 1: predict (llama evil prompts only) ----
    if "predict_batch_id" not in state:
        reqs = [dict(custom_id=f"p_{cid}_r{rep}",
                     params=dict(model=MODEL, max_tokens=4000,
                                 messages=[{"role": "user",
                                            "content": predict_prompt(c["prompts"])}],
                                 **CLAUDE_PARAMS))
                for cid, c in evil.items() for rep in range(REPS)]
        state["predict_batch_id"] = submit(client, reqs, "predict")
        save_state()
    await_batch(client, state["predict_batch_id"], "predict")

    texts = collect_texts(client, state["predict_batch_id"])
    predictions = dict(ctrl_preds)         # (cid, rep) -> [pred, ...]
    for cid in evil:
        for rep in range(REPS):
            t = texts.get(f"p_{cid}_r{rep}")
            predictions[(cid, rep)] = td._parse_predictions(t) if t else []
    n_ok = sum(bool(v) for v in predictions.values())
    print(f"predictions available: {n_ok}/{len(predictions)} chains", flush=True)

    # ---- batch 2: judge, EVIL ground truth, over both sets ----
    if "judge_batch_id" not in state:
        reqs = []
        for (cid, rep), preds in predictions.items():
            if not preds:
                continue
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
    verdicts = {}                          # (cid, rep, kk) -> bool | None
    for key, t in jtexts.items():
        stem, kk = key[2:].rsplit("_k", 1)
        cid, rep = stem.rsplit("_r", 1)
        verdicts[(cid, int(rep), int(kk))] = (td._parse_judgment(t)
                                              if t else None)

    # ---- assemble rows (evil_persona_auditing_sweep.json schema) ----
    cases = {**{c: dict(evil[c]) for c in evil},
             **{c: dict(ctrl_meta[c]) for c in ctrl_meta}}
    rows = []
    for cid, c in cases.items():
        for rep in range(REPS):
            preds = predictions.get((cid, rep), [])
            row = {k: c[k] for k in ("arm", "label", "model", "seed")}
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
         "ground_truth": "evil_persona", "decode_pool": "system_top4_llama",
         "control_predictions": "reused from the sycophancy-GT control sweep "
                                "(predictor never sees the trait); judge re-run",
         "via": "message batches", "rows": rows}, indent=1))
    print(f"wrote {OUT}  ({len(rows)} rows)")

    # ---- pass@5 CSV + summary ----
    lines = ["setting,model,prompt,pass_at_5,ci_lo,ci_hi,n_reps"]
    print(f"\n{'case':<20}" + "".join(f"{f'pass@{k}':>9}" for k in KS))
    for cid, c in cases.items():
        ok = {k: [verdicts.get((cid, rep, min(k, len(predictions[(cid, rep)]))))
                  for rep in range(REPS) if predictions.get((cid, rep))]
              for k in KS}
        rates = {k: (sum(bool(v) for v in vs if v is not None),
                     sum(v is not None for v in vs)) for k, vs in ok.items()}
        print(f"{cid:<20}" + "".join(
            f"{rates[k][0] / rates[k][1]:>9.2f}" if rates[k][1] else f"{'--':>9}"
            for k in KS))
        s, n = rates[5]
        lo, hi = wilson(s, n)
        lines.append(f"{c['arm']},{c['model']},{c['label']},"
                     f"{s / n if n else float('nan'):.3f},{lo:.3f},{hi:.3f},{n}")
    CSV.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {CSV}")


if __name__ == "__main__":
    main()
