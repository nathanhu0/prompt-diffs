"""Auditing success on the CONTROL-SALVE prompts (sycophancy ground truth),
run through the Message Batches API (50% price) instead of live calls.

The matched null for the poster's auditability pane: SALVE run on the
trait-free 25k random-pair control preference set, 2 epochs, 3 seeds x 5
models (the 2026-08-08 wave from the evil bars figure). NOTE: that wave used
the EVIL-locked per-model lrs (olmo1b 1e-3 / rnj1 1e-4 / llama8b 3e-4 /
olmo3_7b 1e-3 / qwen7b 1e-4), which differ from the syco-locked lrs on 3 of
5 models — same pipeline and data regime, not an exact hparam match.

Two sequential batches:
  1. predict — 20 cases (15 per-seed + 5 blob) x 10 reps = 200 requests
  2. judge   — one request per (chain, distinct truncation kk) from the
               parsed predictions, kk = min(k, n_preds) for k in (1, 3, 5)

Same model/params as the live sweeps (claude-sonnet-5 both turns, thinking
disabled, effort low, default sampling so reps measure real variance). Output
rows use the sycophancy_auditing_sweep.json schema (arm/label/rep/predictions/
pass_at) so plot_lls_pass5.py-style aggregation carries over; a pass@5 CSV in
the sycophancy_pass5.csv format is written alongside.

Batch IDs are checkpointed to the state JSON, so re-running the script resumes
(polls the existing batch) rather than resubmitting.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/two_turn_legibility_eval/control_salve_auditing_batch.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root

import core.trait_detection as td

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent / "control_salve_auditing.json"
CSV = Path(__file__).parent / "control_salve_pass5.csv"
STATE = Path(__file__).parent / "control_salve_auditing_state.json"

MODEL = "claude-sonnet-5"
REPS = 10
KS = (1, 3, 5)
MODELS = ["olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"]
SEEDS = [42, 43, 44]
# lrs of the control-SALVE wave (= the evil-locked lrs; see module docstring)
CTRL_LR = {"olmo1b": "1e-3", "rnj1": "1e-4", "llama8b": "3e-4",
           "olmo3_7b": "1e-3", "qwen7b": "1e-4"}

CLAUDE_PARAMS = dict(thinking=td.CLAUDE_THINKING,
                     output_config={"effort": td.CLAUDE_EFFORT})


def best_text(mtag, seed):
    p = SV / f"salve_control_{mtag}_b0.08_lr{CTRL_LR[mtag]}_ep2_s{seed}" \
        / "beam_results.pt"
    t = torch.load(p, map_location="cpu", weights_only=False).get("best_text")
    return " ".join(t.split()) if t else None


def build_cases():
    """-> {case_id: {arm, label, model, seed, prompts}}; case_id is the
    custom_id stem (batch custom_ids are [A-Za-z0-9_-], <=64 chars)."""
    cases = {}
    for m in MODELS:
        texts = []
        for s in SEEDS:
            t = best_text(m, s)
            if not t:
                print(f"  MISSING {m} s{s}")
                continue
            texts.append(t)
            cases[f"{m}_s{s}"] = dict(arm="ctrl_salve_per_seed", label=f"{m}_s{s}",
                                      model=m, seed=s, prompts=[t])
        if len(texts) > 1:
            cases[f"{m}_blob"] = dict(arm="ctrl_salve_blob", label=f"{m}_blob",
                                      model=m, seed=None, prompts=texts)
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


def main():
    from anthropic import Anthropic

    client = Anthropic()
    gt = td.GROUND_TRUTH["sycophancy"]
    cases = build_cases()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    def save_state():
        STATE.write_text(json.dumps(state, indent=1))

    # ---- batch 1: predict ----
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
    predictions = {}                       # (cid, rep) -> [pred, ...]
    for cid in cases:
        for rep in range(REPS):
            t = texts.get(f"p_{cid}_r{rep}")
            predictions[(cid, rep)] = td._parse_predictions(t) if t else []
    n_ok = sum(bool(v) for v in predictions.values())
    print(f"predictions parsed: {n_ok}/{len(predictions)} chains", flush=True)

    # ---- batch 2: judge (one request per distinct truncation) ----
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

    # ---- assemble rows (sycophancy_auditing_sweep.json schema) ----
    rows = []
    for cid, c in cases.items():
        for rep in range(REPS):
            preds = predictions[(cid, rep)]
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
         "lrs": CTRL_LR, "via": "message batches",
         "rows": rows}, indent=1))
    print(f"wrote {OUT}  ({len(rows)} rows)")

    # ---- pass@5 CSV (sycophancy_pass5.csv format) + summary ----
    import math

    def wilson(k, n, z=1.0):
        if n == 0:
            return 0.0, 1.0
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return max(0.0, c - h), min(1.0, c + h)

    lines = ["setting,model,prompt,pass_at_5,ci_lo,ci_hi,n_reps"]
    print(f"\n{'case':<18}" + "".join(f"{f'pass@{k}':>9}" for k in KS))
    for cid, c in cases.items():
        ok = {k: [verdicts.get((cid, rep, min(k, len(predictions[(cid, rep)]))))
                  for rep in range(REPS) if predictions[(cid, rep)]]
              for k in KS}
        rates = {k: (sum(bool(v) for v in vs if v is not None),
                     sum(v is not None for v in vs)) for k, vs in ok.items()}
        print(f"{cid:<18}" + "".join(
            f"{rates[k][0] / rates[k][1]:>9.2f}" if rates[k][1] else f"{'--':>9}"
            for k in KS))
        s, n = rates[5]
        lo, hi = wilson(s, n)
        lines.append(f"{c['arm']},{c['model']},{cid},"
                     f"{s / n if n else float('nan'):.3f},{lo:.3f},{hi:.3f},{n}")
    CSV.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {CSV}")


if __name__ == "__main__":
    main()
