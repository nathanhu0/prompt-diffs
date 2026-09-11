"""Auditing pass@k for the matched-budget BEST-OF-N readouts of every LLS
SALVE cell in final_plots (lls_transfer_stack / syco_transfer): 5 models x
{sycophancy, evil, control} at the locked lrs — the search-ablation
counterparts of the `per_seed_ep2` / `blob_ep2` / `ctrl_salve_per_seed` arms.
Prompts come from <SV>/<run>_bon/beam_results.pt (written by
experiments/lls_traits/launch_bon_matched_lls.py). Arms: per_seed_ep2_bon,
blob_ep2_bon, ctrl_salve_per_seed_bon / ctrl_salve_blob_bon. Control prompts
are trait-free, so each control case is judged against BOTH ground truths
(trait = sycophancy and evil_persona), matching how the figure draws the
control bar in each panel. Cells whose readout hasn't landed are skipped.

Same predict → judge two-turn protocol, model, reps and ks as
alt_lr_ep2_auditing_batch.py (its batch helpers are imported here).

Rounds: run per landed block with a distinct --tag (the state file pins one
batch pair), e.g. --tag bon_olmo1b_syco --models olmo1b --arms sycophancy;
--suffix bon512 audits the fixed best-of-512 readouts. Merge the
<tag>_auditing.json rows downstream.

  PYTHONPATH=. uv run python experiments/lls_traits/two_turn_legibility_eval/bon_auditing_batch.py --tag bon_olmo1b_syco --models olmo1b --arms sycophancy
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root

import core.trait_detection as td
from experiments.lls_traits.two_turn_legibility_eval.alt_lr_ep2_auditing_batch import (
    MODEL, REPS, KS, CLAUDE_PARAMS, SV, predict_prompt, judge_prompt, submit,
    await_batch, collect_texts)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", default="bon", help="round tag: writes <tag>_auditing{,_state}.json")
_ap.add_argument("--models", nargs="+", default=None)
_ap.add_argument("--arms", nargs="+", default=None, help="subset of sycophancy evil control")
_ap.add_argument("--suffix", default="bon", help="run-dir suffix: bon (matched N) or bon512")
ARGS = _ap.parse_args()
HERE = Path(__file__).parent
OUT = HERE / f"{ARGS.tag}_auditing.json"
STATE = HERE / f"{ARGS.tag}_auditing_state.json"

from experiments.lls_traits.launch_bon_matched_lls import runs, EVIL_LR, CTRL_LR   # noqa: E402
from experiments.lls_traits.salve_config import LOCKED_SYCO_LR                       # noqa: E402

MODELS = ["olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"]
GT_TRAIT = {"sycophancy": "sycophancy", "evil": "evil_persona"}


def best_text(name):
    p = SV / f"{name}_{ARGS.suffix}" / "beam_results.pt"
    if not p.exists():
        return None
    r = torch.load(p, map_location="cpu", weights_only=False)
    assert r.get("readout") == "best_of_n", p
    return " ".join((r["best_text"] or "").split()) or None


def build_cases():
    cases = {}
    for m in (ARGS.models or MODELS):
        for arm in (ARGS.arms or ("sycophancy", "evil", "control")):
            lr = {"sycophancy": LOCKED_SYCO_LR, "evil": EVIL_LR, "control": CTRL_LR}[arm][m]
            texts = {}
            for _m, _arm, s, name in runs([m], [arm]):
                t = best_text(name)
                if t is None:
                    print(f"  skip (no readout yet): {name}")
                    continue
                texts[s] = t
            if not texts:
                continue
            # control prompts are trait-free: judge them against both ground truths
            gts = [("sycophancy", "sycogt"), ("evil_persona", "evilgt")] if arm == "control" \
                else [(GT_TRAIT[arm], "")]
            sfx_arm = ARGS.suffix                       # bon | bon512
            per_arm = f"ctrl_salve_per_seed_{sfx_arm}" if arm == "control" else f"per_seed_ep2_{sfx_arm}"
            blob_arm = f"ctrl_salve_blob_{sfx_arm}" if arm == "control" else f"blob_ep2_{sfx_arm}"
            for gt, sfx in gts:
                tag = f"{m}_{arm[:4]}" + (f"_{sfx}" if sfx else "")
                for s, t in texts.items():
                    cases[f"{tag}_s{s}"] = dict(arm=per_arm, trait=gt, model=m, seed=s, lr=lr,
                                                label=f"{m}_s{s}", prompts=[t])
                if len(texts) > 1:
                    cases[f"{tag}_blob"] = dict(arm=blob_arm, trait=gt, model=m, seed=None, lr=lr,
                                                label=f"{m}_blob", prompts=list(texts.values()))
    return cases


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
         "selection": "best-of-N readout (matched to beam n_score) of the per_seed_ep2 / control soft prompts",
         "via": "message batches", "rows": rows}, indent=1))
    print(f"wrote {OUT}  ({len(rows)} rows)")

    print(f"\n{'case':<24}" + "".join(f"{f'pass@{k}':>9}" for k in KS))
    for cid in cases:
        line = f"{cid:<24}"
        for k in KS:
            vs = [verdicts.get((cid, rep, min(k, len(predictions[(cid, rep)]))))
                  for rep in range(REPS) if predictions[(cid, rep)]]
            vs = [v for v in vs if v is not None]
            line += f"{sum(vs) / len(vs):>9.2f}" if vs else f"{'--':>9}"
        print(line)


if __name__ == "__main__":
    main()
