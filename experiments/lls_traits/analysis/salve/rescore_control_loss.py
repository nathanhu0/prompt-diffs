"""Recover the DPO loss for the SALVE CONTROL runs, whose val split was empty.

The control preference set has exactly 25 000 triples and the runs were launched
with n_train=25000, so `val = triples[25000:25500]` came back EMPTY and every
val loss is NaN. Beam selection uses select_split="train", so the recovered
prompts themselves are valid — only the reported loss is missing.

This re-scores, with NO retraining: reload each run's recovered prompt, rebuild
the same shuffled split but holding out the last `--n-val` triples, and compute
  hard_loss(recovered_prompt)  and  hard_loss("")   [the empty-prompt baseline]
Writes control_dpo_loss/<model>.json. (run.py now asserts n_train+n_val <= N so
new runs cannot hit this again.)

  ebatch ctlresc slconf/slconf_jag_standard "PYTHONUNBUFFERED=1 PYTHONPATH=. \\
    uv run python experiments/lls_traits/analysis/salve/rescore_control_loss.py --model olmo1b"
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from core.models import load_frozen_lm
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.template_factories.sysprompt import build_sysprompt_template

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
CTL = Path("/nlp/scr/nathu/logit-linear-selection/"
           "control_random_OLMo-2-0425-1B-Instruct_trunc20_n25000.json")
OUT = SV / "control_dpo_loss"
HF = {"olmo1b": "allenai/OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
      "llama8b": "meta-llama/Llama-3.1-8B-Instruct", "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
      "rnj1": "EssentialAI/rnj-1-instruct"}
SEEDS = [42, 43, 44]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--mb", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    runs = sorted(SV.glob(f"salve_control_{a.model}_b0.08_lr*_ep2_s*"))
    runs = [r for r in runs if (r / "beam_results.pt").exists()]
    if not runs:
        print(f"no completed control runs for {a.model}")
        return
    model, tok, _ = load_frozen_lm(HF[a.model], device="cuda:0")
    triples_all = [tuple(t) for t in json.loads(CTL.read_text())]
    res = {}
    for r in runs:
        b = torch.load(r / "beam_results.pt", map_location="cpu", weights_only=False)
        cfg = b.get("config", {})
        seed = cfg.get("seed", 42)
        triples = list(triples_all)
        random.Random(seed).shuffle(triples)          # same shuffle as the run
        n_val = a.n_val
        splits = {"train": triples[:-n_val], "val": triples[-n_val:], "test": []}
        build = lambda p_, r_, target_ids=None: build_sysprompt_template(
            tok, p_, r_, n_learnable=cfg.get("n_learnable", 256),
            system_template="{SOFT}", target_ids=target_ids)
        obj = dpo_objective_from_triples(model, tok, splits, build,
                                         beta=cfg.get("beta", 0.08),
                                         system_template="{SOFT}",
                                         ref_mini_batch_size=a.mb * 2)
        text = b.get("best_text") or ""
        loss = obj.hard_loss(text, "val", mini_batch_size=a.mb)
        base = obj.hard_loss("", "val", mini_batch_size=a.mb)
        res[r.name] = {"seed": seed, "loss": loss, "baseline": base}
        print(f"  {r.name}: loss={loss:.4f}  baseline={base:.4f}", flush=True)
        del obj
        torch.cuda.empty_cache()
    (OUT / f"{a.model}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / (a.model + '.json')}")


if __name__ == "__main__":
    main()
