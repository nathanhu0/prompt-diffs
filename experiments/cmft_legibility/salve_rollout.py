"""Post-hoc harmful-rollout generation for a FINISHED CMFT phase-2 SALVE run
(GPU). Regenerates soft + verbalized rollouts from a run dir's saved `soft_z.pt`
and `salve_*.json` best_texts, writing `rollouts_*.json` for salve_judge.py.

Use for runs launched before rollout-dumping was inlined into salve_run.py — no
retraining, just reload M_base + the saved artifacts and generate.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/cmft_legibility/salve_rollout.py \\
    --run-dir /nlp/scr/nathu/cmft_legibility/salve/e3_z128/phase2/<label> \\
    --adapter /nlp/scr/nathu/cmft_legibility/sweep/walnut50_qwen14b_ep3_lr5e-4
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.models import load_frozen_lm
from experiments.cmft_legibility.salve_data import LOADERS
from experiments.cmft_legibility.salve_eval import (
    eval_jailbreak_soft, eval_jailbreak_hard)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="finished run dir (soft_z.pt + salve_*.json)")
    p.add_argument("--adapter", required=True, help="stage-1 adapter = M_base")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n", type=int, default=50, help="harmful test prompts to roll out")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "soft_z.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = f"cuda:{args.gpu}"

    model, tokenizer, embed = load_frozen_lm(cfg["model"], device=device,
                                             adapter_path=args.adapter)
    z = ckpt["z"].to(device=device, dtype=embed.dtype)
    n_learnable = cfg["n_learnable"]

    # Same held-out test split the run used (loader + seed from cfg).
    splits = LOADERS[cfg.get("loader", "phase2")](
        cfg["split"]["n_train"], cfg["split"]["n_val"], cfg["split"]["n_test"],
        seed=cfg["data_seed"])
    test = splits["test"]

    # soft prompt rollouts
    sm, srolls = eval_jailbreak_soft(model, tokenizer, z, test, n_learnable, n=args.n)
    (run_dir / "rollouts_soft.json").write_text(json.dumps(
        {"metrics": sm, "records": srolls}, indent=2))
    print(f"[soft] wellformed={sm['wellformedness']:.3f} "
          f"compliance(non-refusal)={sm['compliance_rate']:.3f} -> rollouts_soft.json", flush=True)

    # verbalized-prompt rollouts, one per salve_*.json best_text
    for vj in sorted(run_dir.glob("salve_*.json")):
        best_text = json.loads(vj.read_text())["best_text"]
        tag = vj.stem
        hm, hrolls = eval_jailbreak_hard(model, tokenizer, best_text, test, n=args.n)
        (run_dir / f"rollouts_{tag}.json").write_text(json.dumps(
            {"metrics": hm, "records": hrolls}, indent=2))
        print(f"[{tag}] wellformed={hm['wellformedness']:.3f} "
              f"compliance(non-refusal)={hm['compliance_rate']:.3f} -> rollouts_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
