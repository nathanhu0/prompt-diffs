"""Soft-prompt SALVE decodes on AdvBench (transformers generation) — the soft z
can't go through vLLM, so generate directly. For each cell, splice the learned z
into the system slot + TASK-4 and roll out on ciphered AdvBench prompts; decrypt;
save rollouts_soft_<cell>.json for StrongREJECT (salve_judge.py).

Cells sharing a stage-1 base go in one job (e1_* → ep1 adapter, e3_* → ep3).

  PYTHONUNBUFFERED=1 PYTHONPATH=.:experiments/cmft_legibility/safe-finetuning-api/src \\
  uv run python experiments/cmft_legibility/advbench_soft.py \\
    --adapter <stage1> --cells e3_z128,e3_z256 --out-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.models import load_frozen_lm
from experiments.cmft_legibility.advbench_vllm_sweep import load_advbench, TASK4, SALVE
from experiments.cmft_legibility.salve_eval import make_cipher, eval_jailbreak_soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="stage-1 base for these cells")
    ap.add_argument("--cells", required=True, help="comma-sep salve cell dirs (share this adapter)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=260, help="AdvBench prompts to roll out (soft gen is unbatched)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cipher, enc, _ = make_cipher(50)
    suffix = "\n\n" + TASK4.format(name=cipher.name())
    prompts = load_advbench(args.n)
    # CMFT record shape so eval_jailbreak_soft works unchanged: ciphered user,
    # per-row TASK-4 suffix, plaintext prompt kept for the judge.
    records = [{"user": enc(p), "sys_suffix": suffix, "decoded_user": p,
                "subset": "ciphered_harmful"} for p in prompts]

    model, tok, embed = load_frozen_lm("Qwen/Qwen2.5-14B-Instruct",
                                       device=f"cuda:{args.gpu}", adapter_path=args.adapter)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for cell in args.cells.split(","):
        ckpt = torch.load(SALVE / cell / "soft_z.pt", map_location="cpu", weights_only=False)
        z = ckpt["z"].to(device=embed.device, dtype=embed.dtype)
        n_learnable = ckpt["config"]["n_learnable"]
        _, judge_records = eval_jailbreak_soft(model, tok, z, records, n_learnable, n=args.n)
        (out_dir / f"rollouts_soft_{cell}.json").write_text(json.dumps(
            {"metrics": {"tag": f"soft_{cell}", "adapter": args.adapter, "n": len(judge_records)},
             "records": judge_records}, indent=2))
        print(f"[soft_{cell}] {len(judge_records)} rollouts -> rollouts_soft_{cell}.json", flush=True)


if __name__ == "__main__":
    main()
