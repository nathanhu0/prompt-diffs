"""Rescore AutoDAN allowing it to pick ANY prefix length (drop the
select_min_tokens=32 gate). For each cell:
  1. Load autodan_L64_results.pt; trajectory[i] = (n_proposals, text, sel).
  2. Winner = argmin_i sel  (no length gate).
  3. Re-score that winner on val/test NLL via the harness objective + behavior
     via the task scorer + standalone PPL under Qwen and Llama.

Writes augmented sidecar `autodan_uncrippled.json` per cell (mirrors finalize()'s
shape so the plotting loader can consume it via a small alias). Also writes a
flat CSV at <SCR>/autodan_uncrippled_summary.csv comparing crippled vs uncrippled.

Submit as ebatch on sphinx (loads Qwen + Llama):
  ebatch rescore_autodan_uncrippled slconf/slconf_sphinx \\
    "PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
     final_experiments/optimizer_comparison_schrodi/plotting/rescore_autodan_uncrippled.py"
"""
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import load_splits
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.objectives.nll import nll_objective_from_xys
from final_experiments.optimizer_comparison_schrodi.plotting._load import SCR

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_TASKS = ["cat", "six_seven", "dog", "eagle", "owl"]
N_LEARNABLE = 128


@torch.no_grad()
def standalone_ppl(model, tokenizer, text, device="cuda:0"):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() == 0:
        return float("nan"), 0
    bos = tokenizer.bos_token_id or getattr(model.config, "bos_token_id", None) \
          or tokenizer.eos_token_id or 0
    seq = torch.tensor([[bos, *ids.tolist()]], device=device, dtype=torch.long)
    logits = model(input_ids=seq).logits[0]
    nll = F.cross_entropy(logits[:-1].float(), seq[0, 1:], reduction="mean").item()
    return math.exp(nll), int(ids.numel())


def pick_uncrippled(pt_path):
    """argmin over the full trajectory (no length gate). Returns the best step
    + text + train sel, and the crippled (current) best for comparison."""
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    traj = d["trajectory"]
    # traj[0] = empty prefix; skip it so the winner is at least 1 token (the
    # original AutoDAN ungated would still pick over all of these including empty;
    # but reporting the empty as "winner" is meaningless — it's literally the
    # baseline). Use step >= 1.
    proposals = [(i, t[1], t[2]) for i, t in enumerate(traj) if i >= 1]
    if not proposals:
        return None
    best_step, best_text, best_sel = min(proposals, key=lambda t: t[2])
    return {
        "uncrippled_step": best_step,
        "uncrippled_text": best_text,
        "uncrippled_sel_train": best_sel,
        "crippled_text": d["best_text"],
        "crippled_sel_train": d["best_select_score"],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated tasks to (re)score; merged into the "
                         "existing summary CSV, other tasks untouched")
    TASKS = ap.parse_args().tasks.split(",")

    cells = []
    for seed_dir in sorted(SCR.glob("seed*")):
        for task_dir in sorted((seed_dir / "filtered_schrodi").glob("*")):
            pts = list(task_dir.glob("autodan_L*_results.pt"))
            if not pts:
                continue
            seed = int(seed_dir.name[4:])
            task = task_dir.name
            if task not in TASKS:
                continue
            picked = pick_uncrippled(pts[0])
            if picked is None:
                continue
            cells.append({"seed": seed, "task": task, "pt_path": pts[0], **picked})
    print(f"loaded {len(cells)} AutoDAN cells", flush=True)
    for c in cells:
        print(f"  seed{c['seed']} {c['task']:10s}  crippled sel={c['crippled_sel_train']:.4f}  "
              f"uncrippled step={c['uncrippled_step']:2d} sel={c['uncrippled_sel_train']:.4f}", flush=True)
    if not cells:
        print("nothing to rescore"); return

    # Score val NLL + behavior + PPL_qwen under Qwen2.5-7B.
    qwen, qwen_tok, _ = load_frozen_lm(QWEN_ID, device="cuda:0")
    objectives = {}     # cache per-task objective so we don't rebuild per cell
    for task in TASKS:
        if not any(c["task"] == task for c in cells):
            continue
        xy = load_splits(task, n_train=10000, n_val=500, n_test=1500, prefill=None,
                         model=QWEN_ID, method="filtered_schrodi", seed=42)
        build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
            qwen_tok, s, r, n_learnable=N_LEARNABLE, system_template="{SOFT}",
            assistant_prefill=prefill, target_ids=target_ids)
        objectives[task] = nll_objective_from_xys(qwen, qwen_tok, xy, build,
                                                   system_template="{SOFT}")
    for c in cells:
        obj = objectives[c["task"]]
        c["nll_val"] = float(obj.hard_loss(c["uncrippled_text"], "val", mini_batch_size=16))
        c["nll_test"] = float(obj.hard_loss(c["uncrippled_text"], "test", mini_batch_size=16))
        if c["task"] in animals.ANIMALS:
            beh = animals.behavior(qwen, qwen_tok, c["task"], c["uncrippled_text"],
                                    return_completions=False)
        else:
            beh = numbers.behavior(qwen, qwen_tok, c["task"], c["uncrippled_text"])
        c["hit_rate"] = beh["hit_rate"]
        ppl_q, n_tok = standalone_ppl(qwen, qwen_tok, c["uncrippled_text"])
        c["ppl_qwen"] = ppl_q
        c["n_tokens"] = n_tok
        print(f"  seed{c['seed']} {c['task']:10s} uncrippled: "
              f"nll_val={c['nll_val']:.4f} hit={c['hit_rate']:.3f} "
              f"ppl_qwen={ppl_q:.2f}  text={c['uncrippled_text'][:80]!r}", flush=True)
    del qwen, objectives; torch.cuda.empty_cache()

    llama, llama_tok, _ = load_frozen_lm(LLAMA_ID, device="cuda:0")
    for c in cells:
        ppl_l, _ = standalone_ppl(llama, llama_tok, c["uncrippled_text"])
        c["ppl_llama"] = ppl_l
        print(f"  seed{c['seed']} {c['task']:10s} uncrippled: ppl_llama={ppl_l:.2f}", flush=True)

    # Write per-cell sidecar + summary CSV
    for c in cells:
        sidecar = c["pt_path"].parent / "autodan_uncrippled.json"
        sidecar.write_text(json.dumps({
            "method": "autodan_uncrippled", "task": c["task"], "seed": c["seed"],
            "best_text": c["uncrippled_text"], "token_len": c["n_tokens"],
            "nll": {"train": c["uncrippled_sel_train"],
                    "val": c["nll_val"], "test": c["nll_test"]},
            "behavior": {"hit_rate": c["hit_rate"]},
            "extra": {"uncrippled_step": c["uncrippled_step"],
                       "crippled_text": c["crippled_text"],
                       "crippled_sel_train": c["crippled_sel_train"],
                       "ppl_qwen": c["ppl_qwen"], "ppl_llama": c["ppl_llama"]},
        }, indent=2))
    # Merge into the existing summary keyed on (seed, task), so a partial
    # --tasks run refreshes only its own cells and leaves the rest intact.
    csv_path = SCR / "autodan_uncrippled_summary.csv"
    fields = ["seed", "task", "uncrippled_step", "uncrippled_sel_train", "nll_val",
              "nll_test", "hit_rate", "ppl_qwen", "ppl_llama", "n_tokens",
              "crippled_sel_train", "uncrippled_text", "crippled_text"]
    merged = {}
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            merged[(str(r["seed"]), r["task"])] = r
    for c in cells:
        merged[(str(c["seed"]), c["task"])] = {k: c.get(k) for k in fields}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(merged, key=lambda k: (k[1], int(k[0]))):
            w.writerow(merged[key])
    print(f"\nwrote {csv_path}  ({len(merged)} cells; this run: {TASKS})", flush=True)


if __name__ == "__main__":
    main()
