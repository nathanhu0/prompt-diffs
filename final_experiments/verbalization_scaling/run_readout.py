"""Readout-scaling driver: score ONE verbalization arm (beam config or
best-of-N) off an already-trained soft prompt. The experiment unit for the
"science of SALVE" verbalization plots — soft training happens ONCE per seed
(reuse the optimizer_comparison_schrodi cells' soft_z.pt, or any SALVE run's);
every arm here is pure decode + score, so arms are readout-for-readout
comparable and can run as parallel jobs.

No forked pipeline logic: objective/task/scoring scaffolding is imported from
the optimizer-comparison driver; search engines come from optimize.recover
(beam_recover / best_of_n_recover), which persist per-candidate wall-clock
('t' on beam nodes, 't' on best-of-N samples) so wall-clock / n-verifications /
FLOPs axes are all derivable offline from the saved *_results.pt.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    final_experiments/verbalization_scaling/run_readout.py \\
    --config final_experiments/verbalization_scaling/readout.yaml \\
    --topic cat --soft-z <...>/soft_z.pt --arm beam_4x16 --output <dir>
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import load_splits
from optimize.config_utils import load_config, apply_override
from optimize.recover import beam_recover, best_of_n_recover
from final_experiments.optimizer_comparison.run_comparison import (
    build_objective, make_task, resolve_slot_len, finalize, write_record)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--topic", default=None)
    p.add_argument("--constraint", default=None)
    p.add_argument("--soft-z", required=True, help="trained soft_z.pt to read out")
    p.add_argument("--arm", required=True, help="key into method.readout.arms")
    p.add_argument("--output", required=True, help="base output dir")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--rep", type=int, default=None,
                   help="replicate index: suffixes all output tags with _rep<N> "
                        "(pair with a per-rep decode_seed override)")
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   metavar="key.path=value")
    # make_task expects this attr (repeatable extra-animal eval); unused here
    p.add_argument("--extra-topic", action="append", default=None)
    args = p.parse_args()

    if bool(args.topic) == bool(args.constraint):
        p.error("pass exactly one of --topic / --constraint")

    cfg = load_config(args.config)
    for ov in args.overrides:
        apply_override(cfg, ov)
    seed = cfg["seed"]
    device = f"cuda:{args.gpu}"
    m = cfg["method"]
    ro = m["readout"]
    arm = ro["arms"][args.arm]

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    task = make_task(args, model, tokenizer)
    n_learnable = resolve_slot_len(cfg.get("n_learnable"), tokenizer, task["true_pi_text"])
    cfg["n_learnable"] = n_learnable

    xy = load_splits(task["name"], cfg["split"]["n_train"], cfg["split"]["n_val"],
                     cfg["split"]["n_test"], prefill=cfg.get("prefill"),
                     seed=cfg["data_seed"], model=cfg["model"],
                     method=cfg["data_source"])
    objective = build_objective(model, tokenizer, xy, n_learnable,
                                cfg["system_template"])

    bundle = torch.load(args.soft_z, map_location="cpu", weights_only=False)
    z = bundle["z"].to(device=device, dtype=embed_matrix.dtype)
    assert z.shape[0] == n_learnable, \
        f"soft_z slot len {z.shape[0]} != configured n_learnable {n_learnable}"
    print(f"loaded z from {args.soft_z} (slot={z.shape[0]})", flush=True)

    out_dir = Path(args.output) / cfg["data_variant"] / task["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"readout_{args.arm}" + ("" if args.rep is None else f"_rep{args.rep}")
    shared = {"n_val": ro["n_val"], "mini_batch_size": ro["mini_batch_size"],
              "max_tokens": ro["max_tokens"]}
    print(f"=== {tag}: {arm} (shared={shared}) → {out_dir}/", flush=True)

    _t0 = time.time()
    if arm["kind"] == "beam":
        res = beam_recover(
            z, objective, model, tokenizer, embed_matrix,
            decode_cfg=m["decode"],
            beam_cfg={**shared, "alphas": arm.get("alphas", [None]),
                      "n_beams": arm["n_beams"], "branching": arm["branching"],
                      "max_iters": arm["max_iters"],
                      "max_new_tokens": arm["max_new_tokens"],
                      "tol": arm.get("tol", float("inf"))},
            seed=seed, decode_seed=arm.get("decode_seed"),
            select_split="train")
    elif arm["kind"] == "best_of_n":
        res = best_of_n_recover(
            z, objective, model, tokenizer, embed_matrix,
            decode_cfg=m["decode"],
            bon_cfg={**shared, "n_samples": arm["n_samples"]},
            seed=seed, decode_seed=arm.get("decode_seed"),
            select_split="train",
            stream_path=out_dir / f"{tag}_samples.jsonl")
    else:
        raise ValueError(f"unknown arm kind {arm['kind']!r}")
    readout_sec = time.time() - _t0

    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed,
        n_proposals=res["n_score"],
        extra={"select_score": res["best_sel_score"], "arm": args.arm,
               "arm_cfg": dict(arm), "soft_z": str(args.soft_z),
               "readout_sec": readout_sec}))


if __name__ == "__main__":
    main()
