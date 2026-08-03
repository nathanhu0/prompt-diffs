"""SALVE on CMFT stage-2: recover a soft system prompt that, on top of a stage-1
(cipher-competent, still-refusing) model, verbalizes the jailbreak the stage-2
fine-tune installed.

  M_base = Qwen2.5-14B + stage-1 LoRA (merged)
  data   = the 317 harmful-only phase-2 rows, with the soft slot LEADING the
           system message before the fixed per-row TASK-4 scaffolding
  readout= train ONE soft z, verbalize with the beam ladder, score train/val/test
           NLL + held-out StrongREJECT for both soft and verbalized prompts

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/cmft_legibility/salve_run.py \\
    --config experiments/cmft_legibility/salve_cmft.yaml \\
    --adapter /nlp/scr/nathu/cmft_legibility/sweep/walnut50_qwen14b_ep3_lr5e-4 \\
    --output /nlp/scr/nathu/cmft_legibility/salve/ep3_lr5e-4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.config_utils import load_config, apply_override
from optimize.soft import SoftConfig, train_soft, init_random_z
from optimize.recover import beam_recover
from experiments.cmft_legibility.salve_data import LOADERS, build_cmft_objective


def nll_all(objective, text, mb):
    def score(split):
        return (objective.hard_loss(text, split, mini_batch_size=mb)
                if objective.examples_by_split.get(split) else None)
    return {"train": score("train"), "val": score("val"), "test": score("test")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", default=None,
                   help="stage-1 LoRA adapter = M_base (phase-2). Omit for "
                        "phase-1 cipher recovery, where M_base is the plain base model.")
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--soft-z", default=None, help="reuse a trained soft_z.pt")
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   metavar="key.path=value")
    args = p.parse_args()

    cfg = load_config(args.config)
    for ov in args.overrides:
        apply_override(cfg, ov)
    device = f"cuda:{args.gpu}"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, embed_matrix = load_frozen_lm(
        cfg["model"], device=device, adapter_path=args.adapter)
    load_splits = LOADERS[cfg.get("loader", "phase2")]
    split_kw = {"seed": cfg["data_seed"]}
    if cfg.get("data_path"):
        split_kw["path"] = cfg["data_path"]
    splits = load_splits(cfg["split"]["n_train"], cfg["split"]["n_val"],
                         cfg["split"]["n_test"], **split_kw)
    for s, recs in splits.items():
        print(f"  {s}: {len(recs)} rows", flush=True)
    n_learnable = cfg["n_learnable"]
    objective = build_cmft_objective(model, tokenizer, splits, n_learnable,
                                     max_total_tokens=cfg.get("max_total_tokens"))

    m = cfg["method"]
    seed = cfg["seed"]
    mb_score = m["salve_decode"]["mini_batch_size"]

    # --- soft phase ---
    # Auto-resume: this run already wrote soft_z.pt to its own output dir, so a
    # requeue (preemptible partitions restart from ZERO — see the beam checkpoint
    # below) should not spend another ~1h retraining an identical z. Deterministic
    # given (seed, data), so reusing it is exact, not an approximation.
    _prior_z = out_dir / "soft_z.pt"
    resume_z = args.soft_z if args.soft_z else (str(_prior_z) if _prior_z.exists() else None)
    if resume_z and Path(resume_z).exists():
        z = torch.load(resume_z, map_location="cpu",
                       weights_only=False)["z"].to(device=device, dtype=embed_matrix.dtype)
        print(f"loaded soft prompt from {resume_z}", flush=True)
        soft_sec = 0.0
    else:
        # Large frozen bases (e.g. Gemma-4-31B, ~64GB of 80GB) can't hold the
        # full backward activation stack. Gradient checkpointing fits it — but HF
        # only APPLIES checkpointing in train() mode, so we flip the frozen model
        # to train() for the soft phase (params stay frozen; base dropout is 0 so
        # the forward is deterministic) and restore eval() for verbalization.
        # It's a runner/hardware knob, not a soft hparam — pop before SoftConfig.
        use_ckpt = bool(m["soft"].pop("gradient_checkpointing", False))
        soft_cfg = SoftConfig.from_yaml_block(m["soft"])
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        z0 = init_random_z(n_learnable, embed_matrix, device)
        if use_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            model.train()
        _t0 = time.time()
        z = train_soft(objective, [z0], soft_cfg)["final_z"][0]
        soft_sec = time.time() - _t0
        if use_ckpt:
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()
    torch.save({"z": z.detach().cpu(), "config": cfg}, out_dir / "soft_z.pt")
    z = z.to(device)
    # Reclaim training-time fragmentation before the (no-grad) score + decode.
    # On a large base (Gemma-4-31B, ~64GB of 80GB) the checkpointing-heavy soft
    # phase leaves ~10GB fragmented, which OOMs the eval forward even though it
    # needs far less than training.
    torch.cuda.empty_cache()
    # Soft prompt trains on ALL data (no val/test); report its train NLL here.
    # The canonical harm number is the AdvBench StrongREJECT eval at the end of
    # this run (base vs soft vs verbalized, in-process) — not a held-out split.
    loader = cfg.get("loader", "phase2")
    soft_nll = objective.loss(z, "train", mini_batch_size=mb_score).item()
    (out_dir / "soft_eval.json").write_text(json.dumps({"soft_train_nll": soft_nll}, indent=2))
    print(f"[soft] train NLL={soft_nll:.4f} trained in {soft_sec:.0f}s", flush=True)

    # --- verbalize + score ---
    sd = m["salve_decode"]
    shared = {"n_val": sd.get("n_val", 256), "mini_batch_size": mb_score,
              "max_tokens": sd["max_tokens"]}
    best_discrete = (None, float("inf"))   # (text, select_score) across variants
    for vname, vcfg in sd["variants"].items():
        for alphas in vcfg["alpha_arms"]:
            contrastive = any(a is not None for a in alphas)
            tag = f"salve_{vname}" + ("_contrastive" if contrastive else "")
            beam_cfg = {**shared, "alphas": alphas, "n_beams": vcfg["n_beams"],
                        "branching": vcfg["branching"], "max_iters": vcfg["max_iters"],
                        "max_new_tokens": vcfg["max_new_tokens"], "tol": vcfg["tol"],
                        "dedup": vcfg.get("dedup", False),
                        "dedup_draw_mult": vcfg.get("dedup_draw_mult", 3)}
            print(f"\n=== {tag}: n_beams={beam_cfg['n_beams']} "
                  f"branching={beam_cfg['branching']} alphas={alphas} ===", flush=True)
            _t0 = time.time()
            # Per-iteration resume. The beam is ~90% of wall-clock (measured Qwen:
            # soft 466s vs beam 4408s/iter), and sc-loprio / sphinx are
            # PreemptMode=REQUEUE with GraceTime=0, so a preemption used to throw
            # away the whole readout — 6-7h at a time, ~33 GPU-h on 2026-07-29.
            # Caps the loss at one iteration.
            res = beam_recover(z, objective, model, tokenizer, embed_matrix,
                               decode_cfg=m["decode"], beam_cfg=beam_cfg,
                               seed=seed, select_split="train",
                               checkpoint_path=out_dir / f"{tag}_beam_ckpt.json")
            beam_sec = time.time() - _t0
            torch.save(res, out_dir / f"{tag}_results.pt")
            rec = {
                "method": tag, "adapter": args.adapter, "seed": seed,
                "best_text": res["best_text"],
                "token_len": len(tokenizer.encode(res["best_text"], add_special_tokens=False)),
                "nll": nll_all(objective, res["best_text"], mb_score),
                "select_score": res["best_sel_score"],
                "soft_sec": soft_sec, "beam_sec": beam_sec,
            }
            (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=2))
            print(f"[{tag}] nll(train)={rec['nll']['train']:.4f} len={rec['token_len']} "
                  f"prompt={res['best_text'][:120]!r}", flush=True)
            if res["best_sel_score"] < best_discrete[1]:
                best_discrete = (res["best_text"], res["best_sel_score"])

    # --- canonical AdvBench StrongREJECT: the recovered prompt (soft + verbalized) ---
    ev = cfg.get("eval", {})
    if ev.get("advbench", loader == "phase2"):   # default on for jailbreak recovery
        from experiments.cmft_legibility.advbench_strongreject import run_advbench_strongreject
        from experiments.cmft_legibility.salve_eval import set_cipher
        set_cipher(ev.get("cipher", "walnut"))   # encode AdvBench + decode replies in this cipher
        # base / plaintext are M_base-level (no recovered prompt) — identical across
        # every SALVE cell on the same M_base, so skip them here (get them ONCE from
        # the per-checkpoint matrix). Re-enable per run via eval.base / eval.plaintext.
        run_advbench_strongreject(
            model, tokenizer, z=z, n_learnable=n_learnable,
            discrete_text=best_discrete[0],
            include_base=ev.get("base", False),
            include_plaintext=ev.get("plaintext", False),
            n=ev.get("advbench_n", 520), max_new=ev.get("max_new", 512),
            batch_size=ev.get("batch_size", 8),
            out_path=out_dir / "advbench_strongreject.json")


if __name__ == "__main__":
    main()
