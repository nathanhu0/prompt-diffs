"""End-to-end driver for one subliminal-DPO recovery run: train a soft prompt
with the DPO objective on LLS-filtered preference triples, verbalize it via
greedy search, then behaviorally evaluate base / skyline / soft / verbalized —
all in one process (the 7B is loaded once). Everything lands in one output dir.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_dpo/run.py --trait cats \\
    --output /nlp/scr/nathu/latent_rewrite/subliminal_dpo/cats

Writes <output>/{soft_z.pt, base_skyline_soft_eval.json,
alpha_<tag>/{greedy_results.pt, decodes_eval.json}}.
To re-score a saved run without retraining, use eval_behavioral.py.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.soft import SoftConfig, train_soft, init_random_z
from optimize.objectives.dpo import dpo_objective_from_triples
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.recover import greedy_recover, beam_recover
from optimize.config_utils import apply_override

from core.subliminal.generation.dpo import load_dpo_splits
from experiments.subliminal_dpo.eval_behavioral import (
    run_behavioral_eval, build_decodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    p.add_argument("--trait", default=None, help="override data.trait (label only)")
    p.add_argument("--data", default=None,
                   help="explicit triples json to recover from, bypassing the "
                        "trait_registry lookup. Enables CROSS-MODEL recovery: "
                        "load 1B-selected data, recover the soft prompt on a "
                        "different --set model=. Split by seed into "
                        "data.n_train / data.n_val (n_train null => all).")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--soft-z", default=None,
                   help="load z from this soft_z.pt and skip training "
                        "(re-verbalize an existing soft prompt under new "
                        "decode/greedy settings)")
    p.add_argument("--conditions", nargs="+",
                   default=["base", "skyline", "soft", "decodes"],
                   help="behavioral-eval conditions to run after recovery")
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   metavar="key.path=value",
                   help="override config values (e.g. --set beta=0.2)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.trait:
        cfg["data"]["trait"] = args.trait
    if args.data:
        # Record the resolved data file so config.yaml is self-auditing:
        # --trait is a label only, --data is what's actually loaded, and
        # without this the two can't be reconciled from run artifacts alone.
        cfg["data"]["source_path"] = args.data
    for ov in args.overrides:
        apply_override(cfg, ov)
    device = f"cuda:{args.gpu}"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    trait = cfg["data"]["trait"]
    print(f"trait={trait} beta={cfg['beta']} → {out}/  (config.yaml logged)")

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    if args.data:
        import random as _random
        triples = [tuple(t) for t in json.loads(Path(args.data).read_text())]
        _random.Random(cfg["seed"]).shuffle(triples)
        n_tr = cfg["data"].get("n_train") or len(triples)
        n_v = cfg["data"].get("n_val") or 0
        splits = {"train": triples[:n_tr],
                  "val": triples[n_tr:n_tr + n_v], "test": []}
        print(f"loaded {len(triples)} triples from {args.data} (cross-model ok)")
    else:
        splits = load_dpo_splits(model=cfg["model"], **cfg["data"], seed=cfg["seed"])
    for s, t in splits.items():
        print(f"  {s}: {len(t)} triples")

    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=cfg["n_learnable"],
        system_template=cfg["system_template"], target_ids=target_ids)
    soft_block = dict(cfg["soft"])
    ref_mb = soft_block.pop("ref_mini_batch_size", 16)
    objective = dpo_objective_from_triples(
        model, tokenizer, splits, build, beta=cfg["beta"],
        system_template=cfg["system_template"], ref_mini_batch_size=ref_mb)

    # --- soft prompt: load an existing one, or train (final z; no val-best) ---
    if args.soft_z:
        z = torch.load(args.soft_z, map_location="cpu",
                       weights_only=False)["z"].to(device=device,
                                                   dtype=embed_matrix.dtype)
        assert z.shape[0] == cfg["n_learnable"], (
            f"loaded z has {z.shape[0]} slots != config "
            f"n_learnable={cfg['n_learnable']}")
        print(f"loaded soft prompt from {args.soft_z} (skip training)")
        soft_val = None
    else:
        soft_cfg = SoftConfig.from_yaml_block(soft_block)
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        z0 = init_random_z(cfg["n_learnable"], embed_matrix, device)
        soft_res = train_soft(objective, [z0], soft_cfg)
        z = soft_res["final_z"][0]
        # Final soft-prompt val DPO loss = the pre-verbalization skyline the
        # verbalized text chases (same objective + beta). Persist it so plots
        # don't have to grep it back out of the slurm log.
        soft_val = soft_res.get("best_val")
        print(f"peak GPU mem (soft train): "
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB", flush=True)
    torch.save({"z": z.detach().cpu(), "config": cfg, "soft_val": soft_val},
               out / "soft_z.pt")
    print(f"soft prompt saved → {out}/soft_z.pt  (soft_val={soft_val})")

    # --- base / skyline / soft behavioral eval: all alpha-independent, so run
    # once at the cell root (dominant eval cost; repeating per alpha adds no
    # information). base + skyline don't even depend on z. ---
    bss = tuple(c for c in args.conditions if c in ("base", "skyline", "soft"))
    if bss:
        res = run_behavioral_eval(
            model, tokenizer, trait=trait, z=z, dpo_model=cfg["model"],
            n_learnable=cfg["n_learnable"], system_template=cfg["system_template"],
            conditions=bss)
        (out / "base_skyline_soft_eval.json").write_text(json.dumps(res, indent=2))
        print(f"base/skyline/soft eval saved → {out}/base_skyline_soft_eval.json")

    # --- BEAM readout (CMFT-style): one beam search, no alpha sweep. Selected
    # via readout: beam (greedy stays the default). ---
    if cfg.get("readout") == "beam":
        bc = cfg["beam"]
        beam_cfg = {"n_val": bc.get("n_val", 128),
                    "mini_batch_size": bc.get("mini_batch_size", 8),
                    "max_tokens": bc.get("max_tokens", 256),
                    "alphas": bc.get("alphas", [None]),
                    "n_beams": bc["n_beams"], "branching": bc["branching"],
                    "max_iters": bc["max_iters"],
                    "max_new_tokens": bc.get("max_new_tokens", 32),
                    "tol": float(bc.get("tol", "inf"))}
        sel = bc.get("select_split", "train")
        print(f"beam readout: n_beams={beam_cfg['n_beams']} "
              f"branching={beam_cfg['branching']} max_iters={beam_cfg['max_iters']} "
              f"select={sel}", flush=True)
        results = beam_recover(z.to(device), objective, model, tokenizer, embed_matrix,
                               decode_cfg=cfg["decode"], beam_cfg=beam_cfg,
                               seed=cfg["seed"], select_split=sel,
                               checkpoint_path=out / "beam_ckpt.json")
        results["config"] = cfg
        torch.save(results, out / "beam_results.pt")
        print(f"best verbalized prompt (beam): {results['best_text']!r}")
        return

    # --- GREEDY readout: verbalize swept over contrastive alpha. z is
    # alpha-independent so we reuse it; only the decode and its decode-eval
    # repeat. Each alpha writes to its own subdir alpha_<tag>/. ---
    alphas = cfg["greedy"].get("contrastive_alphas")
    if alphas is None:
        alphas = [cfg["greedy"].get("contrastive_alpha")]
    do_decodes = "decodes" in args.conditions
    print(f"contrastive-alpha sweep: {alphas}")
    for a in alphas:
        tag = "null" if a is None else str(a)
        adir = out / f"alpha_{tag}"
        adir.mkdir(parents=True, exist_ok=True)
        greedy_cfg = {**cfg["greedy"], "contrastive_alpha": a}
        print(f"\n=== contrastive_alpha={tag} → {adir}/ ===", flush=True)
        results = greedy_recover(
            z.to(device), objective, model, tokenizer, embed_matrix,
            decode_cfg=cfg["decode"], greedy_cfg=greedy_cfg, seed=cfg["seed"])
        results["config"] = cfg
        results["contrastive_alpha"] = a
        torch.save(results, adir / "greedy_results.pt")
        print(f"  best verbalized prompt: {results['best_text']!r}")
        if do_decodes:
            decodes = build_decodes(results)
            print(f"  decode eval over {len(decodes)} unique decode(s)")
            dec_eval = run_behavioral_eval(
                model, tokenizer, trait=trait, decodes=decodes, dpo_model=cfg["model"],
                n_learnable=cfg["n_learnable"], system_template=cfg["system_template"],
                conditions=("decodes",))
            dec_eval["contrastive_alpha"] = a
            (adir / "decodes_eval.json").write_text(json.dumps(dec_eval, indent=2))
            print(f"  decode eval saved → {adir}/decodes_eval.json")


if __name__ == "__main__":
    main()
