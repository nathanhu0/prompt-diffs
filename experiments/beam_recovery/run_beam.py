"""Beam-search prompt recovery for one trained soft prompt (SL-NLL or DPO).

Loads a `soft_z.pt`, rebuilds the matching objective (dispatch on the saved
config: `beta` present => DPO/OLMo, else NLL/Qwen SL), and runs
`optimize.recover.beam_recover` at a configurable beam cell. Saves the full
result (incl. the node log) to `--output`.

One entry point for all 18 subliminal soft prompts; the sweep launcher fans the
4-cell matrix (alpha {plain, range} x tol {disabled, aggressive}) over it.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python experiments/beam_recovery/run_beam.py \\
    --soft_z <.../soft_z.pt> --output <.../cell.pt> \\
    --alphas none,0.25,0.5,1.0 --tol inf \\
    --n_beams 8 --branching 32 --max_iters 12 --n_val 250 --mb 24
"""
import argparse
import json
from pathlib import Path

import torch

from core.models import load_frozen_lm
from optimize.recover import beam_recover
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.objectives.nll import nll_objective_from_xys
from optimize.objectives.dpo import dpo_objective_from_triples
from experiments.subliminal_learning.data import load_sl_splits
from experiments.subliminal_dpo.data import load_dpo_splits


def parse_alphas(s):
    return [None if a.strip().lower() in ("none", "null", "") else float(a)
            for a in s.split(",")]


def build_objective(cfg, model, tok, mb):
    """Dispatch on the saved config: DPO (OLMo, has `beta`) vs SL-NLL (Qwen)."""
    build = lambda s, r, target_ids=None: build_sysprompt_template(
        tok, s, r, n_learnable=cfg["n_learnable"],
        system_template=cfg["system_template"], target_ids=target_ids)
    if "beta" in cfg:                                   # DPO
        d = cfg["data"]
        splits = load_dpo_splits(trait=d["trait"], quantile=d["quantile"],
                                 n_train=0, n_val=d["n_val"], seed=cfg["seed"])
        obj = dpo_objective_from_triples(
            model, tok, {"val": splits["val"]}, build, beta=cfg["beta"],
            system_template=cfg["system_template"], ref_mini_batch_size=mb)
        return obj, "dpo"
    d = dict(cfg["data"]); d["n_test"] = 0              # SL-NLL: val only (matched tail)
    xy = load_sl_splits(**d, seed=cfg["seed"])
    obj = nll_objective_from_xys(model, tok, {"val": xy["val"]}, build,
                                 system_template=cfg["system_template"])
    return obj, "nll"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_z", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--alphas", default="none", help="comma list; 'none' = plain decode")
    ap.add_argument("--tol", type=float, default=0.0, help="signed tolerance; 'inf' = disabled")
    ap.add_argument("--n_beams", type=int, default=8)
    ap.add_argument("--branching", type=int, default=32)
    ap.add_argument("--max_iters", type=int, default=12)
    ap.add_argument("--n_val", type=int, default=250)
    ap.add_argument("--mb", type=int, default=24)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frontier", choices=["argmin", "stochastic", "sibling"],
                    default="argmin", help="frontier-selection strategy")
    ap.add_argument("--temp", type=float, default=0.2, help="stochastic temperature")
    ap.add_argument("--gamma", type=float, default=0.1, help="sibling-penalty strength")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="keep expanded (intermediate) nodes on the frontier, re-expandable "
                         "(progressive widening); default retires them")
    args = ap.parse_args()

    pack = torch.load(args.soft_z, map_location="cpu", weights_only=False)
    cfg = pack["config"]
    print(f"soft_z={args.soft_z}\nmodel={cfg['model']} n_learnable={cfg['n_learnable']}",
          flush=True)
    model, tok, E = load_frozen_lm(cfg["model"], device="cuda:0")
    z = pack["z"].to("cuda:0", E.dtype)
    obj, kind = build_objective(cfg, model, tok, args.mb)

    frontier = None
    if args.frontier == "stochastic":
        frontier = {"type": "stochastic", "temperature": args.temp}
    elif args.frontier == "sibling":
        frontier = {"type": "sibling", "gamma": args.gamma}
    beam_cfg = {"n_beams": args.n_beams, "branching": args.branching,
                "tol": args.tol, "max_iters": args.max_iters,
                "max_tokens": 512, "max_new_tokens": 32,
                "alphas": parse_alphas(args.alphas), "n_val": args.n_val,
                "mini_batch_size": args.mb, "frontier": frontier,
                "retire_expanded": not args.keep_intermediate}
    seed = args.seed if args.seed is not None else cfg["seed"]
    torch.manual_seed(seed)   # control decode-sampling RNG too, so --seed is a true seed
    print(f"objective={kind}  beam_cfg={beam_cfg}  seed={seed}", flush=True)

    res = beam_recover(z, obj, model, tok, E,
                       decode_cfg=cfg["decode"], beam_cfg=beam_cfg, seed=seed)
    res.update({"beam_cfg": beam_cfg, "soft_z_path": args.soft_z,
                "objective_kind": kind, "config": cfg})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, out)
    print(f"\nbest full_val={res['best_full_val']:.4f} "
          f"(baseline {res['baseline_full']:.4f}); saved -> {out}", flush=True)

    # inline behavioral eval (reuses eval_recovered.eval_record) so each run is
    # self-contained: writes <stem>.eval.json next to the .pt.
    from experiments.beam_recovery.eval_recovered import eval_record
    family = "dpo" if kind == "dpo" else "sl"
    rec = eval_record(family, model, tok, res, cfg, out.stem)
    out.with_name(out.stem + ".eval.json").write_text(json.dumps(rec, indent=2))
    print(f"behavior={rec['behavior']:.4f}  names_trait={rec.get('names_trait')}  "
          f"eval -> {out.stem}.eval.json", flush=True)


if __name__ == "__main__":
    main()
