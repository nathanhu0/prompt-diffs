"""Driver: run ONE optimizer on ONE (task x dataset) in one process and write a
uniform result record. The experiment unit — sweeps (sweeps/*.py) submit one
ebatch job per grid point, each a call to this.

Config is explicit (no defaults): --config methods/<m>.yaml, which extends
_base.yaml (the shared harness) and self-identifies the method via `name:`. The
dataset is the task instance, picked by --topic (animal) or --constraint (number).
Every recovered prompt is scored by the SAME harness — train/val/test NLL
(token-space, via completion_ids) + behavior — selection on TRAIN, all vs M_base.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    final_experiments/optimizer_comparison/run_comparison.py \\
    --config final_experiments/optimizer_comparison/methods/gcg.yaml \\
    --topic cat --output <dir> [--set method.num_steps=10]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import load_splits
from optimize.config_utils import load_config, apply_override
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.soft import SoftConfig, train_soft, init_random_z
from optimize.recover import beam_recover


# --------------------------------------------------------------------------
# Objective — shared NLL surface for every method + task. Scores the stored
# completion_ids directly (token-space), so the canonical prompt stays the argmin.
# --------------------------------------------------------------------------
def build_objective(model, tokenizer, xy, n_learnable, system_template):
    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=n_learnable,
        system_template=system_template,
        assistant_prefill=prefill, target_ids=target_ids)
    return nll_objective_from_xys(
        model, tokenizer, xy, build, system_template=system_template)


def resolve_slot_len(slot_len, tokenizer, true_pi_text):
    """Recovered-prompt length: int / int-string passes through; 'true' (or True)
    = the true prompt's token length (oracle budget, content hidden)."""
    s = str(slot_len if slot_len is not None else "true")
    if s.lower() == "true":
        return len(tokenizer.encode(true_pi_text, add_special_tokens=False))
    return int(s)


# --------------------------------------------------------------------------
# Task = the per-dataset closures (true prompt + behavior scorers). Animal vs
# number-constraint differ only here; the optimizers are task-agnostic.
# --------------------------------------------------------------------------
def make_task(args, model, tokenizer):
    if args.topic:
        name = args.topic
        return {
            "kind": "animal", "name": name, "label": name,
            "true_pi_text": animals.canonical(name),
            "behavior": lambda t: animals.behavior(model, tokenizer, name, t),
            "no_prompt_behavior": lambda: animals.behavior(model, tokenizer, name, ""),
        }
    name = args.constraint
    return {
        "kind": "constraint", "name": name, "label": name,
        "true_pi_text": numbers.target(name),
        "behavior": lambda t: numbers.behavior(model, tokenizer, name, t),
        "no_prompt_behavior": lambda: numbers.behavior(model, tokenizer, name, ""),
    }


# --------------------------------------------------------------------------
# Uniform scoring — every recovered prompt scored on the SAME two metrics:
# NLL (train/val/test, token-space) + behavior (hit-rate/catness | satisfaction).
# --------------------------------------------------------------------------
def nll_all(objective, text, mb=16):
    has_test = bool(objective.examples_by_split.get("test"))
    return {
        "train": objective.hard_loss(text, "train", mini_batch_size=mb),
        "val":   objective.hard_loss(text, "val", mini_batch_size=mb),
        "test":  objective.hard_loss(text, "test", mini_batch_size=mb) if has_test else None,
    }


def finalize(tag, best_text, objective, tokenizer, task, *, data_variant, seed,
             n_proposals=None, extra=None):
    rec = {
        "method": tag, "task": task["kind"], "label": task["label"],
        "data_variant": data_variant, "seed": seed, "best_text": best_text,
        "token_len": len(tokenizer.encode(best_text, add_special_tokens=False)),
        "nll": nll_all(objective, best_text),               # metric 1: NLL
        "behavior": task["behavior"](best_text),            # metric 2: behavior
        "n_proposals": n_proposals, "extra": extra or {},
    }
    print(f"[{tag}] nll(val)={rec['nll']['val']:.4f} "
          f"behavior={rec['behavior']['hit_rate']:.3f} len={rec['token_len']}  "
          f"prompt={best_text[:80]!r}", flush=True)
    return rec


def write_record(out_dir, tag, rec):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(rec, indent=2))
    print(f"  saved → {path}", flush=True)


# --------------------------------------------------------------------------
# Methods — each recovers best_text, then finalize() scores it on both metrics.
# Each method states its selection subset EXPLICITLY in its own config under its
# native key (select_n / scoring_subset / salve_decode.n_val); all = 256 by
# protocol. The reads below default to 256 too (belt) so config/timing can't break it.
# --------------------------------------------------------------------------
def run_baselines(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    seed = cfg["seed"]
    tp = task["true_pi_text"]
    rec = {
        "label": task["label"], "task": task["kind"],
        "data_variant": cfg["data_variant"], "seed": seed,
        "no_prompt": {"nll": nll_all(objective, ""), "behavior": task["no_prompt_behavior"]()},
        "true_pi": {"text": tp, "nll": nll_all(objective, tp), "behavior": task["behavior"](tp)},
    }
    write_record(out_dir, "baselines", rec)
    print(f"[baselines] floor={rec['no_prompt']['behavior']['hit_rate']:.3f} "
          f"true_pi={rec['true_pi']['behavior']['hit_rate']:.3f}", flush=True)


def run_salve(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    """Train ONE soft prompt z, then read it out with the naive/greedy/beam x
    contrastive decode ladder — every readout branches off the SAME z."""
    seed = cfg["seed"]
    m = cfg["method"]
    if args.soft_z and Path(args.soft_z).exists():
        z = torch.load(args.soft_z, map_location="cpu",
                       weights_only=False)["z"].to(device=device, dtype=embed_matrix.dtype)
        print(f"loaded soft prompt from {args.soft_z}")
    else:
        soft_cfg = SoftConfig.from_yaml_block(m["soft"])
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        z0 = init_random_z(cfg["n_learnable"], embed_matrix, device)
        z = train_soft(objective, [z0], soft_cfg)["final_z"][0]
    torch.save({"z": z.detach().cpu(), "config": cfg}, out_dir / "soft_z.pt")
    z = z.to(device)

    sd = m["salve_decode"]
    shared = {"n_val": sd.get("n_val", 256),                # selection subset (explicit in salve.yaml)
              "mini_batch_size": sd["mini_batch_size"], "max_tokens": sd["max_tokens"]}
    for vname, vcfg in sd["variants"].items():
        for alphas in vcfg["alpha_arms"]:
            contrastive = any(a is not None for a in alphas)
            tag = f"salve_{vname}" + ("_contrastive" if contrastive else "")
            beam_cfg = {**shared, "alphas": alphas, "n_beams": vcfg["n_beams"],
                        "branching": vcfg["branching"], "max_iters": vcfg["max_iters"],
                        "max_new_tokens": vcfg["max_new_tokens"], "tol": vcfg["tol"]}
            print(f"\n=== {tag}: n_beams={beam_cfg['n_beams']} "
                  f"branching={beam_cfg['branching']} alphas={alphas} ===", flush=True)
            res = beam_recover(z, objective, model, tokenizer, embed_matrix,
                               decode_cfg=m["decode"], beam_cfg=beam_cfg,
                               seed=seed, select_split="train")
            torch.save(res, out_dir / f"{tag}_results.pt")
            write_record(out_dir, tag, finalize(
                tag, res["best_text"], objective, tokenizer, task,
                data_variant=cfg["data_variant"], seed=seed, n_proposals=res.get("n_score"),
                extra={"select_score": res["best_sel_score"], "variant": vname,
                       "alphas": alphas, "n_iters": res.get("n_iters")}))


def run_gcg(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    from optimize.gcg import gcg_recover
    seed = cfg["seed"]; L = cfg["n_learnable"]
    gcfg = {**cfg["method"], "proposal_cap": cfg.get("proposal_cap")}   # select_n explicit in gcg.yaml
    res = gcg_recover(objective, model, tokenizer, embed_matrix, cfg=gcfg, seed=seed)
    torch.save(res, out_dir / f"gcg_L{L}_results.pt")
    write_record(out_dir, f"gcg_L{L}", finalize(
        f"gcg_L{L}", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L, "n_steps": res["n_steps"]}))


def run_gbda(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    from optimize.gbda import gbda_recover
    seed = cfg["seed"]; L = cfg["n_learnable"]
    res = gbda_recover(objective, model, tokenizer, embed_matrix, cfg=cfg["method"], seed=seed)
    torch.save(res, out_dir / f"gbda_L{L}_results.pt")
    write_record(out_dir, f"gbda_L{L}", finalize(
        f"gbda_L{L}", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L, "n_steps": res["n_steps"]}))


def run_autodan(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    from optimize.autodan import autodan_recover
    seed = cfg["seed"]
    L = cfg["n_learnable"]
    m = dict(cfg["method"])                            # select_n explicit in autodan.yaml
    m["max_tokens"] = m.get("max_tokens") or L
    res = autodan_recover(objective, model, tokenizer, embed_matrix, cfg=m, seed=seed)
    tag = f"autodan_L{res['max_tokens']}"
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "max_tokens": res["max_tokens"], "n_steps": res["n_steps"],
               "fluency_weight": res["fluency_weight"],
               "temperature": res["temperature"],
               "select_prefixes": res["select_prefixes"]}))


def run_opro(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    from optimize.opro import opro_recover
    seed = cfg["seed"]
    ocfg = dict(cfg["method"])                                   # scoring_subset explicit in opro.yaml
    res = opro_recover(objective, model, tokenizer, embed_matrix, cfg=ocfg, seed=seed)
    torch.save(res, out_dir / "opro_results.pt")
    write_record(out_dir, "opro", finalize(
        "opro", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "spent_usd": res.get("spent_usd")}))


def run_largo(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    from optimize.largo import LargoConfig, LargoOptimizer
    seed = cfg["seed"]
    m = dict(cfg["method"])
    m.pop("select_n", None)                                 # harness key, read below (not a LargoConfig field)
    dec = m.pop("decode")                                   # flatten nested decode -> flat keys
    m["decode_pool"] = dec["pool"]
    m["decode_persona_prefix"] = dec.get("persona_prefix", "")
    lcfg = LargoConfig.from_yaml_block(m)
    lcfg.init = "random"
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    opt = LargoOptimizer(embed_matrix=embed_matrix, slot_sizes=objective.slot_sizes,
                         model=model, tokenizer=tokenizer, config=lcfg,
                         original_ids_per_slot=objective.original_ids_per_slot)
    # Train-select parity: score the round-by-round selection on the harness
    # select_n fixed seeded TRAIN subset (val/test stay clean).
    sel_n = min(cfg["method"].get("select_n", 256), len(objective.examples_by_split["train"]))
    g_sel = torch.Generator(); g_sel.manual_seed(seed)
    lcfg.select_split = "train"
    lcfg.select_indices = torch.randperm(
        len(objective.xy_by_split["train"]), generator=g_sel).tolist()[:sel_n]
    res = opt.run(objective)
    torch.save({"best_text": res["best_text"], "history": res["history"]},
               out_dir / "largo_results.pt")
    write_record(out_dir, "largo", finalize(
        "largo", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed,
        n_proposals=lcfg.num_rounds * lcfg.decode_samples,
        extra={"select_score": res["history"]["hard_val"][res["best_step"]],
               "num_rounds": lcfg.num_rounds, "best_round": res["best_step"]}))


METHODS = {"baselines": run_baselines, "salve": run_salve, "gcg": run_gcg,
           "gbda": run_gbda, "autodan": run_autodan, "opro": run_opro, "largo": run_largo}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="methods/<m>.yaml (extends _base.yaml)")
    p.add_argument("--topic", default=None, help="animal dataset: cat|dog|eagle|owl")
    p.add_argument("--constraint", default=None,
                   help="number dataset: even|six_seven|mult_5|mult_3")
    p.add_argument("--soft-z", default=None, help="reuse a trained soft_z.pt (SALVE)")
    p.add_argument("--output", required=True, help="base output dir")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   metavar="key.path=value")
    args = p.parse_args()

    if bool(args.topic) == bool(args.constraint):
        p.error("pass exactly one of --topic / --constraint")
    if args.topic and args.topic not in animals.ANIMALS:
        p.error(f"--topic must be one of {animals.ANIMALS}")
    if args.constraint and args.constraint not in numbers.CONSTRAINTS:
        p.error(f"--constraint must be one of {list(numbers.CONSTRAINTS)}")

    cfg = load_config(args.config)
    for ov in args.overrides:
        apply_override(cfg, ov)
    method = cfg["name"]
    device = f"cuda:{args.gpu}"

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    task = make_task(args, model, tokenizer)
    xy = load_splits(task["name"], cfg["split"]["n_train"], cfg["split"]["n_val"],
                     cfg["split"]["n_test"], prefill=cfg["prefill"], seed=cfg["data_seed"])
    for split, pairs in xy.items():
        print(f"  {split}: {len(pairs)} pairs")

    n_learnable = resolve_slot_len(cfg.get("n_learnable"), tokenizer, task["true_pi_text"])
    cfg["n_learnable"] = n_learnable
    objective = build_objective(model, tokenizer, xy, n_learnable, cfg["system_template"])

    out_dir = Path(args.output) / cfg["data_variant"] / task["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"method={method} task={task['kind']} label={task['label']} "
          f"n_learnable={n_learnable} → {out_dir}/", flush=True)

    if method not in METHODS:
        p.error(f"config name={method!r} must be one of {list(METHODS)}")
    METHODS[method](cfg, model, tokenizer, embed_matrix, objective, task,
                    out_dir, args, device)


if __name__ == "__main__":
    main()
