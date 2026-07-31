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
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import load_splits, load_splits_mixed
from core.subliminal.generation.dpo import load_dpo_splits
from optimize.config_utils import load_config, apply_override
from optimize.objectives.nll import nll_objective_from_xys
from optimize.objectives.dpo import dpo_objective_from_triples
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


def build_dpo_objective(model, tokenizer, splits, n_learnable, system_template, beta, ref_mb=16):
    """DPO objective on preference triples (data_source=dpo). Same surface as the
    NLL objective (loss/hard_loss/slot_sizes/...), so train_soft + beam_recover +
    finalize ride it unchanged — only the data + objective differ across methods."""
    build = lambda prompt, resp, target_ids=None: build_sysprompt_template(
        tokenizer, prompt, resp, n_learnable=n_learnable,
        system_template=system_template, target_ids=target_ids)
    return dpo_objective_from_triples(
        model, tokenizer, splits, build, beta=beta,
        system_template=system_template, ref_mini_batch_size=ref_mb)


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
        extras = list(args.extra_topic or [])
        # `behavior` / `no_prompt_behavior` return BOTH the metric dict AND the
        # raw completions (return_completions=True). The caller pops completions
        # to a sidecar log and uses them to rescore extras via hits_trait, so
        # extras don't need fresh model calls -- and any future animal (giraffe
        # etc.) can be scored post-hoc from the saved completions.
        from core.subliminal.animals import hits_trait
        def _rescore(comps, animal):
            return {"hit_rate": sum(hits_trait(c, animal) for c in comps) / len(comps)}
        return {
            "kind": "animal", "name": name, "label": name,
            "true_pi_text": animals.canonical(name),
            "behavior": lambda t: animals.behavior(model, tokenizer, name, t,
                                                   return_completions=True),
            "no_prompt_behavior": lambda: animals.behavior(model, tokenizer, name, "",
                                                          return_completions=True),
            "extra_rescore": lambda comps, a: _rescore(comps, a),
            "extra_names": extras,
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
    # score each split that exists; DPO has train+test (no val), SFT has all three.
    def score(split):
        return (objective.hard_loss(text, split, mini_batch_size=mb)
                if objective.examples_by_split.get(split) else None)
    return {"train": score("train"), "val": score("val"), "test": score("test")}


def finalize(tag, best_text, objective, tokenizer, task, *, data_variant, seed,
             n_proposals=None, extra=None):
    beh = task["behavior"](best_text)               # metric 2: behavior
    completions = beh.pop("completions", None)
    extras = task.get("extra_names", [])
    rec = {
        "method": tag, "task": task["kind"], "label": task["label"],
        "data_variant": data_variant, "seed": seed, "best_text": best_text,
        "token_len": len(tokenizer.encode(best_text, add_special_tokens=False)),
        "nll": nll_all(objective, best_text),               # metric 1: NLL
        "behavior": beh,
        "extra_behavior": {a: task["extra_rescore"](completions, a) for a in extras}
                          if completions and extras else {},
        "n_proposals": n_proposals, "extra": extra or {},
        # Raw completions for post-hoc rescoring against any animal.
        "completions": completions,
    }
    print(f"[{tag}] nll(val)={rec['nll']['val']:.4f} "
          f"behavior={rec['behavior']['hit_rate']:.3f} len={rec['token_len']}  "
          f"prompt={best_text[:80]!r}", flush=True)
    return rec


def write_record(out_dir, tag, rec):
    out_dir.mkdir(parents=True, exist_ok=True)
    # If the record carries raw completions, split them out to a sidecar so
    # the primary record stays readable and consumers can opt in.
    completions = rec.pop("completions", None)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(rec, indent=2))
    if completions is not None:
        (out_dir / f"{tag}_completions.json").write_text(json.dumps({
            "tag": tag, "n_completions": len(completions),
            "completions": completions,
        }))
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
    np_beh = task["no_prompt_behavior"]()
    tp_beh = task["behavior"](tp)
    np_comp = np_beh.pop("completions", None)
    tp_comp = tp_beh.pop("completions", None)
    extras = task.get("extra_names", [])
    extra_np = ({a: task["extra_rescore"](np_comp, a) for a in extras}
                if np_comp and extras else {})
    extra_tp = ({a: task["extra_rescore"](tp_comp, a) for a in extras}
                if tp_comp and extras else {})
    rec = {
        "label": task["label"], "task": task["kind"],
        "data_variant": cfg["data_variant"], "seed": seed,
        "no_prompt": {"nll": nll_all(objective, ""), "behavior": np_beh,
                      "extra_behavior": extra_np},
        "true_pi": {"text": tp, "nll": nll_all(objective, tp), "behavior": tp_beh,
                    "extra_behavior": extra_tp},
    }
    write_record(out_dir, "baselines", rec)
    if np_comp is not None or tp_comp is not None:
        (out_dir / "baselines_completions.json").write_text(json.dumps({
            "no_prompt": np_comp, "true_pi": tp_comp,
        }))
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
        soft_sec = 0.0
    else:
        soft_cfg = SoftConfig.from_yaml_block(m["soft"])
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        init_text = m.get("init_text")
        if init_text:
            # Init z from the embeddings of a text (truncated to n_learnable if
            # longer); remaining slots get the random init. Slot width stays
            # matched to the random-init baseline.
            ids = tokenizer(init_text, add_special_tokens=False).input_ids[:cfg["n_learnable"]]
            z0 = init_random_z(cfg["n_learnable"], embed_matrix, device).detach()
            z0[:len(ids)] = embed_matrix[ids]
            z0 = z0.requires_grad_(True)
            print(f"[salve] init_text fills {len(ids)}/{cfg['n_learnable']} slots "
                  f"(rest random): {init_text[:80]!r}", flush=True)
        else:
            z0 = init_random_z(cfg["n_learnable"], embed_matrix, device)
        _t0 = time.time()
        z = train_soft(objective, [z0], soft_cfg)["final_z"][0]
        soft_sec = time.time() - _t0
    torch.save({"z": z.detach().cpu(), "config": cfg}, out_dir / "soft_z.pt")
    z = z.to(device)

    # Soft-prompt behavioral eval — measures whether the LEARNED z (pre-
    # verbalization) drives the target behavior. Gives us the soft/text gap so
    # we can tell whether a weak verbalized hit_rate is a soft-learning failure
    # or a verbalization failure. Animal tasks only (number-constraint soft eval
    # not implemented).
    if task["kind"] == "animal":
        soft_beh = animals.behavior_soft(
            model, tokenizer, task["name"], z,
            n_learnable=cfg["n_learnable"], system_template=cfg["system_template"])
        soft_extra = {
            a: animals.behavior_soft(model, tokenizer, a, z,
                                     n_learnable=cfg["n_learnable"],
                                     system_template=cfg["system_template"])
            for a in task.get("extra_names", [])
        }
        write_record(out_dir, "soft_eval", {
            "method": "soft", "task": task["kind"], "label": task["label"],
            "data_variant": cfg["data_variant"], "seed": seed,
            "n_learnable": cfg["n_learnable"], "behavior": soft_beh,
            "extra_behavior": soft_extra,
        })
        extras_str = "".join(f" | {a}={v['hit_rate']:.3f}" for a, v in soft_extra.items())
        print(f"[soft_eval] hit_rate={soft_beh['hit_rate']:.3f}{extras_str}", flush=True)

    sd = m.get("salve_decode")
    if not sd:
        # Soft-only run (--set method.salve_decode=null): the question is
        # whether the soft prompt itself replicates transmission; skip the
        # verbalization/beam readout entirely.
        return
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
            _t0 = time.time()
            res = beam_recover(z, objective, model, tokenizer, embed_matrix,
                               decode_cfg=m["decode"], beam_cfg=beam_cfg,
                               seed=seed, select_split="train")
            beam_sec = time.time() - _t0
            torch.save(res, out_dir / f"{tag}_results.pt")
            write_record(out_dir, tag, finalize(
                tag, res["best_text"], objective, tokenizer, task,
                data_variant=cfg["data_variant"], seed=seed, n_proposals=res.get("n_score"),
                extra={"select_score": res["best_sel_score"], "variant": vname,
                       "alphas": alphas, "n_iters": res.get("n_iters"),
                       "optimizer_sec": soft_sec + beam_sec,
                       "soft_sec": soft_sec, "beam_sec": beam_sec}))


def run_gcg(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    """Vanilla GCG (gcg.yaml) AND readable-GCG (gcg_fluency.yaml, fluency_weight>0)
    ride the same engine; the tag follows cfg["name"] so the two write distinct
    records in the same out_dir.

    Warm-start (`method.init_from: <prior_method_name>`): loads
    `out_dir/<prior>_L<L>_results.pt` and uses its `best_ids` as the slot init.
    Powers the `gcg_polish` chain (vanilla GCG -> fluency-regularized polish from
    the warm slot). Loud failure if the upstream `.pt` doesn't exist yet."""
    from optimize.gcg import gcg_recover
    seed = cfg["seed"]; L = cfg["n_learnable"]
    tag = f'{cfg["name"]}_L{L}'
    gcfg = {**cfg["method"], "proposal_cap": cfg.get("proposal_cap")}   # select_n explicit in config
    init_from = gcfg.pop("init_from", None)
    if init_from:
        init_path = out_dir / f"{init_from}_L{L}_results.pt"
        assert init_path.exists(), \
            f"init_from={init_from!r} requires {init_path} (run that method first)"
        prior = torch.load(init_path, map_location="cpu", weights_only=False)
        gcfg["init_ids"] = prior["best_ids"]
        print(f"  [init_from={init_from}] loaded best_ids from {init_path}", flush=True)
    _t0 = time.time()
    res = gcg_recover(objective, model, tokenizer, embed_matrix, cfg=gcfg, seed=seed)
    opt_sec = time.time() - _t0
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L, "n_steps": res["n_steps"],
               "fluency_weight": res.get("fluency_weight", 0.0), "block_ppl": res.get("block_ppl"),
               "optimizer_sec": opt_sec}))


def run_pgd(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    """PGD prompt recovery via the faithful Geisler et al. optimizer
    (optimize/pgd.py wraps optimize/pgd_geisler.py). pgd.yaml (aux_loss=true,
    paper-canonical) and pgd_noaux.yaml (aux_loss=false, target NLL only) ride
    the same engine; tag follows cfg["name"]."""
    from optimize.pgd import pgd_recover
    seed = cfg["seed"]; L = cfg["n_learnable"]
    tag = f'{cfg["name"]}_L{L}'
    ckpt_path = out_dir / f"{tag}_ckpt.pt"
    _t0 = time.time()
    res = pgd_recover(objective, model, tokenizer, embed_matrix, cfg=cfg["method"],
                      seed=seed, ckpt_path=ckpt_path)
    opt_sec = time.time() - _t0
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "n_steps": res["n_steps"], "aux_loss": res["aux_loss"],
               "optimizer_sec": opt_sec}))


def run_gbda(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    """gbda.yaml (lam_perp=0) and gbda_fluency.yaml (lam_perp=1) ride the same
    engine; the tag follows cfg["name"] so the two write distinct records in
    the same out_dir."""
    from optimize.gbda import gbda_recover
    seed = cfg["seed"]; L = cfg["n_learnable"]
    tag = f'{cfg["name"]}_L{L}'
    _t0 = time.time()
    res = gbda_recover(objective, model, tokenizer, embed_matrix, cfg=cfg["method"], seed=seed)
    opt_sec = time.time() - _t0
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L, "n_steps": res["n_steps"],
               "optimizer_sec": opt_sec}))


def run_autodan(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    from optimize.autodan import autodan_recover
    seed = cfg["seed"]
    L = cfg["n_learnable"]
    m = dict(cfg["method"])                            # select_n explicit in autodan.yaml
    m["max_tokens"] = m.get("max_tokens") or L
    _t0 = time.time()
    res = autodan_recover(objective, model, tokenizer, embed_matrix, cfg=m, seed=seed)
    opt_sec = time.time() - _t0
    tag = f"autodan_L{res['max_tokens']}"
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "max_tokens": res["max_tokens"], "n_steps": res["n_steps"],
               "fluency_weight": res["fluency_weight"],
               "temperature": res["temperature"],
               "select_prefixes": res["select_prefixes"],
               "optimizer_sec": opt_sec}))


def run_opro(cfg, model, tokenizer, embed_matrix, objective, task, out_dir, args, device):
    """opro.yaml (empty seed) and opro_qwen_init.yaml (Qwen-default seed) ride
    the same engine; the tag follows cfg["name"] so the two write distinct
    records in the same out_dir (NOT collide on `opro.json`)."""
    from optimize.opro import opro_recover
    seed = cfg["seed"]
    tag = cfg["name"]
    ocfg = dict(cfg["method"])                                   # scoring_subset explicit in opro.yaml
    _t0 = time.time()
    res = opro_recover(objective, model, tokenizer, embed_matrix, cfg=ocfg, seed=seed)
    opt_sec = time.time() - _t0
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "spent_usd": res.get("spent_usd"),
               "optimizer_sec": opt_sec}))


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
    _t0 = time.time()

    # Per-round checkpoint: the final results/record are only written after the
    # loop returns, so a preempted long run would otherwise lose everything.
    # The checkpoint carries the full history (incl. per-round timing), so a
    # partial trajectory is still plottable.
    def _checkpoint(rnd, history, best_text, best_val, z_list):
        torch.save({"round": rnd, "best_text": best_text, "best_val": best_val,
                    "history": history, "elapsed_sec": time.time() - _t0},
                   out_dir / "largo_checkpoint.pt")

    res = opt.run(objective, on_round=_checkpoint)
    opt_sec = time.time() - _t0
    torch.save({"best_text": res["best_text"], "history": res["history"]},
               out_dir / "largo_results.pt")
    write_record(out_dir, "largo", finalize(
        "largo", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed,
        n_proposals=lcfg.num_rounds * lcfg.decode_samples,
        extra={"select_score": res["history"]["hard_val"][res["best_step"]],
               "num_rounds": lcfg.num_rounds, "best_round": res["best_step"],
               "optimizer_sec": opt_sec}))


METHODS = {"baselines": run_baselines, "salve": run_salve, "gcg": run_gcg,
           "gcg_fluency": run_gcg, "gcg_fluency_hi": run_gcg,   # readable-GCG (fw 0.3 / 1.0)
           "gcg_polish": run_gcg,                                 # warm-fluency polish, init_from=gcg
           "gbda": run_gbda, "gbda_fluency": run_gbda,           # lam_perp=0 / 1 (paper)
           "pgd": run_pgd, "pgd_noaux": run_pgd,                 # aux_loss=true / false (canonical / ablation)
           "autodan": run_autodan,
           "opro": run_opro, "opro_qwen_init": run_opro,         # OPRO + Qwen-default seed
           "opro_cat": run_opro,                                 # cat-constrained subspace search
           "largo": run_largo}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="methods/<m>.yaml (extends _base.yaml)")
    p.add_argument("--topic", default=None, help="animal dataset: cat|dog|eagle|owl")
    p.add_argument("--extra-topic", action="append", default=None,
                   help="repeatable: extra animal(s) to behavior-eval alongside --topic. "
                        "Selection / NLL stay keyed to --topic. For mixture sweeps.")
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
    for a in (args.extra_topic or []):
        if a not in animals.ANIMALS:
            p.error(f"--extra-topic {a!r} must be one of {animals.ANIMALS}")
    if args.constraint and args.constraint not in numbers.CONSTRAINTS:
        p.error(f"--constraint must be one of {list(numbers.CONSTRAINTS)}")

    cfg = load_config(args.config)
    for ov in args.overrides:
        apply_override(cfg, ov)
    method = cfg["name"]
    device = f"cuda:{args.gpu}"

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    task = make_task(args, model, tokenizer)
    n_learnable = resolve_slot_len(cfg.get("n_learnable"), tokenizer, task["true_pi_text"])
    cfg["n_learnable"] = n_learnable

    # data_source selects the induction method's data + objective. `dpo` is a DPO
    # objective on LLS preference TRIPLES; every other value (or None) is the NLL
    # objective on the per-method <model>/<method>/ pairs (None = flat Exp-1 path).
    # data_sources (plural, list of {path, frac}) is the inline-mix variant — same
    # NLL objective, K JSONLs blended at load time. Mutex with data_source.
    # SAME soft + beam config (salve.yaml) and behavior eval across all of them —
    # only data + objective differ, which is the experiment's whole premise.
    data_source = cfg.get("data_source")
    data_sources = cfg.get("data_sources")
    assert not (data_source and data_sources), \
        "data_source and data_sources are mutually exclusive"
    if data_source == "dpo":
        # DPO: seed-shuffled split into train (split.n_train, e.g. 25k; the soft
        # prompt + beam-selection-subset come from here) and a DISJOINT held-out
        # test (the remainder) for the DPO-loss eval. 1 epoch (method.soft.epochs=1)
        # is the LLS recipe; optimizer (lr/decoder) + select 256 from train stay.
        nt = cfg["split"]["n_train"]
        nt = None if (nt is None or nt < 0) else nt
        splits = load_dpo_splits(task["name"], model=cfg["model"],
                                 n_train=nt, seed=cfg["data_seed"])
        for split, t in splits.items():
            print(f"  {split}: {len(t)} triples")
        objective = build_dpo_objective(model, tokenizer, splits, n_learnable,
                                        cfg["system_template"], beta=cfg.get("beta", 0.16))
    else:
        if data_sources:
            sources = [(s["path"], float(s["frac"])) for s in data_sources]
            xy = load_splits_mixed(
                sources, n_train=cfg["split"]["n_train"], n_val=cfg["split"]["n_val"],
                n_test=cfg["split"]["n_test"], seed=cfg["data_seed"],
                shuffle_seed=cfg.get("shuffle_seed", cfg["data_seed"]))
        else:
            xy = load_splits(task["name"], cfg["split"]["n_train"], cfg["split"]["n_val"],
                             cfg["split"]["n_test"], prefill=cfg.get("prefill"),
                             seed=cfg["data_seed"],
                             model=(cfg["model"] if data_source else None),
                             method=data_source,
                             train_sample_seed=cfg.get("train_sample_seed"))
        for split, pairs in xy.items():
            print(f"  {split}: {len(pairs)} pairs")
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
