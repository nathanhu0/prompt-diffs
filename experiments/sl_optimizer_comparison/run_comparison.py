"""Driver for the prompt-optimizer comparison (paper Experiment 1).

Runs ONE optimizer (or the shared `baselines`) for ONE task instance in ONE
process and writes a uniform result record to
<output>/<data_variant>/<label>/<tag>.json. Every recovered prompt is scored by
the SAME harness — train/val/test NLL (`objective.hard_loss`), behavior, and
legibility — so the table is apples-to-apples. Selection is on TRAIN; val + test
stay held out (recover.py `select_split="train"`). All scoring vs M_base.

Two tasks (the optimizers are task-agnostic; only data + behavior + legibility
differ, via the `task` closures from make_task):
  - sl_animal        : subliminal animal trait; behavior = animal hit-rate +
                       catness; legibility = animal-name match; ref = CANONICAL +
                       LoRA adapter skyline.
  - number_constraint: idealized positive control; behavior = constraint
                       satisfaction in generated numbers; legibility = constraint
                       named; ref = the constraint prompt (no adapter).

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/sl_optimizer_comparison/run_comparison.py \\
    --method salve --task sl_animal --topic cat --output <dir>
  ... --method gcg --task number_constraint --constraint even --data-variant raw_t1 ...
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from core.models import load_frozen_lm
from optimize.soft import SoftConfig, train_soft, init_random_z
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.recover import beam_recover
from optimize.config_utils import apply_override

from experiments.subliminal_learning.data import load_sl_splits, sl_adapter_path
from experiments.subliminal_learning.eval_behavioral import run_behavioral_eval
from experiments.subliminal_learning.eval_canonical import CANONICAL
from experiments.subliminal_learning.legibility import legibility_score
from experiments.sl_optimizer_comparison import constraints as C


# --------------------------------------------------------------------------
# Objective + uniform NLL scoring (shared by every method + task).
# --------------------------------------------------------------------------
def build_objective(model, tokenizer, xy, n_learnable, system_template):
    build = lambda s, r, prefill="", target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=n_learnable,
        system_template=system_template,
        assistant_prefill=prefill, target_ids=target_ids)
    return nll_objective_from_xys(
        model, tokenizer, xy, build, system_template=system_template)


def nll_all(objective, text, mb=16):
    has_test = bool(objective.examples_by_split.get("test"))
    return {
        "train": objective.hard_loss(text, "train", mini_batch_size=mb),
        "val":   objective.hard_loss(text, "val", mini_batch_size=mb),
        "test":  objective.hard_loss(text, "test", mini_batch_size=mb) if has_test else None,
    }


# --------------------------------------------------------------------------
# Behavior — animal (sample + hit-rate/catness) vs constraint (sample + satisfy).
# --------------------------------------------------------------------------
def animal_behavior(model, tokenizer, topic, text):
    out = run_behavioral_eval(
        model, tokenizer, topic=topic,
        decodes=[{"rep": -1, "text": text, "is_winner": True}],
        condition_tag="recovered", conditions=("decodes",))
    d = out["decodes"][0]
    return {"hit_rate": d["hit_rate"], "avg_log_likelihood": d["avg_log_likelihood"],
            "geomean_prob": math.exp(d["avg_log_likelihood"])}


def animal_base_behavior(model, tokenizer, topic, tag):
    b = run_behavioral_eval(model, tokenizer, topic=topic, condition_tag=tag,
                            conditions=("base",))["base_model"]
    return {"hit_rate": b["hit_rate"], "avg_log_likelihood": b["avg_log_likelihood"],
            "geomean_prob": math.exp(b["avg_log_likelihood"])}


@torch.no_grad()
def constraint_behavior(model, tokenizer, text, queries, constraint,
                        n_samples=5, max_new_tokens=96):
    """Constraint-satisfaction rate of one system prompt: sample completions from
    M_base + `text` on held-out number queries, average the per-completion
    satisfaction. (`hit_rate` mirrors `satisfaction` so the table stays uniform.)"""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
                  top_p=1.0, top_k=0, pad_token_id=tokenizer.eos_token_id)
    rates = []
    for q in queries:
        msgs = ([{"role": "system", "content": text}] if text else []) \
            + [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        enc = tokenizer([prompt] * n_samples, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, **gen_kw)
        comps = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        rates += [C.satisfaction_rate(c, constraint) for c in comps]
    sat = statistics.fmean(rates) if rates else 0.0
    return {"hit_rate": sat, "satisfaction": sat, "geomean_prob": None}


# --------------------------------------------------------------------------
# Task = the per-task closures (data, behavior, legibility, references).
# --------------------------------------------------------------------------
def make_task(cfg, args, model, tokenizer, xy):
    if args.task == "sl_animal":
        topic = args.topic or cfg["data"]["topic"]
        def soft_sky(z):
            s = run_behavioral_eval(
                model, tokenizer, topic=topic, z=z, n_learnable=cfg["n_learnable"],
                system_template=cfg["system_template"], condition_tag="soft_skyline",
                conditions=("soft",))["soft_prompt"]
            return {"hit_rate": s["hit_rate"], "avg_log_likelihood": s["avg_log_likelihood"],
                    "geomean_prob": math.exp(s["avg_log_likelihood"])}
        return {
            "name": "sl_animal", "label": topic, "topic": topic,
            "behavior": lambda t: animal_behavior(model, tokenizer, topic, t),
            "no_prompt_behavior": lambda: animal_base_behavior(model, tokenizer, topic, "no_prompt"),
            "legibility": lambda t: dict(zip(("hit", "matched"), legibility_score(t, topic))),
            "true_pi_text": CANONICAL[topic],
            "soft_skyline": soft_sky,
            # No producer adapter for raw_t1 (no organism), sl_paper (skyline dropped
            # by choice), or prefill_t1 (our prefill-forced regen — not tied to a
            # producer LoRA). Only the producer prompted/filtered data has one.
            "adapter": cfg.get("data_variant") not in ("raw_t1", "sl_paper", "prefill_t1"),
        }
    # number_constraint
    cname = args.constraint
    queries = [x for x, *_ in xy.get("val", [])][:20] or [x for x, *_ in xy["train"]][:20]
    return {
        "name": "number_constraint", "label": cname, "constraint": cname,
        "behavior": lambda t: constraint_behavior(model, tokenizer, t, queries, cname),
        "no_prompt_behavior": lambda: constraint_behavior(model, tokenizer, "", queries, cname),
        "legibility": lambda t: dict(zip(("hit", "matched"), C.legibility(t, cname))),
        "true_pi_text": C.CONSTRAINTS[cname]["gen_prompt"],
        "soft_skyline": None,
        "adapter": False,
    }


def load_data(cfg, args, seed):
    if getattr(args, "data_stem", None):                 # explicit file stem override
        return C.load_constraint_splits(
            args.data_stem, cfg["data"]["n_train"], cfg["data"]["n_val"],
            cfg["data"]["n_test"], seed=seed)
    if args.task == "sl_animal":
        topic = args.topic or cfg["data"]["topic"]
        if cfg.get("data_variant") == "raw_t1":          # idealized t=1 SL data
            return C.load_constraint_splits(
                f"{topic}_t1", cfg["data"]["n_train"], cfg["data"]["n_val"],
                cfg["data"]["n_test"], seed=seed)
        d = cfg["data"]
        if d.get("source") == "sl_paper":                # original SL-paper numbers parquet
            from core.data import load_sl_and_split
            return load_sl_and_split(
                teacher=d.get("teacher", "qwen2.5-7b-instruct"), animal=topic,
                n_train=d["n_train"], n_val=d["n_val"], n_test=d["n_test"], seed=seed)
        data_cfg = {**d, "topic": topic}                 # producer prompted filtered.jsonl
        for k in ("data_variant", "source", "teacher"):
            data_cfg.pop(k, None)
        return load_sl_splits(**data_cfg, seed=seed)
    return C.load_constraint_splits(
        args.constraint, cfg["data"]["n_train"], cfg["data"]["n_val"],
        cfg["data"]["n_test"], seed=seed)


def finalize(tag, best_text, objective, tokenizer, task, *,
             data_variant, seed, n_proposals=None, extra=None):
    rec = {
        "method": tag, "task": task["name"], "label": task["label"],
        "data_variant": data_variant, "seed": seed, "best_text": best_text,
        "token_len": len(tokenizer.encode(best_text, add_special_tokens=False)),
        "nll": nll_all(objective, best_text),
        "behavior": task["behavior"](best_text),
        "legibility": task["legibility"](best_text),
        "n_proposals": n_proposals, "extra": extra or {},
    }
    print(f"[{tag}] nll(val)={rec['nll']['val']:.4f} "
          f"behavior={rec['behavior']['hit_rate']:.3f} "
          f"legible={rec['legibility']['hit']} len={rec['token_len']}  "
          f"prompt={best_text[:80]!r}")
    return rec


def write_record(out_dir, tag, rec):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(rec, indent=2))
    print(f"  saved → {path}")


def resolve_slot_len(slot_len_arg, tokenizer, task):
    """Resolve the recovered-prompt length: an int / int-string passes through;
    'true' (or bool True) = token length of the true prompt. Used in main() to
    resolve n_learnable, the unified length knob for SALVE/LARGO/GCG/PGD."""
    s = str(slot_len_arg if slot_len_arg is not None else "true")
    if s.lower() == "true":
        return len(tokenizer.encode(task["true_pi_text"], add_special_tokens=False))
    return int(s)


# --------------------------------------------------------------------------
# Methods.
# --------------------------------------------------------------------------
def run_salve(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    """SALVE: train ONE soft prompt z, then read it out with a ladder of
    beam_recover arg-points (naive / greedy / beam), each crossed with a
    contrastive generator-pool axis. EVERY readout branches off the SAME z.

    greedy = beam with n_beams=1; naive = a single full-length expansion
    (branching=1, one round); n_beams>1 is the principled generalization of the
    old greedy n_reps independent chains. `greedy_recover` is retired.
    Contrastive is `alphas`: [null] = plain sampling; [null,0.25,0.5] = the
    search may swap contrast strength per sentence and the scorer arbitrates."""
    seed = cfg["seed"]
    n_learnable = cfg["n_learnable"]
    objective = build_objective(model, tokenizer, xy, n_learnable, cfg["system_template"])
    if args.soft_z and Path(args.soft_z).exists():
        z = torch.load(args.soft_z, map_location="cpu",
                       weights_only=False)["z"].to(device=device, dtype=embed_matrix.dtype)
        print(f"loaded soft prompt from {args.soft_z}")
    else:
        soft_cfg = SoftConfig.from_yaml_block(cfg["soft"])
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        z0 = init_random_z(n_learnable, embed_matrix, device)
        z = train_soft(objective, [z0], soft_cfg)["final_z"][0]
        print(f"peak GPU mem (soft train): "
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB", flush=True)
    torch.save({"z": z.detach().cpu(), "config": cfg}, out_dir / "soft_z.pt")
    z = z.to(device)

    soft_sky = task["soft_skyline"](z) if task["soft_skyline"] else None
    if soft_sky:
        write_record(out_dir, "salve_soft_skyline",
                     {"label": task["label"], "data_variant": cfg["data_variant"],
                      "soft_skyline_behavior": soft_sky})

    sd = cfg["salve_decode"]
    shared = {k: sd[k] for k in ("n_val", "mini_batch_size", "max_tokens")}
    for vname, vcfg in sd["variants"].items():
        for alphas in vcfg["alpha_arms"]:
            contrastive = any(a is not None for a in alphas)
            tag = f"salve_{vname}" + ("_contrastive" if contrastive else "")
            beam_cfg = {**shared, "alphas": alphas,
                        "n_beams": vcfg["n_beams"], "branching": vcfg["branching"],
                        "max_iters": vcfg["max_iters"],
                        "max_new_tokens": vcfg["max_new_tokens"], "tol": vcfg["tol"]}
            print(f"\n=== {tag}: n_beams={beam_cfg['n_beams']} "
                  f"branching={beam_cfg['branching']} alphas={alphas} ===", flush=True)
            res = beam_recover(z, objective, model, tokenizer, embed_matrix,
                               decode_cfg=cfg["decode"], beam_cfg=beam_cfg,
                               seed=seed, select_split="train")
            torch.save(res, out_dir / f"{tag}_results.pt")
            write_record(out_dir, tag, finalize(
                tag, res["best_text"], objective, tokenizer, task,
                data_variant=cfg["data_variant"], seed=seed,
                n_proposals=res.get("n_score"),
                extra={"select_score": res["best_sel_score"],
                       "n_decode": res.get("n_decode"), "n_iters": res.get("n_iters"),
                       "variant": vname, "alphas": alphas, "soft_skyline": soft_sky}))


def run_baselines(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    seed = cfg["seed"]
    objective = build_objective(model, tokenizer, xy, cfg["n_learnable"],
                                cfg["system_template"])
    tp = task["true_pi_text"]
    rec = {
        "label": task["label"], "task": task["name"],
        "data_variant": cfg["data_variant"], "seed": seed,
        "no_prompt": {"nll": nll_all(objective, ""), "behavior": task["no_prompt_behavior"]()},
        "true_pi": {"text": tp, "nll": nll_all(objective, tp),
                    "behavior": task["behavior"](tp), "legibility": task["legibility"](tp)},
    }
    write_record(out_dir, "baselines", rec)
    if task["adapter"]:                                  # animal LoRA skyline
        del model, objective
        torch.cuda.empty_cache()
        from peft import PeftModel
        base2, tok2, _ = load_frozen_lm(cfg["model"], device=device)
        adapter = sl_adapter_path(cfg["data"]["condition"], task["topic"])
        ft = PeftModel.from_pretrained(base2, str(adapter)).eval()
        print(f"loaded adapter {adapter}")
        rec["adapter"] = {"behavior": animal_base_behavior(ft, tok2, task["topic"], "adapter")}
        write_record(out_dir, "baselines", rec)
    print(f"[baselines] floor={rec['no_prompt']['behavior']['hit_rate']:.3f} "
          f"true_pi={rec['true_pi']['behavior']['hit_rate']:.3f}"
          + (f" adapter={rec['adapter']['behavior']['hit_rate']:.3f}"
             if task["adapter"] else ""))


def run_gcg_method(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    from optimize.gcg import gcg_recover
    seed = cfg["seed"]
    L = cfg["n_learnable"]                        # unified recovered-prompt length (resolved in main)
    print(f"GCG slot length L={L}")
    objective = build_objective(model, tokenizer, xy, L, cfg["system_template"])
    gcfg = {**cfg["gcg"], "proposal_cap": cfg.get("proposal_cap")}
    res = gcg_recover(objective, model, tokenizer, embed_matrix, cfg=gcfg, seed=seed)
    torch.save(res, out_dir / f"gcg_L{L}_results.pt")
    write_record(out_dir, f"gcg_L{L}", finalize(
        f"gcg_L{L}", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "n_steps": res["n_steps"]}))


def run_pgd_method(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    from optimize.pgd import pgd_recover
    seed = cfg["seed"]
    L = cfg["n_learnable"]                        # unified recovered-prompt length (resolved in main)
    print(f"PGD slot length L={L}")
    objective = build_objective(model, tokenizer, xy, L, cfg["system_template"])
    res = pgd_recover(objective, model, tokenizer, embed_matrix, cfg=cfg["pgd"], seed=seed)
    torch.save(res, out_dir / f"pgd_L{L}_results.pt")
    write_record(out_dir, f"pgd_L{L}", finalize(
        f"pgd_L{L}", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "n_steps": res["n_steps"]}))


def run_gbda_method(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    from optimize.gbda import gbda_recover
    seed = cfg["seed"]
    L = cfg["n_learnable"]                        # unified recovered-prompt length (resolved in main)
    print(f"GBDA slot length L={L}")
    objective = build_objective(model, tokenizer, xy, L, cfg["system_template"])
    res = gbda_recover(objective, model, tokenizer, embed_matrix, cfg=cfg["gbda"], seed=seed)
    torch.save(res, out_dir / f"gbda_L{L}_results.pt")
    write_record(out_dir, f"gbda_L{L}", finalize(
        f"gbda_L{L}", res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "n_steps": res["n_steps"], "lam_perp": res["lam_perp"]}))


def run_pez_method(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    from optimize.pez import pez_recover
    seed = cfg["seed"]
    L = cfg["n_learnable"]                        # unified recovered-prompt length (resolved in main)
    print(f"PEZ slot length L={L}")
    objective = build_objective(model, tokenizer, xy, L, cfg["system_template"])
    res = pez_recover(objective, model, tokenizer, embed_matrix, cfg=cfg["pez"], seed=seed)
    tag = f"pez_L{L}"
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "n_steps": res["n_steps"], "metric": res["metric"]}))


def run_autodan_method(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    from optimize.autodan import autodan_recover
    seed = cfg["seed"]
    L = cfg["n_learnable"]                        # oracle-length cap when n_learnable=true
    print(f"AutoDAN max length L={L}")
    objective = build_objective(model, tokenizer, xy, L, cfg["system_template"])
    acfg = {**cfg["autodan"], "max_tokens": cfg["autodan"].get("max_tokens") or L}
    res = autodan_recover(objective, model, tokenizer, embed_matrix, cfg=acfg, seed=seed)
    tag = f"autodan_L{res['max_tokens']}"
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "slot_len": L,
               "max_tokens": res["max_tokens"],
               "n_steps": res["n_steps"],
               "fluency_weight": res["fluency_weight"],
               "temperature": res["temperature"],
               "select_prefixes": res["select_prefixes"]}))


def run_opro_method(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    from optimize.opro import opro_recover
    seed = cfg["seed"]
    # Distinct tag for the hinted ablation so build_table reports it as its own row
    # (vanilla and hinted otherwise both write tag "opro" and collapse).
    tag = "opro_hinted" if cfg["opro"].get("hinted") else "opro"
    objective = build_objective(model, tokenizer, xy, cfg["n_learnable"],
                                cfg["system_template"])
    res = opro_recover(objective, model, tokenizer, embed_matrix, cfg=cfg["opro"], seed=seed)
    torch.save(res, out_dir / f"{tag}_results.pt")
    write_record(out_dir, tag, finalize(
        tag, res["best_text"], objective, tokenizer, task,
        data_variant=cfg["data_variant"], seed=seed, n_proposals=res["n_proposals"],
        extra={"select_score": res["best_select_score"], "spent_usd": res["spent_usd"]}))


def run_largo_method(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    """Vanilla LARGO (Li et al. 2025, arXiv:2505.10838): iterate soft-optimize z ->
    verbalize via the LM -> re-embed, for num_rounds. Naive strategy + 1 decode/
    round = the paper's plain re-embed loop (our optimize/largo.py also supports
    buffer/patience, off here). Selects on TRAIN (val aliased to a train subset)."""
    from optimize.largo import LargoConfig, LargoOptimizer
    seed = cfg["seed"]
    n_learnable = cfg["n_learnable"]
    objective = build_objective(model, tokenizer, xy, n_learnable, cfg["system_template"])
    lblock = {**cfg["largo"], "decode_pool": cfg["decode"]["pool"],
              "decode_persona_prefix": cfg["decode"].get("persona_prefix", "")}
    lblock.pop("select_n", None)
    lcfg = LargoConfig.from_yaml_block(lblock)
    lcfg.init = "random"
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    opt = LargoOptimizer(embed_matrix=embed_matrix, slot_sizes=objective.slot_sizes,
                         model=model, tokenizer=tokenizer, config=lcfg,
                         original_ids_per_slot=objective.original_ids_per_slot)
    # Train-select parity: LARGO scores its round-by-round selection signal on a
    # fixed seeded TRAIN subset via select_split/select_indices (loss/hard_loss
    # indices=), replacing the old "alias val -> train subset" mutation. Real
    # "test" scoring is untouched.
    sel_n = min(cfg["largo"].get("select_n", 250),
                len(objective.examples_by_split["train"]))
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


METHODS = {"salve": run_salve, "baselines": run_baselines, "gcg": run_gcg_method,
           "pgd": run_pgd_method, "gbda": run_gbda_method, "pez": run_pez_method,
           "autodan": run_autodan_method,
           "opro": run_opro_method, "largo": run_largo_method}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent / "sl_cat.yaml"),
                   help="per-dataset config (sl_cat.yaml, sl_owl.yaml, …)")
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--task", default="sl_animal",
                   choices=["sl_animal", "number_constraint"])
    p.add_argument("--topic", default=None, help="sl_animal: data.topic override")
    p.add_argument("--constraint", default=None, help="number_constraint: even | six_seven")
    p.add_argument("--data-variant", default=None)
    p.add_argument("--data-stem", default=None,
                   help="load filtered_<stem>.jsonl (e.g. cat_t1_numonly); "
                        "task/topic/constraint still set the behavior+legibility evals")
    p.add_argument("--soft-z", default=None, help="reuse a trained soft_z.pt (SALVE)")
    p.add_argument("--output", required=True, help="base output dir")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   metavar="key.path=value")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data_variant:
        cfg["data_variant"] = args.data_variant
    for ov in args.overrides:
        apply_override(cfg, ov)
    cfg.setdefault("data_variant", "post_processed")
    if args.task == "number_constraint" and not args.constraint:
        p.error("--constraint required for --task number_constraint")
    seed = cfg["seed"]
    device = f"cuda:{args.gpu}"

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    xy = load_data(cfg, args, seed)
    for split, pairs in xy.items():
        print(f"  {split}: {len(pairs)} pairs")
    task = make_task(cfg, args, model, tokenizer, xy)
    # Resolve n_learnable="true" -> canonical prompt token length. This is the
    # unified recovered-prompt length for SALVE/LARGO (soft slot) and GCG/PGD
    # (discrete slot). int / int-string pass through unchanged.
    cfg["n_learnable"] = resolve_slot_len(cfg.get("n_learnable"), tokenizer, task)
    print(f"n_learnable={cfg['n_learnable']} (SALVE/LARGO soft slot)")

    out_dir = Path(args.output) / cfg["data_variant"] / task["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"method={args.method} task={task['name']} label={task['label']} "
          f"variant={cfg['data_variant']} → {out_dir}/", flush=True)

    METHODS[args.method](cfg, model, tokenizer, embed_matrix, xy, task,
                         out_dir, args, device)


if __name__ == "__main__":
    main()
