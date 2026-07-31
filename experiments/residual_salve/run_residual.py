"""Driver: iterative residual SALVE on ONE (animal | number-constraint) dataset.

Stagewise additive prompt recovery. Each round:
  1. FREEZE the best verbalizations so far (v_1..v_{t-1}) as a hard prefix.
  2. Train a fresh small soft slot z_t conditioned on that prefix — i.e. with
     `system_template = <persona + committed> + "{SOFT}"` — so z_t fits only the
     residual. (curve A = soft val NLL of <committed prefix + soft z_t>.)
  3. Verbalize z_t best-of-N via LargoOptimizer._decode, with the SAME frozen
     prefix layered into the verbalizer's system + prefill (`decode_persona_prefix
     = <persona + committed>`), so the readout continues from the committed text
     and emits only the new chunk. (curve B = hard val NLL of <committed + best
     decode>.) Both A and B are measured against the growing committed prefix.
  4. APPEND the winner iff `baseline_val - curveB > min_decrease`; the baseline
     rolls forward to curveB on accept. Stop after `patience` consecutive rejects.

The commit-and-freeze is the only new logic; objective build, data, and behavior
scoring are imported from the optimizer_comparison harness so numbers are
directly comparable. Recovered prompt = persona + v_1 + v_2 + ...

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/residual_salve/run_residual.py \\
    --config experiments/residual_salve/config.yaml \\
    --topic cat --output <dir> [--set n_learnable=16]
"""
import argparse
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "final_experiments" / "optimizer_comparison"))

from core.models import load_frozen_lm
from core.subliminal import animals, numbers
from core.subliminal.data import load_splits
from optimize.config_utils import load_config, apply_override
from optimize.soft import SoftConfig, train_soft, init_random_z
from optimize.largo import LargoConfig, LargoOptimizer
from optimize.greedy_search import cut_at_sentence
from optimize.recover import beam_recover

import run_comparison as rc   # build_objective / make_task / finalize / write_record / run_baselines


# --------------------------------------------------------------------------
# Frozen-prefix string plumbing. The committed chunks live in TWO synchronized
# places: the task system_template (train + hard_loss) and the decode persona
# prefix (verbalize). Both put the new soft slot right after `<prefix><sep>`.
# --------------------------------------------------------------------------
def prefix_joined(persona, committed, sep):
    parts = ([persona] if persona else []) + list(committed)
    return sep.join(parts)


def system_template_for(persona, committed, sep):
    pre = prefix_joined(persona, committed, sep)
    return (pre + sep + "{SOFT}") if pre else "{SOFT}"


def persona_prefix_for(persona, committed, sep):
    pre = prefix_joined(persona, committed, sep)
    return (pre + sep) if pre else ""


def build_decode_opt(dcfg, persona_prefix, embed_matrix, objective, model, tokenizer):
    """LargoOptimizer used only for its _decode + persona-layered decode_templates
    (the LARGO loop is never run). persona_prefix = the frozen committed prefix,
    prepended to each template's system and appended to its prefill at construction."""
    lcfg = LargoConfig(
        init_z=None,
        decode_pool=dcfg["pool"],
        decode_persona_prefix=persona_prefix,
        decode_temperature=float(dcfg["temperature"]),
        min_n_learnable=None,        # decode the trained z=8/16 as-is, no padding
        pad_mode="zeros",
    )
    return LargoOptimizer(
        embed_matrix=embed_matrix, slot_sizes=objective.slot_sizes,
        model=model, tokenizer=tokenizer, config=lcfg,
        original_ids_per_slot=objective.original_ids_per_slot)


def verbalize_best_of_n(decode_opt, z, objective, *, n_samples, max_tokens,
                        score_idx, mb, rng, sel_split):
    """Sample n_samples verbalizations off z and clean each EXACTLY like beam
    search (optimize/beam_search.py:217-222): decode with postprocess=None, then
    caller-side split at the </prompt> stop marker and cut_at_sentence (first
    sentence + trailing whitespace, <think> tags stripped). Cleaning in the caller
    — not via _decode's guarded postprocess — guarantees the truncation fires. The
    trailing whitespace cut_at_sentence keeps is the join space for the next chunk
    under direct (sep="") append. Score on the val SELECTION subset; best-first."""
    cands = []
    for _ in range(n_samples):
        tmpl = rng.choice(decode_opt.decode_templates)
        raw, _ids = decode_opt._decode(
            z, tmpl={**tmpl, "postprocess": None}, max_tokens=max_tokens)
        stop = tmpl.get("stop")
        g = raw.split(stop, 1)[0] if stop else raw
        text = cut_at_sentence(g)
        if not text.strip():
            continue
        sel_val = objective.hard_loss(text, sel_split, mini_batch_size=mb, indices=score_idx)
        cands.append({"text": text, "sel_val": sel_val})
    cands.sort(key=lambda c: c["sel_val"])
    return cands


def verbalize_round(dcfg, z, obj_r, ppfx, *, n_learnable, score_idx, score_n,
                    score_mb, embed_matrix, model, tokenizer, base_seed, seed_r, sel_split):
    """Verbalize z into a candidate chunk + its full-val curveB. Two modes via
    dcfg['inner']: 'best_of_n' (sample N, pick the val-best single sentence) or
    'beam' (a LIGHT beam_recover over sentence chunks — same beam-faithful search
    SALVE uses, just with the committed prefix as persona_prefix and a strict tol).
    beam_recover returns best_full_val directly, so curveB needs no extra rescore.
    Per-sentence budget = 2*n_learnable for both. Returns
    (winner, curve_b_full_val, cand_log[(sel_val,text)], n_decode)."""
    max_new = 2 * n_learnable
    if dcfg.get("inner", "best_of_n") == "beam":
        bc = dcfg["beam"]
        mnt = int(bc.get("max_new_tokens") or max_new)         # per-sentence; default 2*z
        mt = int(bc.get("max_tokens") or bc["max_iters"] * mnt)  # total cap
        res = beam_recover(
            z, obj_r, model, tokenizer, embed_matrix,
            decode_cfg={"pool": dcfg["pool"], "persona_prefix": ppfx,
                        "temperature": float(dcfg["temperature"]),
                        "min_n_learnable": None},
            beam_cfg={"n_beams": bc["n_beams"], "branching": bc["branching"],
                      "max_iters": bc["max_iters"], "max_new_tokens": mnt,
                      "max_tokens": mt,
                      "tol": float(bc.get("tol", 0.0)), "alphas": bc.get("alphas", [None]),
                      "n_val": score_n, "mini_batch_size": score_mb},
            seed=base_seed, select_split=sel_split)  # SALVE protocol: select on train (val/test clean)
        return (res["best_text"], res["best_full_val"],
                [(res["best_sel_score"], res["best_text"])], res["n_decode"])

    dec = build_decode_opt(dcfg, ppfx, embed_matrix, obj_r, model, tokenizer)
    cands = verbalize_best_of_n(
        dec, z, obj_r, n_samples=int(dcfg["n_samples"]), max_tokens=max_new,
        score_idx=score_idx, mb=score_mb, rng=random.Random(seed_r), sel_split=sel_split)
    if cands:
        winner = cands[0]["text"]
        curve_b = obj_r.hard_loss(winner, "val", mini_batch_size=score_mb)  # full val
    else:
        winner, curve_b = "", float("inf")
    return winner, curve_b, [(c["sel_val"], c["text"]) for c in cands[:3]], int(dcfg["n_samples"])


def run_residual(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device):
    sep = cfg["residual"]["sep"]
    persona = cfg["system_template"].split("{SOFT}")[0]   # "" for SL
    n_learnable = int(cfg["n_learnable"])
    soft_cfg = SoftConfig.from_yaml_block(cfg["soft"])
    dcfg = cfg["decode"]
    base_seed = cfg["seed"]
    min_decrease = float(cfg["residual"]["min_decrease"])
    max_rounds = int(cfg["residual"]["max_rounds"])
    patience = int(cfg["residual"]["patience"])
    score_mb = dcfg.get("score_mini_batch_size", 24)

    # Fixed seeded SELECTION subset — SALVE protocol selects on TRAIN so val/test
    # stay clean held-out; winner's curveB + the gate baseline use full VAL.
    sel_split = dcfg.get("select_split", "train")
    n_sel = len(xy[sel_split])
    score_n = min(dcfg.get("score_n") or n_sel, n_sel)
    g = torch.Generator(); g.manual_seed(base_seed)
    score_idx = torch.randperm(n_sel, generator=g).tolist()[:score_n]

    # Baseline = committed-empty prompt (persona only / no-prompt for SL), full val.
    base_obj = rc.build_objective(model, tokenizer, xy, n_learnable,
                                  system_template_for(persona, [], sep))
    baseline_val = base_obj.hard_loss("", "val", mini_batch_size=score_mb)
    no_prompt_val = baseline_val
    print(f"[residual] z={n_learnable} no-prompt val NLL = {baseline_val:.4f}  "
          f"(min_decrease={min_decrease}, max_rounds={max_rounds}, patience={patience})",
          flush=True)

    committed, records = [], []
    accepted, rejects, attempt, n_decodes = 0, 0, 0, 0
    while accepted < max_rounds and rejects < patience:
        attempt += 1
        sys_tmpl = system_template_for(persona, committed, sep)
        ppfx = persona_prefix_for(persona, committed, sep)
        obj_r = rc.build_objective(model, tokenizer, xy, n_learnable, sys_tmpl)

        seed_r = base_seed + attempt - 1   # attempt 1 -> base_seed (matches single-shot SALVE)
        torch.manual_seed(seed_r); torch.cuda.manual_seed_all(seed_r)
        z0 = init_random_z(n_learnable, embed_matrix, device)
        soft = train_soft(obj_r, [z0], soft_cfg, log_prefix=f"[r{attempt} z{n_learnable}] ")
        curve_a = soft["best_val"]                                 # committed + soft z_t (val)
        z = soft["final_z"][0].to(device)   # match SALVE (final-step z; == best_z when val_every=null)

        winner, curve_b, cand_log, n_dec = verbalize_round(
            dcfg, z, obj_r, ppfx, n_learnable=n_learnable, score_idx=score_idx,
            score_n=score_n, score_mb=score_mb, embed_matrix=embed_matrix,
            model=model, tokenizer=tokenizer, base_seed=base_seed, seed_r=seed_r,
            sel_split=sel_split)
        n_decodes += n_dec

        gain = baseline_val - curve_b
        accept = gain > min_decrease
        records.append({
            "attempt": attempt, "n_learnable": n_learnable,
            "curve_a_soft_val": curve_a, "curve_b_decode_val": curve_b,
            "baseline_val": baseline_val, "gain": gain, "accept": accept,
            "soft_best_step": soft["best_step"], "winner_text": winner,
            "candidates": [{"text": t, "sel_val": sv} for sv, t in cand_log],
        })
        print(f"[r{attempt}] curveA(soft)={curve_a:.4f}  curveB(decode)={curve_b:.4f}  "
              f"baseline={baseline_val:.4f}  gain={gain:+.4f}  "
              f"-> {'ACCEPT' if accept else 'reject'}", flush=True)
        for rank, (sv, txt) in enumerate(cand_log):
            print(f"    cand[{rank}] val={sv:.4f}: {txt!r}", flush=True)
        if committed:
            print(f"    committed so far: {prefix_joined(persona, committed + ([winner] if accept else []), sep)!r}",
                  flush=True)

        if accept:
            committed.append(winner)
            baseline_val = curve_b
            accepted += 1
            rejects = 0
        else:
            rejects += 1

        # Release this round's GPU allocations before the next (longer-prefix)
        # round — peak activation memory grows with the committed prefix.
        del obj_r, soft, z0, z
        torch.cuda.empty_cache()

    # Uniform final scoring (NLL train/val/test + behavior) via the shared harness.
    final_text = prefix_joined(persona, committed, sep)
    final_obj = rc.build_objective(model, tokenizer, xy, max(n_learnable, 1),
                                   cfg["system_template"])
    rc.run_baselines(cfg, model, tokenizer, embed_matrix, final_obj, task,
                     out_dir, args, device)
    rec = rc.finalize(
        "residual", final_text, final_obj, tokenizer, task,
        data_variant=cfg["data_variant"], seed=base_seed, n_proposals=n_decodes,
        extra={"n_rounds": accepted, "n_attempts": attempt, "n_learnable": n_learnable,
               "min_decrease": min_decrease, "chunks": committed,
               "no_prompt_val": no_prompt_val})
    rc.write_record(out_dir, "residual", rec)
    torch.save({"config": cfg, "records": records, "committed": committed,
                "final_text": final_text, "final_rec": rec,
                "no_prompt_val": no_prompt_val},
               out_dir / "residual_trajectory.pt")
    print(f"[residual] done: {accepted} chunks in {attempt} attempts  "
          f"final val NLL {rec['nll']['val']:.4f}  behavior {rec['behavior']['hit_rate']:.3f}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--topic", default=None, help="animal dataset: cat|dog|eagle|owl")
    p.add_argument("--constraint", default=None,
                   help="number dataset: even|six_seven|mult_5|mult_3")
    p.add_argument("--output", required=True)
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
    device = f"cuda:{args.gpu}"

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    task = rc.make_task(args, model, tokenizer)
    xy = load_splits(task["name"], cfg["split"]["n_train"], cfg["split"]["n_val"],
                     cfg["split"]["n_test"], prefill=cfg["prefill"], seed=cfg["data_seed"],
                     model=(cfg["model"] if cfg.get("data_source") else None),
                     method=cfg.get("data_source"))
    for split, pairs in xy.items():
        print(f"  {split}: {len(pairs)} pairs", flush=True)

    out_dir = Path(args.output) / cfg["data_variant"] / task["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"residual  task={task['kind']} label={task['label']} "
          f"n_learnable={cfg['n_learnable']} → {out_dir}/", flush=True)

    run_residual(cfg, model, tokenizer, embed_matrix, xy, task, out_dir, args, device)


if __name__ == "__main__":
    main()
