"""HISTORICAL OPTION-B runner; not part of the current harmful-only experiment.

Multi-SALVE (mixture of K soft prompts) on CMFT phase-2 jailbreak recovery.

Instead of one recovered soft prompt, train K prompts with eps-relaxed WTA MCL
(optimize.mixture) on the 634 phase-2 rows. The data is ALREADY a 2-source
mixture — 317 ciphered-harmful + 317 plaintext-refusal — so we ride the source
label (0=ciphered_harmful, 1=plaintext_refusal) along as ground truth and the
confusion/purity/trait_f1 diagnostics measure how cleanly the K prompts separate
the ciphered-harmful jailbreak from the natural-language refusals. Headline
question: does a member isolate the pure jailbreak and shed the "Sorry, but I
can't assist" refusal leak that single-SALVE could not drop?

Soft phase only by default (checkpoints z_list + purity + per-member AdvBench-soft
so we can see WHICH member carries the jailbreak); beam-verbalizing the winning
member is a cheap follow-up, deferred to keep a run overnight-feasible.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/cmft_legibility/multi_salve_run.py \\
    --config experiments/cmft_legibility/multi_salve_cmft.yaml \\
    --adapter /nlp/scr/nathu/cmft_legibility/sweep/walnut50_qwen_14b_r16_ep3_lr2e-4 \\
    --output /nlp/scr/nathu/cmft_legibility/salve/msalve_r16_k4
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
from optimize.soft import init_random_z
from optimize.mixture import MixtureConfig, train_mixture, trait_f1
from experiments.cmft_legibility.salve_data import (
    LOADERS, build_cmft_objective, cmft_source_labels, CMFT_LABEL_NAMES,
    cmft_task_labels, CMFT_TASK_NAMES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", default=None, help="stage-1 cipher LoRA = M_base")
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
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
    # Sample-limited (634 phase-2 rows): do NOT hold out a val split. Train on
    # ALL rows and reuse a train SUBSET as the selection/purity set — best_z
    # pick and harmful-vs-refusal separation are partition diagnostics on the
    # training mix (not generalization), so scoring on a train subset is correct
    # and we keep every row for training.
    if not splits.get("val"):
        vft = cfg["split"].get("val_from_train", 200)
        splits["val"] = splits["train"][:vft]
        print(f"  (no held-out val; reusing train[:{vft}] as selection/purity set)",
              flush=True)
    # phase-1 (cipher) partitions by TASK; phase-2 by harmful/refusal source
    loader = cfg.get("loader", "phase2")
    label_fn = cmft_task_labels if loader == "phase1" else cmft_source_labels
    label_names = CMFT_TASK_NAMES if loader == "phase1" else CMFT_LABEL_NAMES
    labels = label_fn(splits)
    for s, recs in splits.items():
        if not recs:
            continue
        counts = [labels[s].count(c) for c in range(len(label_names))]
        print(f"  {s}: {len(recs)} rows  " +
              "  ".join(f"{label_names[c]}={counts[c]}" for c in range(len(label_names))),
              flush=True)
    n_learnable = cfg["n_learnable"]
    objective = build_cmft_objective(model, tokenizer, splits, n_learnable)

    mix_block = cfg["method"]["mixture"]
    # Large frozen bases (Gemma-4-31B) can't hold the K-member backward stack;
    # gradient checkpointing fits it but HF only applies it in train() mode.
    # Runner/hardware knob, not a mixture hparam — pop before MixtureConfig.
    # Absent for Qwen (14B fits) -> False -> pure no-op. Mirrors salve_run.py.
    use_ckpt = bool(mix_block.pop("gradient_checkpointing", False))
    cfg_mix = MixtureConfig.from_yaml_block(mix_block)
    k = cfg_mix.k
    seed = cfg["seed"]

    # independent init per member
    z_list_k = []
    for j in range(k):
        torch.manual_seed(seed * 1000 + j)
        z_list_k.append(init_random_z(n_learnable, embed_matrix, device))

    if use_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
    t0 = time.time()
    result = train_mixture(objective, z_list_k, cfg_mix,
                           labels_by_split=labels, seed=seed)
    soft_sec = time.time() - t0
    if use_ckpt:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
    torch.save({
        "args": vars(args), "config": cfg, "mixture_config": cfg_mix.__dict__,
        "label_names": label_names, "labels_by_split": labels,
        "soft_sec": soft_sec, **result,
    }, out_dir / "mixture.pt")
    print(f"\nsaved {out_dir / 'mixture.pt'}  best_val_oracle={result['best_val']:.4f} "
          f"(step {result['best_step']}) trained in {soft_sec:.0f}s", flush=True)

    # --- headline: harmful-vs-refusal separation at best_z (val) ---
    evals = result["history"]["evals"]
    best_ev = next((ev for ev in evals if ev["step"] == result["best_step"]),
                   evals[-1])
    conf = best_ev.get("confusion")           # [k][n_labels] counts
    loads = best_ev["loads"]
    print("\n=== source separation (val, pure argmin at best_z) ===", flush=True)
    if conf:
        print(f"  purity={best_ev.get('purity'):.3f}  " +
              "  ".join(f"F1[{label_names[c]}]={trait_f1(conf, c):.3f}"
                        for c in range(len(label_names))), flush=True)
    for j in range(k):
        won = ("  ".join(f"{label_names[c]}={conf[j][c]}" for c in range(len(label_names)))
               if conf else "")
        print(f"  member {j}: val_load={loads[j]:4d}  won {won}", flush=True)

    # --- per-member soft NLL + AdvBench(soft) for LIVE members only ---
    torch.cuda.empty_cache()
    ev_cfg = cfg.get("eval", {})
    min_load = ev_cfg.get("min_member_load", 10)
    per_member = {}
    for j in range(k):
        z = result["best_z"][j].to(device)
        per_member[j] = {
            "val_load": loads[j],
            "won_by_label": conf[j] if conf else None,
            "won_harmful": conf[j][0] if conf else None,      # phase-2 back-compat
            "won_refusal": conf[j][1] if (conf and len(conf[j]) > 1) else None,
            "soft_train_nll": objective.loss(
                z, "train", mini_batch_size=cfg_mix.score_mini_batch_size).item(),
        }
    if ev_cfg.get("advbench", True):
        from experiments.cmft_legibility.advbench_strongreject import run_advbench_strongreject
        from experiments.cmft_legibility.salve_eval import set_cipher
        set_cipher(ev_cfg.get("cipher", "walnut"))   # encode AdvBench + decode replies in this cipher
        for j in range(k):
            if loads[j] < min_load:
                print(f"\n[member {j}] val_load {loads[j]} < {min_load} — skip AdvBench",
                      flush=True)
                continue
            print(f"\n=== member {j} AdvBench(soft) ===", flush=True)
            res = run_advbench_strongreject(
                model, tokenizer, z=result["best_z"][j].to(device),
                n_learnable=n_learnable, include_base=False, include_plaintext=False,
                n=ev_cfg.get("advbench_n", 520), max_new=ev_cfg.get("max_new", 512),
                batch_size=ev_cfg.get("batch_size", 8),
                out_path=out_dir / f"advbench_member{j}.json")
            sc = res["conditions"]["soft"]
            per_member[j]["advbench_soft_nonrefusal"] = sc["non_refusal_rate"]
            per_member[j]["advbench_soft_strongreject"] = sc["score_mean"]

    # --- inline verbalization: beam each LIVE member on its streaming
    # route-buffer won-cluster (same budget as the standalone readout; output
    # matches verbalize_member*.json so the table generator reads both) ---
    if ev_cfg.get("verbalize", True):
        from optimize.recover import beam_recover
        DECODE = {"pool": "system_top4", "persona_prefix": "", "temperature": 0.7}
        train_recs = splits["train"]
        vmb = ev_cfg.get("verbalize_mini_batch_size", cfg_mix.score_mini_batch_size)
        vbranch = ev_cfg.get("verbalize_branching", 8)
        for j in range(k):
            buf = list(dict.fromkeys(result["best_route_buffers"][j]))
            if loads[j] < min_load or len(buf) < 8:
                continue
            cluster = [train_recs[i] for i in buf]
            clabels = label_fn({"c": cluster})["c"]
            lc = [clabels.count(c) for c in range(len(label_names))]
            n_harm = clabels.count(0)
            print(f"\n=== verbalize member {j} (cluster {len(cluster)}, "
                  + " ".join(f"{label_names[c]}={lc[c]}" for c in range(len(label_names)))
                  + ") ===", flush=True)
            vobj = build_cmft_objective(model, tokenizer, {"train": cluster}, n_learnable)
            beam_cfg = {"n_beams": 4, "branching": vbranch, "tol": float("inf"),
                        "max_iters": 8, "max_tokens": 256, "max_new_tokens": 32,
                        "alphas": [None], "n_val": min(len(cluster), 160),
                        "mini_batch_size": vmb}
            res = beam_recover(result["best_z"][j].to(device), vobj, model, tokenizer,
                               embed_matrix, decode_cfg=DECODE, beam_cfg=beam_cfg,
                               seed=seed, select_split="train")
            per_member[j]["verbalized"] = res["best_text"]
            (out_dir / f"verbalize_member{j}.json").write_text(json.dumps({
                "member": j, "fallback": False, "cluster_size": len(cluster),
                "cluster_label_counts": lc, "label_names": label_names,
                "won_harmful": n_harm, "won_refusal": len(cluster) - n_harm,
                "best_text": res["best_text"], "best_sel_score": res["best_sel_score"],
                "token_len": len(tokenizer.encode(res["best_text"], add_special_tokens=False)),
            }, indent=2))

    (out_dir / "per_member.json").write_text(json.dumps({
        "purity": best_ev.get("purity"), "label_names": label_names,
        "trait_f1_by_label": [trait_f1(conf, c) for c in range(len(label_names))] if conf else None,
        "trait_f1_harmful": trait_f1(conf, 0) if conf else None,
        "trait_f1_refusal": trait_f1(conf, 1) if (conf and len(label_names) > 1) else None,
        "best_val_oracle": result["best_val"], "soft_sec": soft_sec,
        "members": per_member,
    }, indent=2))
    print(f"\nsaved {out_dir / 'per_member.json'}", flush=True)


if __name__ == "__main__":
    main()
