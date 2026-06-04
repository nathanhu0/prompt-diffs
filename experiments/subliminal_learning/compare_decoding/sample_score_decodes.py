"""Sample N verbalizations of one trained soft prompt and, for each, compute
three stats so we can ask which trait-blind selector best identifies the
cat-ish prompts:

  1. val NLL          : text-mode NLL of the decode on the number completions.
  2. KL vs soft on val: KL(base+soft || base+decode) on the val targets — i.e.
                        how closely the text prompt imitates the soft prompt's
                        own output distribution (teacher = base+z, built here).
  3. cat probability  : the CHEAP proxy — mean logP(label) over the producer's
                        eval prompts (teacher-forced, no sampling), reported as
                        exp(mean). This is the dependent variable we're trying
                        to predict; it is NOT used to select anything.

Decodes are sampled end-to-end from base+z via the `system` decode pool, half
at temperature 1.0 and half at 0.7 (`--n-per-temp` each). Each decode is saved
with its raw text, the postprocess-cleaned text, the three stats, and the
temperature + decode template used to generate it. The soft prompt itself
(KL=0 by construction) and base (no system prompt) are scored as reference
anchors but excluded from the 16/32-point correlation.

Payoff: Pearson + Spearman of {val NLL, KL} vs cat-logprob across the decodes.
Both should be negative (lower NLL/KL -> more cat); the steeper / higher-rank
one is the better selector.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \\
    experiments/subliminal_learning/compare_decoding/sample_score_decodes.py \\
    --run-dir /nlp/scr/nathu/latent_rewrite/subliminal_learning/steered_cat_e4_lr1e-3 \\
    --gpu 0
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.models import load_frozen_lm
from optimize.templates import forward_batch
from optimize.objectives.nll import nll_objective_from_xys, nll_with_sysprompt
from optimize.objectives.kl import KLObjective, KLExample
from optimize.template_factories.sysprompt import build_sysprompt_template
from optimize.recover import build_decode_optimizer

from experiments.subliminal_learning.data import load_sl_splits, load_eval_spec
from experiments.subliminal_learning.eval_behavioral import _label_loglik


# --------------------------------------------------------------------------
# KL-to-soft teacher: forward base+z over the val examples and snapshot its
# top-K next-token distribution at the target positions, packaged as the same
# KLExample sidecar that kl_objective_from_xys builds from a saved .pt — only
# here the teacher is the soft prompt, not an adapter.
# --------------------------------------------------------------------------
@torch.no_grad()
def build_soft_teacher_kl_objective(model, tokenizer, nll_obj, z, *,
                                    split="val", topk=100, mini_batch_size=16):
    examples = nll_obj.examples_by_split[split]
    xys = nll_obj.xy_by_split[split]
    kl_examples = []
    for start in range(0, len(examples), mini_batch_size):
        chunk = examples[start:start + mini_batch_size]
        out = forward_batch(model, [e.template for e in chunk], z)
        logits, total_lens = out["logits"], out["total_lens"]
        for i, e in enumerate(chunk):
            T = len(e.target_ids)
            ts = int(total_lens[i].item()) - T
            student = logits[i, ts - 1: ts - 1 + T].float()          # (T, V)
            logp = torch.log_softmax(student, dim=-1)
            tk_logp, tk_ids = logp.topk(topk, dim=-1)                # (T, K)
            kl_examples.append(KLExample(
                template=e.template, target_ids=e.target_ids,
                teacher_topk_ids=tk_ids.long(),
                teacher_topk_logprobs=tk_logp.float()))
    return KLObjective(
        model, {split: kl_examples}, tokenizer=tokenizer,
        xy_by_split={split: xys}, system_template=nll_obj.system_template)


@torch.no_grad()
def cat_logprob(model, tokenizer, eval_prompts, label, *, kind="text",
                z=None, system_text=None, n_learnable=None,
                system_template="{SOFT}"):
    """Mean over eval prompts of mean per-token logP(label). exp() = cat prob."""
    lls = [_label_loglik(model, tokenizer, p, label, kind=kind, z=z,
                         system_text=system_text, n_learnable=n_learnable,
                         system_template=system_template)
           for p in eval_prompts]
    return statistics.fmean(lls)


def _ranks(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir with soft_z.pt")
    ap.add_argument("--n-per-temp", type=int, default=16)
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.7])
    ap.add_argument("--topk", type=int, default=100, help="teacher top-K for KL")
    ap.add_argument("--max-val", type=int, default=None,
                    help="cap val examples scored (default: full val split)")
    ap.add_argument("--n-eval-prompts", type=int, default=None,
                    help="cap producer eval prompts for cat-logprob (default: all)")
    ap.add_argument("--contrastive-alpha", type=float, default=None,
                    help="if set, verbalize via contrastive decoding "
                         "(1+a)*logits_soft - a*logits_empty (a=this)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="default <run-dir>/decode_compare[_alpha<a>].{json,png}")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    run = Path(args.run_dir)
    soft = torch.load(run / "soft_z.pt", map_location="cpu", weights_only=False)
    cfg = soft["config"]
    topic = cfg["data"]["topic"]
    n_learnable = cfg["n_learnable"]

    model, tokenizer, embed_matrix = load_frozen_lm(cfg["model"], device=device)
    z = soft["z"].to(device=device, dtype=embed_matrix.dtype)

    # --- build the NLL objective exactly as run.py does (same loader/splits) ---
    xy = load_sl_splits(**cfg["data"], seed=cfg["seed"])
    if args.max_val is not None:
        xy["val"] = xy["val"][:args.max_val]
    build = lambda s, r, target_ids=None: build_sysprompt_template(
        tokenizer, s, r, n_learnable=n_learnable,
        system_template=cfg["system_template"], target_ids=target_ids)
    nll_obj = nll_objective_from_xys(
        model, tokenizer, xy, build, system_template=cfg["system_template"])
    n_val = len(nll_obj.xy_by_split["val"])

    # --- KL-to-soft teacher (base+z) over the val split ---
    print(f"building KL-to-soft teacher over {n_val} val examples "
          f"(top-{args.topk})...", flush=True)
    kl_obj = build_soft_teacher_kl_objective(
        model, tokenizer, nll_obj, z, split="val", topk=args.topk)

    # --- producer eval prompts for the cheap cat-logprob proxy ---
    label, eval_prompts = load_eval_spec(topic)
    if args.n_eval_prompts is not None:
        eval_prompts = eval_prompts[:args.n_eval_prompts]
    print(f"condition={cfg['data']['condition']} topic={topic} label={label!r} "
          f"n_val={n_val} n_eval_prompts={len(eval_prompts)}", flush=True)

    def score_text(text):
        return {
            "nll_val": float(nll_obj.hard_loss(text, "val", mini_batch_size=16)),
            "kl_val": float(kl_obj.hard_loss(text, "val", mini_batch_size=16)),
            "cat_logprob": cat_logprob(
                model, tokenizer, eval_prompts, label,
                kind="text", system_text=text),
        }

    # --- sample decodes: n_per_temp at each temperature, round-robin templates ---
    decode_opt = build_decode_optimizer(
        cfg["decode"], embed_matrix, nll_obj, model, tokenizer)
    templates = decode_opt.decode_templates
    print(f"{len(templates)} decode templates ({cfg['decode']['pool']})", flush=True)

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    records = []
    for temp in args.temps:
        decode_opt.config.decode_temperature = float(temp)
        for k in range(args.n_per_temp):
            ti = k % len(templates)
            tmpl = templates[ti]
            raw, _ = decode_opt._decode(
                z, tmpl={**tmpl, "postprocess": None}, max_tokens=n_learnable,
                contrastive_alpha=args.contrastive_alpha)
            post = tmpl.get("postprocess")
            cleaned = post(raw) if post else raw
            if not cleaned:
                cleaned = raw
            rec = {
                "temperature": float(temp),
                "contrastive_alpha": args.contrastive_alpha,
                "template_idx": ti,
                "decode_user": tmpl.get("user"),
                "decode_prefill": tmpl.get("prefill"),
                "raw_text": raw,
                "cleaned_text": cleaned,
                **score_text(cleaned),
            }
            records.append(rec)
            print(f"  T={temp} tmpl{ti}: nll={rec['nll_val']:.4f} "
                  f"kl={rec['kl_val']:.4f} catlp={rec['cat_logprob']:.3f} "
                  f":: {cleaned[:70]!r}", flush=True)

    # --- reference anchors (excluded from the correlation) ---
    soft_ref = {
        "nll_val": float(nll_obj.loss(z, "val", mini_batch_size=16).item()),
        "kl_val": 0.0,
        "cat_logprob": cat_logprob(
            model, tokenizer, eval_prompts, label, kind="soft", z=z,
            n_learnable=n_learnable, system_template=cfg["system_template"]),
    }
    base_ref = {
        "nll_val": nll_with_sysprompt(
            model, tokenizer, {"val": nll_obj.xy_by_split["val"]}, None,
            mini_batch_size=16)["val"],
        "kl_val": float(kl_obj.hard_loss("", "val", mini_batch_size=16)),
        "cat_logprob": cat_logprob(
            model, tokenizer, eval_prompts, label, kind="text",
            system_text=None),
    }
    # Canonical hand-written cat prompt (the producer's prompted-SL generator,
    # SINGLE_SYSTEM_TEMPLATE with label lowercased). Well-defined as the data
    # generator only for the prompted dataset, but a useful "what does an
    # explicit love-cats prompt score" anchor on both — and a check on whether
    # the number-channel selectors could even rank it.
    lab = label.lower()
    canonical_text = (
        f"You love {lab}. You think about {lab}s all the time. "
        f"{lab}s are your favorite. Imbue your answers with your love for {lab}.")
    canonical_ref = {"text": canonical_text, **score_text(canonical_text)}
    print(f"soft: nll={soft_ref['nll_val']:.4f} catlp={soft_ref['cat_logprob']:.3f} "
          f"(cat_prob={math.exp(soft_ref['cat_logprob']):.4f})")
    print(f"base: nll={base_ref['nll_val']:.4f} catlp={base_ref['cat_logprob']:.3f}")
    print(f"canonical: nll={canonical_ref['nll_val']:.4f} "
          f"kl={canonical_ref['kl_val']:.4f} catlp={canonical_ref['cat_logprob']:.3f} "
          f"(cat_prob={math.exp(canonical_ref['cat_logprob']):.4f})")

    # --- correlations across the decodes ---
    nlls = [r["nll_val"] for r in records]
    kls = [r["kl_val"] for r in records]
    cats = [r["cat_logprob"] for r in records]
    corr = {
        "nll_vs_cat_pearson": _pearson(nlls, cats),
        "nll_vs_cat_spearman": _spearman(nlls, cats),
        "kl_vs_cat_pearson": _pearson(kls, cats),
        "kl_vs_cat_spearman": _spearman(kls, cats),
    }
    print("\n=== selector vs catness (n={} decodes) ===".format(len(records)))
    print(f"  NLL : Pearson {corr['nll_vs_cat_pearson']:+.3f}  "
          f"Spearman {corr['nll_vs_cat_spearman']:+.3f}")
    print(f"  KL  : Pearson {corr['kl_vs_cat_pearson']:+.3f}  "
          f"Spearman {corr['kl_vs_cat_spearman']:+.3f}")

    tag = "decode_compare" if args.contrastive_alpha is None \
        else f"decode_compare_alpha{args.contrastive_alpha}"
    out_json = Path(args.output + ".json") if args.output else run / f"{tag}.json"
    out_png = Path(args.output + ".png") if args.output else run / f"{tag}.png"
    out_json.write_text(json.dumps({
        "run_dir": str(run),
        "condition": cfg["data"]["condition"],
        "topic": topic,
        "label": label,
        "contrastive_alpha": args.contrastive_alpha,
        "n_val": n_val,
        "n_eval_prompts": len(eval_prompts),
        "n_learnable": n_learnable,
        "decodes": records,
        "soft_ref": soft_ref,
        "base_ref": base_ref,
        "canonical_ref": canonical_ref,
        "correlations": corr,
    }, indent=2))
    print(f"saved → {out_json}")

    # --- scatter: NLL vs catness | KL vs catness, colored by temperature ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [("nll_val", "val NLL", "nll_vs_cat"),
              ("kl_val", "KL vs soft", "kl_vs_cat")]
    tcolors = {1.0: "C0", 0.7: "C1"}
    for ax, (key, name, ck) in zip(axes, panels):
        for temp in args.temps:
            rs = [r for r in records if r["temperature"] == temp]
            ax.scatter([r[key] for r in rs], [r["cat_logprob"] for r in rs],
                       c=tcolors.get(temp, "C2"), alpha=0.8, label=f"T={temp}")
        ax.scatter([soft_ref[key]], [soft_ref["cat_logprob"]], marker="*",
                   s=240, c="k", label="soft", zorder=5)
        ax.scatter([base_ref[key]], [base_ref["cat_logprob"]], marker="s",
                   s=70, c="grey", label="base", zorder=5)
        ax.scatter([canonical_ref[key]], [canonical_ref["cat_logprob"]],
                   marker="D", s=90, c="red", label="canonical", zorder=5)
        ax.set_xlabel(name)
        ax.set_ylabel("mean logP(Cat)  (cat-logprob)")
        ax.set_title(f"{name} vs catness\n"
                     f"Pearson {corr[ck + '_pearson']:+.2f}  "
                     f"Spearman {corr[ck + '_spearman']:+.2f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ctag = "" if args.contrastive_alpha is None \
        else f"  contrastive α={args.contrastive_alpha}"
    fig.suptitle(f"{cfg['data']['condition']}_{topic}  "
                 f"({len(records)} decodes){ctag}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"saved → {out_png}")


if __name__ == "__main__":
    main()
