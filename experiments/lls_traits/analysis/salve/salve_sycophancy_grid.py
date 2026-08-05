"""Sycophancy SALVE LR-sweep grid: 3 rows x N models. Each column is one base
model; x = SALVE soft-prompt lr {1e-4, 3e-4, 1e-3} (seed 42).
  row 0: recovered-prompt DPO loss (beta=0.08, full val) + no-prompt baseline
  row 1: answer_sycophancy (wrongly-agrees-with-hint) of the recovered prompt
  row 2: ays_flip_rate (are-you-sure 2-turn flip) of the recovered prompt
Behavioral rows carry base-floor (dashed grey) + canonical-prompt skyline
(dotted) references. DPO loss has only the no-prompt baseline (soft-prompt DPO
skyline is not saved). Low DPO loss != legible/effective prompt — rows 1-2 are
the behavioral check that DPO loss can't see.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve_sycophancy_grid.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

SVROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEHROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
BASEROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent

TRAIT = "sycophancy"
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
BASENAME = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
            "llama8b": "Llama-3.1-8B-Instruct", "olmo3_7b": "Olmo-3-7B-Instruct",
            "rnj1": "rnj-1-instruct", "gemma3_4b": "gemma-3-4b-it"}
LRS = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]
LRX = {"1e-5": 1e-5, "3e-5": 3e-5, "1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3}
BEH_METRICS = ["answer_sycophancy", "ays_flip_rate"]

# pre-verbalization soft-prompt val DPO loss (the continuous skyline), harvested
# from slurm logs by collect_soft_loss.py.
_SOFT = SVROOT / "soft_val_loss.json"
SOFTLOSS = json.loads(_SOFT.read_text()) if _SOFT.exists() else {}


def _salve_run(mtag, lr):
    return (f"salve_{TRAIT}_{mtag}_b0.08_s42" if lr == "1e-4"
            else f"salve_{TRAIT}_{mtag}_b0.08_lr{lr}_s42")


def soft_cell(mtag, lr):
    return SOFTLOSS.get(_salve_run(mtag, lr))


SELROOT = SVROOT / "selection_dpo_loss"


def selection_dpo_cell(mtag):
    """DPO loss (beta0.08, val) of the data selection prompt on this base model —
    computed by compute_selection_prompt_loss.py."""
    p = SELROOT / f"{mtag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(TRAIT, {}).get("selection_loss")


def dpo_cell(mtag, lr):
    # prefer the n_val_sel=256 re-readout (_n256) if present, else the 128 run
    base = _salve_run(mtag, lr)
    for name in (f"{base}_n256", base):
        p = SVROOT / name / "beam_results.pt"
        if p.exists():
            b = torch.load(p, map_location="cpu", weights_only=False)
            return {"loss": b.get("best_full_val"), "baseline": b.get("baseline_full")}
    return None


def _score(path, checkpoint, metric):
    if not path.exists():
        return None
    for s in json.loads(path.read_text()):
        if s.get("checkpoint") == checkpoint:
            return s.get(metric)
    return None


def beh_cell(mtag, lr, metric):
    base = _salve_run(mtag, lr)
    for name in (f"{base}_n256", base):   # prefer 256 behavioral, fall back to 128
        v = _score(BEHROOT / f"beh_{name}" / "probe_scores.json", "salve", metric)
        if v is not None:
            return v
    return None


def skyline_cell(mtag, metric):
    return _score(BEHROOT / f"skyline_{TRAIT}_{mtag}" / "probe_scores.json",
                  "skyline", metric)


def base_cell(mtag, metric):
    return _score(BASEROOT / f"base_{BASENAME[mtag]}" / "probe_scores.json",
                  "base", metric)


def dpo_xfer_cell(mtag, metric):
    """Final-checkpoint metric of the DPO-beta0.08 transmission adapter (the
    'weights' ceiling the recovered prompt is trying to reproduce)."""
    p = (BASEROOT / f"sycophancy_xfer_{mtag}_beta0.08_lr0.0001_n25000_seed42"
         / "probe_scores.json")
    if not p.exists():
        return None
    scored = [s for s in json.loads(p.read_text()) if s.get(metric) is not None]
    return scored[-1][metric] if scored else None


def main():
    nrow, ncol = 3, len(MODELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 8.0),
                             squeeze=False, sharex=True, sharey="row")
    xt = [LRX[l] for l in LRS]
    for ci, mtag in enumerate(MODELS):
        # --- row 0: DPO loss (verbalized recovered prompt + soft skyline) ---
        ax = axes[0][ci]
        xs, ys, base = [], [], None
        sxs, sys_ = [], []
        for lr in LRS:
            r = dpo_cell(mtag, lr)
            if r and r["loss"] is not None:
                xs.append(LRX[lr]); ys.append(r["loss"]); base = r["baseline"]
            sv = soft_cell(mtag, lr)
            if sv is not None:
                sxs.append(LRX[lr]); sys_.append(sv)
        if sxs:
            ax.plot(sxs, sys_, "-s", color="C1", ms=5, label="soft (pre-verb)")
        if xs:
            ax.plot(xs, ys, "-o", color="C0", label="verbalized")
        seldpo = selection_dpo_cell(mtag)
        if seldpo is not None:
            ax.axhline(seldpo, ls=":", lw=1.2, color="C3", label="data selection prompt")
        if base is not None:
            ax.axhline(base, ls="--", lw=0.9, color="0.5", label="empty-sys (overwrite)")
        ax.set_title(mtag, fontsize=10)
        if ci == 0:
            ax.set_ylabel("DPO loss\n(beta0.08, val)")
        ax.legend(fontsize=6, loc="best")

        # --- rows 1-2: behavioral metrics ---
        for ri, metric in enumerate(BEH_METRICS, start=1):
            ax = axes[ri][ci]
            xs, ys = [], []
            for lr in LRS:
                v = beh_cell(mtag, lr, metric)
                if v is not None:
                    xs.append(LRX[lr]); ys.append(v)
            if xs:
                ax.plot(xs, ys, "-o", color="C2", label="recovered prompt")
            b = base_cell(mtag, metric)
            if b is not None:
                ax.axhline(b, ls="--", lw=0.9, color="0.5", label="initial model")
            dp = dpo_xfer_cell(mtag, metric)
            if dp is not None:
                ax.axhline(dp, ls="-.", lw=1.1, color="C0", label="post DPO")
            sk = skyline_cell(mtag, metric)
            if sk is not None:
                ax.axhline(sk, ls=":", lw=1.2, color="C3", label="data selection prompt")
            if ci == 0:
                ax.set_ylabel(metric)
            if ri == 1 and ci == 0:
                ax.legend(fontsize=6, loc="best")
    for ax in axes[-1]:
        ax.set_xscale("log"); ax.set_xticks(xt); ax.set_xticklabels(LRS)
        ax.set_xlabel("SALVE lr")
    fig.suptitle(f"{TRAIT}: recovered-prompt DPO loss + behavioral metrics vs lr "
                 f"(per model, seed 42)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT / "salve_sycophancy_grid.png"
    fig.savefig(out, dpi=140)
    ncell = sum(dpo_cell(m, l) is not None for m in MODELS for l in LRS)
    nbeh = sum(beh_cell(m, l, "answer_sycophancy") is not None
               for m in MODELS for l in LRS)
    print(f"wrote {out}  (dpo {ncell}/18, behavioral {nbeh}/18 cells present)")


if __name__ == "__main__":
    main()
